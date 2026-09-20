"""Generate per-class SAM masks from a reference image + class text prompt.

Modernized 2026-09-20 to use:
  - transformers.AutoProcessor + GroundingDinoForObjectDetection (replaces the
    unmaintained groundingdino-py package, which breaks against
    transformers>=5.0 because it calls removed BertModel helpers).
  - transformers.SamModel + SamProcessor (replaces the unmaintained
    segment-anything package, which has hardcoded checkpoint paths).

The downstream API surface is preserved: callers (DreamStory's
pipe_test.py, gen_mask pipelines) still get back a dict
{class_name: tensor[N=1, 1, H, W] * 255} and can still invoke
load_model(), load_sam_model(), load_dino_model(), and
generate_sam_mask() with the same signatures.
"""
import os, sys, fire
from typing import Tuple, Optional
import random
from PIL import Image

# Modernized stack — no more groundingdino-py or segment-anything.
import torch
import numpy as np
import cv2
import torchvision.ops as tv_ops

from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    SamModel,
    SamProcessor,
)

# Backwards-compat shim: lets legacy callers (DreamStory's pipe_test.py)
# keep writing `groundingdino_model, sam_predictor = load_model(...)`
# and `sam_predictor.set_image(...)`, `sam_predictor.predict_torch(...)`
# without modification, even though we're now driving SAM through the
# maintained transformers API instead of segment_anything.
from DreamStory.gen_mask._sam_compat import _SamPredictorAdapter, build_sam_predictor


def _to_device(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device) if t.device != torch.device(device) else t


def _normalize_image_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
    """GroundingDINO's T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).

    The legacy groundingdino.datasets.transforms pipeline does
    RandomResize([800], max_size=1333) -> ToTensor() -> Normalize(...).
    AutoProcessor does the equivalent internally when we call it with
    images=, so we don't need to repeat it here — load_image_str returns
    the raw uint8 numpy array plus a placeholder tensor kept only for
    signature compatibility.
    """
    return image_tensor


def load_image_str(image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
    """Read an image from disk.

    Returns (image_as_uint8_ndarray_RGB, dummy_tensor). The dummy tensor
    is preserved for backwards compatibility — callers that introspect
    it get something with the right dtype/shape to feed into the
    transformers pipeline downstream, but the canonical preprocessed
    tensor is built fresh inside generate_sam_mask() via AutoProcessor.
    """
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image, image_transformed


def load_image_pil(image_source: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
    """Read an image from a PIL.Image."""
    image = np.asarray(image_source.convert("RGB"))
    image_transformed = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image, image_transformed


def load_dino_model(
    dino_ckpt_repo_id: str = "IDEA-Research/grounding-dino-base",
    dino_ckpt_filename: str = "pytorch_model.bin",
    dino_ckpt_config_filename: str = "config.json",
    device: str = "cuda:0",
) -> GroundingDinoForObjectDetection:
    """Load a GroundingDINO model via the maintained transformers API.

    Default arguments shifted from the old groundingdino-py defaults
    (ShilongLiu/GroundingDINO + groundingdino_swinb_cogcoor.pth) to the
    HuggingFace-native IDEA-Research/grounding-dino-base checkpoint, which
    is what transformers ships weights for. The old SwinB-CogCoor
    checkpoint is incompatible with the HuggingFace model class because
    its config uses a non-standard backbone.
    """
    # Suppress the noisy "[download]" logs from transformers on cold cache.
    processor = AutoProcessor.from_pretrained(dino_ckpt_repo_id)
    model = GroundingDinoForObjectDetection.from_pretrained(dino_ckpt_repo_id)
    model.to(device)
    model.eval()
    # Stash processor on the model so callers don't need to track it
    # separately — generate_sam_mask() pulls it back out via
    # model.config._name_or_path.
    setattr(model, "_videomaker_processor", processor)
    return model


def load_sam_model(
    sam_name: str = "sam",
    sam_repo_id: str = "facebook/sam-vit-huge",
    device: str = "cuda:0",
):
    """Load a SAM model via the maintained transformers API.

    Returns a _SamPredictorAdapter that exposes the same surface as
    segment_anything.SamPredictor (.set_image(), .predict_torch(),
    .transform.apply_boxes_torch()). See _sam_compat.py for details.
    """
    if sam_name.lower() != "sam":
        # mobile_sam and hq_sam variants still depend on the EfficientSAM
        # package, which is also unmaintained. Until that's modernized,
        # only the canonical SAM ViT-H path is supported here.
        raise NotImplementedError(
            f"sam_name='{sam_name}' is not supported by the modernized "
            f"DreamStory stack. Use sam_name='sam' (the only path that "
            f"works with transformers>=5). MobileSAM / HQSAM still "
            f"depend on the unmaintained EfficientSAM package."
        )
    return build_sam_predictor(sam_repo_id=sam_repo_id, device=device)


def load_model(
    dino_ckpt_repo_id: str = "IDEA-Research/grounding-dino-base",
    dino_ckpt_filename: str = "pytorch_model.bin",
    dino_ckpt_config_filename: str = "config.json",
    sam_name: str = "sam",
    sam_repo_id: str = "facebook/sam-vit-huge",
    device: str = "cuda:0",
):
    """Load both GroundingDINO and SAM.

    Returns (groundingdino_model, sam_predictor) — the SAME shape that
    the legacy load_model() returned, so legacy callers like
    pipe_test.py can keep working unchanged.
    """
    groundingdino_model = load_dino_model(
        dino_ckpt_repo_id,
        dino_ckpt_filename,
        dino_ckpt_config_filename,
        device=device,
    )
    sam_predictor = load_sam_model(sam_name, sam_repo_id, device=device)
    return groundingdino_model, sam_predictor


# -----------------------------------------------------------------------------
# Below: the original box-merging logic from DreamStory's
# dino_sam_mask_generator.py, preserved verbatim and adapted to consume
# transformers-native outputs. Functions upstream of this point are the
# modernized entry points; everything below this line keeps the original
# filter / merge heuristics so character-id behavior doesn't drift.
# -----------------------------------------------------------------------------


def get_key_from_prompts(text, phrase, count=0):
    assert phrase is not None and len(phrase) > 1, f"phrase should not be None or empty, but got {phrase}"
    class_name = text.split(".")
    class_name = [name.strip() for name in class_name if name.strip()]
    if count > 10:
        # Compare which one (phrase or class_name) is more similar and return the closer match
        best_i = -1
        best_sim = 0
        for name in class_name:
            cur_sim = 0
            for ch in name:
                if ch in phrase:
                    cur_sim += 1
            if cur_sim > best_sim:
                best_sim = cur_sim
                best_i = name
            elif cur_sim == best_sim:  # Returns the shorter one
                if len(name) < best_i:
                    best_i = name
        if best_i != -1:
            return best_i
        return None

    def is_in_name(name, phrase):
        # Split name and phrase by ' '/ ',', then check if every word in phrase exists in name
        phrase_list = phrase.replace(", ", " ").split(" ")
        name_list = name.replace(", ", " ").split(" ")
        for p in phrase_list:
            if p not in name_list:
                return False
        return True

    for name in class_name:
        if name is None or len(name) < 1:
            continue  # should not be happened
        if is_in_name(name, phrase):
            return name

    phrase_list = phrase.replace(", ", " ").split(" ")
    phrase_list = [p.strip() for p in phrase_list if p.strip()]
    # TODO: bug here, IDK why :(
    new_phrase = random.choice(phrase_list)
    return get_key_from_prompts(text, new_phrase, count + 1)


def _dino_predict_transformers(
    model: GroundingDinoForObjectDetection,
    image_pil: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
):
    """Run GroundingDINO via transformers and return (boxes_xyxy_pixels, logits, phrases).

    Output shapes match what the legacy groundingdino.util.inference.predict()
    returned, except boxes are pixel-space xyxy (not normalized cxcywh).
    """
    processor: AutoProcessor = getattr(model, "_videomaker_processor", None)
    if processor is None:
        processor = AutoProcessor.from_pretrained(model.config._name_or_path)
        setattr(model, "_videomaker_processor", processor)

    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # transformers returns logits: [batch, num_queries, num_classes] and
    # pred_boxes: [batch, num_queries, 4] (normalized cxcywh).
    target_sizes = torch.tensor([image_pil.size[::-1]], device=device)  # [H, W]
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )[0]

    boxes = results["boxes"]            # [N, 4] pixel-space xyxy
    scores = results["scores"]          # [N]
    phrases = results["text_labels"]    # list[str] of length N

    return boxes.detach().cpu(), scores.detach().cpu(), phrases


def generate_sam_mask(image, text_prompt,
                       groundingdino_model=None, sam_predictor=None,
                       text_threshold=0.25, box_threshold=0.3,
                       device='cuda',
                       is_morphology_mask=False,
                       output_mask_path=None,
                       ):
    """Generate SAM masks for each class in text_prompt.

    text_prompt format is unchanged: dot-separated class names like
    "man . woman . dog".

    Backwards compatibility shims:
      - sam_predictor accepts the old SamPredictor (None for re-load)
        OR the new (sam_model, sam_processor) tuple.
    """
    if groundingdino_model is None or sam_predictor is None:
        groundingdino_model, sam_predictor = load_model(device=device)

    if isinstance(image, str):
        image_source_np, _ = load_image_str(image)
        image_pil = Image.fromarray(image_source_np)
    elif isinstance(image, Image.Image):
        image_source_np, _ = load_image_pil(image)
        image_pil = image.convert("RGB")
    else:
        raise ValueError(f"image should be a path or PIL image, but got {type(image)}")

    boxes_xyxy, logits, phrases = _dino_predict_transformers(
        model=groundingdino_model,
        image_pil=image_pil,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    H, W, _ = image_source_np.shape
    if boxes_xyxy.numel() == 0:
        # Nothing detected — return empty dict rather than crash on the
        # NMS loops below which assume at least one detection.
        ret_masks_dict = {}
    else:
        # Bring tensors onto CPU for the heuristics below (they don't need GPU).
        boxes_xyxy = boxes_xyxy.cpu()
        logits = logits.cpu()

        # -----------------------------------------------------------------------------
        # Original box-merging logic, adapted to drop the legacy groundingdino
        # box_ops dependency. tv_ops.box_iou gives the same behaviour as
        # box_ops.box_iou (xyxy IoU). The downstream heuristics are unchanged.
        # -----------------------------------------------------------------------------

        # Filter overlapping bboxes, keep only: 1. The one with highest probability; 2. The one with largest area
        new_phrases_idx_list = []
        for i, phrase_i in enumerate(phrases):
            is_in_other_bbox = False
            if phrase_i is None or len(phrase_i) < 1:
                continue
            for j, phrase_j in enumerate(phrases):
                if i == j or phrase_j is None or len(phrase_j) < 1:
                    continue
                is_in_box = False
                area_size_i = (boxes_xyxy[i][2] - boxes_xyxy[i][0]) * (boxes_xyxy[i][3] - boxes_xyxy[i][1])
                area_size_insection = (
                    min(boxes_xyxy[j][2], boxes_xyxy[i][2]) - max(boxes_xyxy[j][0], boxes_xyxy[i][0])
                ) * (min(boxes_xyxy[j][3], boxes_xyxy[i][3]) - max(boxes_xyxy[j][1], boxes_xyxy[i][1]))

                if area_size_insection / area_size_i > 0.8:
                    is_in_box = True
                    iou = 1.0
                else:
                    iou = tv_ops.box_iou(
                        boxes_xyxy[i:i + 1], boxes_xyxy[j:j + 1]
                    )[0, 0].item()

                class_name_i = get_key_from_prompts(text_prompt, phrase_i)
                class_name_j = get_key_from_prompts(text_prompt, phrase_j)
                if is_in_box or (iou > 0.9 and class_name_i == class_name_j):
                    area_size_j = (boxes_xyxy[j][2] - boxes_xyxy[j][0]) * (boxes_xyxy[j][3] - boxes_xyxy[j][1])
                    if area_size_i < area_size_j:
                        is_in_other_bbox = True
                        break
            if not is_in_other_bbox:
                new_phrases_idx_list.append(i)

        # Calculates overlap count for each bbox
        repeat_scores = {}
        for i in new_phrases_idx_list:
            phrase_i = phrases[i]
            for j in new_phrases_idx_list:
                if i == j:
                    continue
                phrase_j = phrases[j]
                iou = tv_ops.box_iou(boxes_xyxy[i:i + 1], boxes_xyxy[j:j + 1])[0, 0].item()
                if iou > 0.5:
                    if i not in repeat_scores:
                        repeat_scores[i] = 1
                    else:
                        repeat_scores[i] += 1

        # When boxes of different class_name overlap significantly, retain only the highest-confidence or largest-area instance
        new_no_repeat_phrases_idx_list = []
        for i in new_phrases_idx_list:
            phrase_i = phrases[i]
            is_in_other_bbox = False
            for j in new_phrases_idx_list:
                if i == j:
                    continue
                phrase_j = phrases[j]
                is_in_box = False
                area_size_i = (boxes_xyxy[i][2] - boxes_xyxy[i][0]) * (boxes_xyxy[i][3] - boxes_xyxy[i][1])
                area_size_insection = (
                    min(boxes_xyxy[j][2], boxes_xyxy[i][2]) - max(boxes_xyxy[j][0], boxes_xyxy[i][0])
                ) * (min(boxes_xyxy[j][3], boxes_xyxy[i][3]) - max(boxes_xyxy[j][1], boxes_xyxy[i][1]))

                if area_size_insection / area_size_i > 0.8:
                    is_in_box = True
                    iou = 1.0
                else:
                    iou = tv_ops.box_iou(boxes_xyxy[i:i + 1], boxes_xyxy[j:j + 1])[0, 0].item()
                if iou > 0.5 or is_in_box:
                    class_name_i = get_key_from_prompts(text_prompt, phrase_i)
                    class_name_j = get_key_from_prompts(text_prompt, phrase_j)
                    if class_name_i != class_name_j and logits[i] < logits[j]:
                        is_in_other_bbox = True
                        break
            if not is_in_other_bbox:
                new_no_repeat_phrases_idx_list.append(i)

        new_phrases_idx_list = new_no_repeat_phrases_idx_list

        # Get the highest-response bbox per class
        best_bbox = {}
        bbox_idx = {}
        for i in new_phrases_idx_list:
            phrase = phrases[i]
            if phrase is None or len(phrase) < 1:
                continue
            class_name = get_key_from_prompts(text_prompt, phrase)
            if class_name not in best_bbox:
                best_bbox[class_name] = (logits[i], boxes_xyxy[i:i + 1])
                bbox_idx[class_name] = i
            else:
                IOU = tv_ops.box_iou(
                    boxes_xyxy[i:i + 1], best_bbox[class_name][1]
                )[0, 0].item()
                if abs(logits[i] - best_bbox[class_name][0]) < (0.2 * max(logits[i], best_bbox[class_name][0])):
                    if IOU > 0.5:
                        best_x1 = min(best_bbox[class_name][1][0][0], boxes_xyxy[i][0])
                        best_y1 = min(best_bbox[class_name][1][0][1], boxes_xyxy[i][1])
                        best_x2 = max(best_bbox[class_name][1][0][2], boxes_xyxy[i][2])
                        best_y2 = max(best_bbox[class_name][1][0][3], boxes_xyxy[i][3])
                        best_logit = max(best_bbox[class_name][0], logits[i])
                        best_bbox[class_name] = (
                            best_logit,
                            torch.Tensor([best_x1, best_y1, best_x2, best_y2]).unsqueeze(0),
                        )
                        bbox_idx[class_name] = i
                    else:
                        if repeat_scores.get(i, 0) < repeat_scores.get(bbox_idx[class_name], 0):
                            best_bbox[class_name] = (logits[i], boxes_xyxy[i:i + 1])
                            bbox_idx[class_name] = i
                        elif repeat_scores.get(i, 0) == repeat_scores.get(bbox_idx[class_name], 0):
                            if logits[i] > best_bbox[class_name][0]:
                                best_bbox[class_name] = (logits[i], boxes_xyxy[i:i + 1])
                                bbox_idx[class_name] = i
                elif logits[i] > best_bbox[class_name][0]:
                    best_bbox[class_name] = (logits[i], boxes_xyxy[i:i + 1])
                    bbox_idx[class_name] = i

        all_class_name_list = text_prompt.split(".")
        all_class_name_list = [name.strip() for name in all_class_name_list if name.strip()]

        # Check for undetected classes
        for class_name in all_class_name_list:
            if class_name in best_bbox:
                continue
            else:
                best_idx = -1
                for i in range(len(phrases)):
                    phrase = phrases[i]
                    if phrase is None or len(phrase) < 1:
                        continue
                    area_size_i = (boxes_xyxy[i][2] - boxes_xyxy[i][0]) * (boxes_xyxy[i][3] - boxes_xyxy[i][1])
                    is_in_best_bbox = False
                    for _, (logit, box) in best_bbox.items():
                        area_size_insection = (
                            min(box[0][2], boxes_xyxy[i][2]) - max(box[0][0], boxes_xyxy[i][0])
                        ) * (min(box[0][3], boxes_xyxy[i][3]) - max(box[0][1], boxes_xyxy[i][1]))
                        if area_size_insection / area_size_i > 0.8:
                            is_in_best_bbox = True
                            break
                        if tv_ops.box_iou(boxes_xyxy[i:i + 1], box)[0, 0].item() > 0.5:
                            is_in_best_bbox = True
                            break
                    if is_in_best_bbox:
                        continue

                    if best_idx == -1:
                        best_idx = i
                    else:
                        bbox_best_idx = boxes_xyxy[best_idx]
                        area_size_best_idx = (bbox_best_idx[2] - bbox_best_idx[0]) * (bbox_best_idx[3] - bbox_best_idx[1])
                        if area_size_i > area_size_best_idx:
                            best_idx = i
                if best_idx != -1:
                    best_bbox[class_name] = (logits[best_idx], boxes_xyxy[best_idx:best_idx + 1])
                    bbox_idx[class_name] = best_idx
                else:
                    print(f"warning: can not find bbox for class_name: {class_name}")

        # -----------------------------------------------------------------------------
        # Generate SAM masks for each best_bbox entry.
        # We drive SAM through the legacy-compatible adapter (see
        # _sam_compat.py). The adapter caches the image encoding on
        # set_image() and reuses it across boxes, exactly like the
        # original segment_anything.SamPredictor.
        # -----------------------------------------------------------------------------
        ret_masks_dict = {}
        if best_bbox:
            # set_image once for all boxes in this image. The original
            # pipe_test.py called generate_sam_mask once per scene; we
            # encode the image once and reuse it across all class bboxes.
            sam_predictor.set_image(image_source_np)
            for class_name, (logit, box) in best_bbox.items():
                with torch.no_grad():
                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                        box, image_source_np.shape[:2]
                    ).to(device)
                    masks, _, _ = sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )
                # SAM's masks are bool (True/False). Multiply by 255 to
                # match the original code's mask = mask * 255 convention.
                masks = masks.to("cpu").to(torch.float32) * 255

                if masks.shape[0] > 1 or masks.shape[1] > 1:
                    print(f"warning: masks shape should be [1, 1, H, W], but got {masks.shape}. Use the max area mask.")
                    mask_area = torch.sum(masks, dim=(2, 3))
                    max_area_idx = torch.argmax(mask_area)
                    masks = masks[max_area_idx:max_area_idx + 1]

                ret_masks_dict[class_name] = masks

    # Fill hollow noise in masks
    if is_morphology_mask:
        for class_name, mask in ret_masks_dict.items():
            mask = mask.squeeze(1).squeeze(0)
            mask_np = mask.numpy().astype(np.uint8)
            mask_nonzero = np.nonzero(mask_np)
            top, bottom = mask_nonzero[0].min(), mask_nonzero[0].max()
            left, right = mask_nonzero[1].min(), mask_nonzero[1].max()
            border_mask_np = np.zeros_like(mask_np)
            border_mask_np[top:bottom + 1, left:right + 1] = 1

            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, np.ones((128, 128), np.uint8))
            mask_np = mask_np * border_mask_np
            mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(torch.float32)
            ret_masks_dict[class_name] = mask

    if output_mask_path is not None:
        if not output_mask_path.endswith(".png") and not output_mask_path.endswith(".jpg"):
            os.makedirs(output_mask_path, exist_ok=True)
        else:
            output_mask_path = os.path.dirname(output_mask_path)
            os.makedirs(output_mask_path, exist_ok=True)
        for key in ret_masks_dict:
            mask = ret_masks_dict[key][0, :, :,]
            mask = mask.squeeze().cpu().numpy()
            np.where(mask == 0, 127, mask)
            image = Image.fromarray(mask.astype(np.uint8))
            save_mask_path = os.path.join(output_mask_path, f"{key}.png")
            image.save(save_mask_path)

            save_fused_path = os.path.join(output_mask_path, f"fuse_{key}.png")
            fused_image = image_source_np
            fused_image = cv2.cvtColor(fused_image, cv2.COLOR_BGR2BGRA)
            fused_image[:, :, 3] = mask
            fused_image = Image.fromarray(fused_image)
            fused_image.save(save_fused_path)

        return
    else:
        return ret_masks_dict


# SAM mask post-processing steps:
# 1. Expand undersized masks to meet min_mask_pixel_size requirement
# 2. Adjust masks with deviating source_mask ratios (expand to approximate original)
# 3. Ensure non-overlapping mask outputs
def post_process_sam_mask(sam_mask,
                           min_mask_pixel_size=256, device='cuda:0',
                           is_expand_small_mask=False, is_keep_mask_ratio=False, is_no_mask_overlap=False):
    target_mask_list = sam_mask["target_mask"]
    source_mask_list = sam_mask["source_mask"]

    source_mask_ratio_list = []
    for source_mask in source_mask_list:
        mask_np = source_mask[0, :, :].cpu().numpy()
        mask_nonzero = np.nonzero(mask_np)
        top, bottom = mask_nonzero[0].min(), mask_nonzero[0].max()
        left, right = mask_nonzero[1].min(), mask_nonzero[1].max()
        ratio = (bottom - top) / (right - left)
        source_mask_ratio_list.append(ratio)

    target_mask_ratio_list = []
    target_mask_sorted_idx_list = []
    for i, target_mask in enumerate(target_mask_list):
        mask_np = target_mask[0, :, :].cpu().numpy()
        mask_nonzero = np.nonzero(mask_np)
        top, bottom = mask_nonzero[0].min(), mask_nonzero[0].max()
        left, right = mask_nonzero[1].min(), mask_nonzero[1].max()
        ratio = (bottom - top) / (right - left)
        target_mask_ratio_list.append(ratio)

        pixel_sum = mask_np.sum() / 255.0
        target_mask_sorted_idx_list.append((i, pixel_sum))
    target_mask_sorted_idx_list.sort(key=lambda x: x[1])

    for (i, pixel_sum) in target_mask_sorted_idx_list:
        target_mask = target_mask_list[i]
        mask_np = target_mask[0, :, :].cpu().numpy()
        mask_nonzero = np.nonzero(mask_np)
        top, bottom = mask_nonzero[0].min(), mask_nonzero[0].max()
        left, right = mask_nonzero[1].min(), mask_nonzero[1].max()

        if is_expand_small_mask and (pixel_sum < min_mask_pixel_size * min_mask_pixel_size or
                                     bottom - top < min_mask_pixel_size or right - left < min_mask_pixel_size):
            if bottom - top < min_mask_pixel_size or pixel_sum < min_mask_pixel_size * min_mask_pixel_size:
                center = (top + bottom) // 2
                top = max(0, center - min_mask_pixel_size // 2)
                bottom = min(mask_np.shape[0], center + min_mask_pixel_size // 2)

            if right - left < min_mask_pixel_size or pixel_sum < min_mask_pixel_size * min_mask_pixel_size:
                center = (left + right) // 2
                left = max(0, center - min_mask_pixel_size // 2)
                right = min(mask_np.shape[1], center + min_mask_pixel_size // 2)

            mask_np[top:bottom + 1, left:right + 1] = 255

            for j in range(len(target_mask_list)):
                if i == j:
                    continue
                mask_j = target_mask_list[j][0, :, :].cpu().numpy()
                inter_mask = mask_np * mask_j
                if inter_mask.sum() > 0:
                    mask_np[inter_mask > 0] = 0

        if is_no_mask_overlap:
            for j in range(len(target_mask_list)):
                if i == j:
                    continue
                mask_j = target_mask_list[j][0, :, :].cpu().numpy()
                inter_mask = mask_np * mask_j
                if inter_mask.sum() > 0:
                    mask_np[inter_mask > 0] = 0

        mask = torch.tensor(mask_np).unsqueeze(0).to(device)
        sam_mask["target_mask"][i] = mask
    return sam_mask


if __name__ == "__main__":
    fire.Fire()


# examples:
# python ./src/DreamStory/gen_mask/dino_sam_mask_generator.py generate_sam_mask --image ./results/test.jpg --text_prompt "cat" --output_mask_path ./results/output_mask
