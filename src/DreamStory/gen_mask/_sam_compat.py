"""Compatibility shim so legacy callers can still write:

    groundingdino_model, sam_predictor = load_model(...)
    sam_predictor.set_image(image)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(...)
    masks, _, _ = sam_predictor.predict_torch(point_coords=None, ...)

without knowing that we now drive SAM through the maintained
transformers.SamModel + SamProcessor stack instead of the
segment_anything.SamPredictor.

This is the smallest possible change that keeps DreamStory's
pipe_test.py working while the rest of dino_sam_mask_generator.py
has been modernized.
"""
from typing import Optional, List
import torch
import numpy as np
from PIL import Image

from transformers import SamModel, SamProcessor


class _BoxTransformShim:
    """Mimics segment_anything.utils.transforms.ResizeLongestSide.

    SamProcessor doesn't expose its box-resize helper directly, so we
    recompute it the same way SAM-original does: scale so the longest
    side equals the model's input size (1024 for ViT-H), keep aspect
    ratio. Coordinates in target frame are then (x, y) of original
    corners times scale.
    """

    def __init__(self, target_length: int = 1024):
        self.target_length = target_length

    def apply_coords(self, coords: np.ndarray, original_size: tuple) -> np.ndarray:
        old_h, old_w = original_size
        scale = self.target_length * 1.0 / max(old_h, old_w)
        new_h = int(round(old_h * scale))
        new_w = int(round(old_w * scale))
        coords = coords.copy()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: tuple) -> np.ndarray:
        # boxes: [N, 4] xyxy in original pixel coords
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_boxes_torch(self, boxes: torch.Tensor, original_size: tuple) -> torch.Tensor:
        np_in = boxes.detach().cpu().numpy()
        np_out = self.apply_boxes(np_in, original_size)
        return torch.from_numpy(np_out).to(boxes.dtype)


class _SamPredictorAdapter:
    """Drop-in replacement for segment_anything.SamPredictor.

    Public surface that legacy callers exercise:
      - .model                       (the underlying SamModel)
      - .transform                   (a ResizeLongestSide shim)
      - .set_image(image_np)         (encode once; reuse for many boxes)
      - .predict_torch(point_coords, point_labels, boxes, multimask_output)
                                   (return (masks, scores, logits) like SAM)
    """

    def __init__(self, sam_model: SamModel, sam_processor: SamProcessor, device: str = "cuda"):
        self.model = sam_model
        self.processor = sam_processor
        self.device = device
        self.target_length = sam_model.config.vision_config.image_size  # 1024 for ViT-H
        self.transform = _BoxTransformShim(target_length=self.target_length)
        self._image_pil: Optional[Image.Image] = None
        self._pixel_values = None
        self._original_size: Optional[tuple] = None

    def set_image(self, image):
        """image can be a numpy ndarray (HxWxC uint8) OR a PIL.Image."""
        if isinstance(image, np.ndarray):
            self._image_pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            self._image_pil = image.convert("RGB")
        else:
            raise TypeError(f"set_image expects np.ndarray or PIL.Image, got {type(image)}")
        self._original_size = (self._image_pil.height, self._image_pil.width)
        # Pre-compute the resized+normalized pixel_values once; we re-use
        # them on every predict_torch call. This mirrors the original
        # SamPredictor's encoder caching.
        with torch.no_grad():
            pixel_values = self.processor.image_processor(
                images=[self._image_pil], return_tensors="pt"
            )["pixel_values"].to(self.device)
        self._pixel_values = pixel_values

    def predict_torch(
        self,
        point_coords,
        point_labels,
        boxes: torch.Tensor,
        multimask_output: bool = False,
    ):
        """Mirrors segment_anything.SamPredictor.predict_torch.

        Returns:
          masks:    Tensor[N, 1, H_orig, W_orig]  (binary 0/1, NOT *255)
          scores:   Tensor[N, 1]
          logits:   Tensor[N, 1, 256, 256]
        """
        if self._pixel_values is None:
            raise RuntimeError("Call set_image(image) before predict_torch")

        # boxes: Tensor[N, 4] in original pixel coords (xyxy)
        # SamProcessor wants input_boxes as a list[list[list[float]]] —
        # outer list is "batches", middle list is "boxes per image",
        # inner list is the 4 coordinates. We have one image and N boxes.
        boxes_list = boxes.detach().cpu().numpy().tolist()
        inputs = self.processor(
            images=self._image_pil,
            input_boxes=[boxes_list],
            return_tensors="pt",
        ).to(self.device)
        # Override with our cached pixel_values so we don't re-encode.
        inputs["pixel_values"] = self._pixel_values

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                multimask_output=multimask_output,
            )

        # Post-process masks to original image size.
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # [N, num_masks, H_orig, W_orig]
        scores = outputs.iou_scores.cpu()  # [N, num_masks]
        logits = outputs.mask_decoder_logits.cpu() if hasattr(outputs, "mask_decoder_logits") else torch.zeros(masks.shape[0], masks.shape[1], 256, 256)

        # SAM's original predict_torch returned masks as boolean tensor.
        # The DreamStory code downstream multiplies by 255, so bool->255
        # works either way (True*255 = 255, False*255 = 0).
        masks = masks.to(torch.bool)
        # Down-select to a single mask per box if multimask_output was False
        if not multimask_output and masks.shape[1] > 1:
            masks = masks[:, :1, :, :]
            scores = scores[:, :1]
            logits = logits[:, :1, :, :]
        return masks, scores, logits


def build_sam_predictor(sam_repo_id: str = "facebook/sam-vit-huge", device: str = "cuda") -> _SamPredictorAdapter:
    """Build the SAM adapter. Convenience for callers that just want a SAM."""
    sam_model = SamModel.from_pretrained(sam_repo_id)
    sam_processor = SamProcessor.from_pretrained(sam_repo_id)
    sam_model.to(device).eval()
    return _SamPredictorAdapter(sam_model, sam_processor, device=device)
