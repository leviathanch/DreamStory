# DreamStory Project

This repository is the **official implementation** of [DreamStory (TPAMI 2025)](https://ieeexplore.ieee.org/document/11130395) and [IR-Diffusion](https://arxiv.org/abs/2411.19261).  


# Installation

### Create Conda Environment.
- `conda create -n ds python=3.10`
- `conda activate ds`

### Install PyTorch (select the appropriate CUDA version).
- `pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 xformers --index-url https://download.pytorch.org/whl/cu121`

### Install DreamStory
- `git clone https://github.com/hehuiguo/DreamStory`
- `cd DreamStory`
- `pip install -e .`

### Install GroundedSAM
- `git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git`
- `cd Grounded-Segment-Anything`
- `pip install -e segment_anything`
- `pip install -e GroundingDINO`

# Usage Examples

- `python ./src/DreamStory/pipe/pipe_test.py test --prompts_path="./results/examples/example.json" --output_root="./results/example_debug/" `

The generated image will be saved as ./results/example_debug/output_image_00.png.

##  Citation
🌟 Support Us! If you find this project useful, please consider giving it a ⭐ to help others discover it!

📖 Cite Us! If this project contributes to your research, we would appreciate it if you could cite our paper:
```bibtex
@article{IR_Diffusion,
  title={Improving Multi-Subject Consistency in Open-Domain Image Generation with Isolation and Reposition Attention},
  author={He, Huiguo and Wang, Qiuyue and Zhou, Yuan and Cai, Yuxuan and Chao, Hongyang and Yin, Jian and Yang, Huan},
  journal={arXiv preprint arXiv:2411.19261},
  year={2024}
}

@article{DreamStory,
  title={{DreamStory}: Open-Domain Story Visualization by LLM-Guided Multi-Subject Consistent Diffusion}, 
  author={He, Huiguo and Yang, Huan and Tuo, Zixi and Zhou, Yuan and Wang, Qiuyue and Zhang, Yuhang and Liu, Zeyu and Huang, Wenhao and Chao, Hongyang and Yin, Jian},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2025},
  volume={47},
  number={12},
  pages={11874-11891},
  doi={10.1109/TPAMI.2025.3600149}
}
```