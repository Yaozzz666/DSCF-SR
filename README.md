<!--
 * @Author: Yaozzz666
 * @Date: 2025-03-21 13:49:25
 * @LastEditors: Yaozzz666
 * @LastEditTime: 2025-03-22 11:11:04
 * 
 * Copyright (c) 2025 by ${Yaozzz666}, All Rights Reserved. 
-->
 

<h1 align="center">
  <a href="https://arxiv.org/abs/2504.11271">
    Distillation-Supervised Convolutional Low-Rank Adaptation for Efficient Image Super-Resolution
  </a>
</h1>

<div align="center">
  <a href="https://arxiv.org/abs/2504.11271" style="text-decoration:none;">
    <img src="https://img.shields.io/badge/cs.CV-2504.11271-%23B22222" alt="cs.CV">
  </a>
</div>



## 📖 The Architecture of DSCF Model
<div align=center>
<img src="https://github.com/Yaozzz666/DSCF-SR/blob/main/figs/DSCF_arch_new.png" width="1000px"/> 
</div>
We replace the SPAB module with the proposed SConvLB module and incorporate
ConvLoRA layers into both the pixel shuffle block and its preceding convolutional layer. Spatial Affinity Distillation Loss is calculated
between each feature map.

## 🚀 Updates
* [2025.04.21] ✅ Upload our model on [Hugging Face](https://huggingface.co/sssefe/DSCLoRA) 🤗.
* [2025.04.15] 🎉 Our [paper](https://arxiv.org/abs/2504.11271) is accepted to CVPR 2025 Workshop!
* [2025.03.26] 🏆 Our team won **1st** place in the [NTIRE 2025 Efficient SR Challenge](https://cvlai.net/ntire/2025/). Challenge report is [here](https://arxiv.org/abs/2504.10686).
* [2025.03.21] ✅ Release our code here.
## 🔧 The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:

- Step1: install Pytorch first:
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

- Step2: install other libs via:
```pip install -r requirements.txt```

or take it as a reference based on your original environments.

## ⚡ How to test the model?
1. Run the [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 23
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.


## 🥰 Citation
If our work is useful to you, please use the following BibTeX for citation.

```
@inproceedings{Chai2025DistillationSupervisedCL,
  title={Distillation-Supervised Convolutional Low-Rank Adaptation for Efficient Image Super-Resolution},
  author={Xinning Chai and Yao Zhang and Yuxuan Zhang and Zhengxue Cheng and Yingsheng Qin and Yucai Yang and Li Song},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:277787382}
}
```

## 📜 License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
