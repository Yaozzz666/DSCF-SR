<!--
 * @Author: Yaozzz666
 * @Date: 2025-03-21 13:49:25
 * @LastEditors: Yaozzz666
 * @LastEditTime: 2025-03-22 11:11:04
 * 
 * Copyright (c) 2025 by ${Yaozzz666}, All Rights Reserved. 
-->
# [NTIRE 2025 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

## Distillation Supervised ConvLora Finetuning for SR

<div align=center>
<img src="https://github.com/Yaozzz666/DSCF-SR/blob/main/figs/DSCF_arch.png" width="1000px"/> 
</div>

- An overview of our DSCF-SR

## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:

- Step1: install Pytorch first:
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

- Step2: install other libs via:
```pip install -r requirements.txt```

or take it as a reference based on your original environments.

## How to test the model?
1. Run the [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 23
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.

## How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team23_DSCF import DSCF
    from fvcore.nn import FlopCountAnalysis

    model = DSCF(3,3,feature_channels=26,upscale=4)
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    # The FLOPs calculation in previous NTIRE_ESR Challenge
    # flops = get_model_flops(model, input_dim, False)
    # flops = flops / 10 ** 9
    # print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    # fvcore is used in NTIRE2025_ESR for FLOPs calculation
    input_fake = torch.rand(1, 3, 256, 256).to(device)
    flops = FlopCountAnalysis(model, input_fake).total()
    flops = flops/10**9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
