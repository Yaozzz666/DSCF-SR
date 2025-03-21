from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
# from IPython import embed

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Lora_Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # print("in init")
        # embed()
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        # Freeze the bias
        # if self.bias is not None:
        #     self.bias.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True): # True for train and False for eval
 
        nn.Conv2d.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            # print("test")
            # embed()
            if self.merge_weights and not self.merged:
                # print("merging")
                # embed()
                # Merge the weights and mark it
                self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        # print(f"LoRA merged status: {self.merged}")
        if self.r > 0 and not self.merged:
            # print(f"lora_A: {self.lora_A}")
            # print(f"lora_B: {self.lora_B}")
            # print(f"LoRA contribution: {(self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling}")
            return F.conv2d(
                x, 
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        
        return nn.Conv2d.forward(self, x)
def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
  

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)

    def forward(self, x):
        out = self.eval_conv(x)
        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out 


class SPAB1(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB1, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        # self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        # out = out3 * sim_att
        # return out, out1, sim_att
        return out, out1, out2,out3


class DSCF(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 feature_channels=26,
                 upscale=4,
                 bias=True,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)
                 ):
        super(DSCF, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
        self.block_1 = SPAB1(feature_channels, bias=bias)
        self.block_2 = SPAB1(feature_channels, bias=bias)
        self.block_3 = SPAB1(feature_channels, bias=bias)
        self.block_4 = SPAB1(feature_channels, bias=bias)
        self.block_5 = SPAB1(feature_channels, bias=bias)
        self.block_6 = SPAB1(feature_channels, bias=bias)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=2, s=1)
        
        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)
        
        # 指定需要替换 LoRA 层的子模块名称
        # desired_submodules = ["conv_1.eval_conv",
        #                       "block_1.c1_r.eval_conv","block_1.c2_r.eval_conv","block_1.c3_r.eval_conv",
        #                       "block_2.c1_r.eval_conv","block_2.c2_r.eval_conv","block_2.c3_r.eval_conv",
        #                       "block_3.c1_r.eval_conv","block_3.c2_r.eval_conv","block_3.c3_r.eval_conv",
        #                       "block_4.c1_r.eval_conv","block_4.c2_r.eval_conv","block_4.c3_r.eval_conv",
        #                       "block_5.c1_r.eval_conv","block_5.c2_r.eval_conv","block_5.c3_r.eval_conv",
        #                       "block_6.c1_r.eval_conv","block_6.c2_r.eval_conv","block_6.c3_r.eval_conv",
        #                       "conv_2.eval_conv",
        #                       "conv_cat", 
        #                       "upsampler.0"]
        
        # desired_submodules = ["conv_2.eval_conv","upsampler.0"]
        # # 替换需要 LoRA 处理的层
        # self.replace_layers(desired_submodules)
        
        # self.mark_only_lora_as_trainable(bias='none')
        # 分层LoRA配置字典（模块名: (r, lora_alpha)）
        self.lora_config = {
            # 高频重建核心层 (最高优先级)
            "conv_2.eval_conv": (8, 16),  # 最大秩
            "upsampler.0": (8, 16),       # 高秩
            
            # 中间处理层 (梯度传播关键路径)
            **{f"block_{i}.c{j}_r.eval_conv": (2, 4) 
            for i in [2,3,4,5]          # block_2到block_5
            for j in [1,2,3]},          # 每个block的三个卷积
            
            # 首尾层 (适度调整)
            "block_1.c1_r.eval_conv": (2, 4),
            "block_1.c2_r.eval_conv": (2, 4),
            "block_1.c3_r.eval_conv": (2, 4),
            "block_6.c1_r.eval_conv": (2, 4),
            "block_6.c2_r.eval_conv": (2, 4),
            "block_6.c3_r.eval_conv": (2, 4),
        }
        
        # 替换需要 LoRA 处理的层
        self.replace_layers_with_strategy()
        
        # 冻结非LoRA参数
        self.mark_only_lora_as_trainable(bias='none')
        # self.cuda()(torch.randn(1, 3, 256, 256).cuda())
        # self.eval().cuda()
        self.cuda()
        input_tensor = torch.randn(1, 3, 256, 256).cuda()
        output = self(input_tensor)
        # 确保 LoRA 层参数可训练
        # print("可训练参数:")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.shape}")
                

    def replace_layers_with_strategy(self):
        """根据分层策略替换卷积层"""
        for full_name, (r, alpha) in self.lora_config.items():
            parent, child_name = self._get_parent_and_child(full_name)
            if parent is None:
                # print(f"⚠️ Skip {full_name}: module not found")
                continue
                
            original_conv = getattr(parent, child_name, None)
            if not isinstance(original_conv, nn.Conv2d):
                # print(f"⚠️ {full_name} is not Conv2d (found {type(original_conv)})")
                continue
                
            # 动态设置参数
            new_layer = Lora_Conv2d(
                in_channels=original_conv.in_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size[0],
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
                r=r,  # 动态设置秩
                lora_alpha=alpha  # 动态设置缩放系数
            )
            
            # 继承原始权重
            with torch.no_grad():
                new_layer.weight.copy_(original_conv.weight)
                if original_conv.bias is not None:
                    new_layer.bias.copy_(original_conv.bias)
            
            setattr(parent, child_name, new_layer)
            # print(f"✅ {full_name} => r={r}, alpha={alpha}")

    def _get_parent_and_child(self, module_name):
        """
        获取模块的父级模块和子模块名称
        例如：
        module_name = "block_5.c1_r.eval_conv"
        则返回 (model.block_5.c1_r, "eval_conv")
        """
        parts = module_name.split(".")
        parent = self
        for part in parts[:-1]:  # 遍历到倒数第二个
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                return None, None  # 没找到路径
        return parent, parts[-1]  # 返回父模块和子模块名称

    # def replace_layers(self, desired_submodules):
    #     """
    #     遍历模型的子模块，将符合条件的层替换为 Lora_Conv2d
    #     """
    #     # 替换conv_layer
    #     for name, module in self._modules.items():
    #         if name in desired_submodules:
    #             print('--------------------self._modules.items--------------------------')
    #             print(name)
    #         if isinstance(module, nn.Conv2d):
    #             print(f"Replacing {name} with Lora_Conv2d")
    #             setattr(self, name, Lora_Conv2d(
    #                 module.in_channels,
    #                 module.out_channels,
    #                 kernel_size=module.kernel_size[0],
    #                 stride=module.stride,
    #                 padding=module.padding,
    #                 bias=True,
    #                 r=2,
    #                 lora_alpha=2
    #             ))
                                    
    def mark_only_lora_as_trainable(self, bias: str = 'none'):
        """
        只训练 LoRA 相关参数，而冻结所有其他参数。
        
        参数:
        - bias: 'none' (不训练 bias), 'all' (训练所有 bias), 'lora_only' (只训练 LoRA 层的 bias)
        """
        # 冻结所有非 LoRA 参数
        # for n, p in self.named_parameters():
        #     if 'lora_' not in n:
        #         p.requires_grad = False
        for n, p in self.named_parameters():
            if 'lora_' not in n:  
                p.requires_grad = False  # 冻结非 LoRA 参数
            else:
                p.requires_grad = True  # 解冻 LoRA 参数

        if bias == 'none':
            return
        elif bias == 'all':
            for n, p in self.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        elif bias == 'lora_only':
            for m in self.modules():
                if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError(f"未知 bias 选项: {bias}")
    def forward(self, x, return_features=False):
        features = []
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out_b1, out_b1_1, out_b1_2, out_b1_3 = self.block_1(out_feature)
        out_b2, out_b2_1, out_b2_2, out_b2_3 = self.block_2(out_b1)
        out_b3, out_b3_1, out_b3_2, out_b3_3 = self.block_3(out_b2)

        out_b4, _, _, _ = self.block_4(out_b3)
        out_b5, _, _, _ = self.block_5(out_b4)
        out_b6, out_b5_2, _, _ = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)
        
        features.append(out_b1_1)
        features.append(out_b1_2)
        features.append(out_b1_3)
        features.append(out_b2_1)
        features.append(out_b2_2)
        features.append(out_b2_3)
        features.append(out_b3_1)
        features.append(out_b3_2)
        features.append(out_b3_3)
 

        if return_features:
            return output, features  # Return output and intermediate features
        return output
    
