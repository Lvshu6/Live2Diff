import torch
import torch.nn as nn
from einops import rearrange

# class Residual3DBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=(1,3,3), stride=(1,1,1)):
#         super().__init__()
#         self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=(0,1,1))
#         self.norm1 = nn.LayerNorm(out_channels)
#         self.act1 = nn.SiLU()
        
#         self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding=(0,1,1))
#         self.norm2 = nn.LayerNorm(out_channels)
#         self.act2 = nn.SiLU()
        
#         self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         out = self.conv1(x)
#         out = rearrange(out, "b c t h w -> b t h w c")
#         out = self.norm1(out)
#         out = self.act1(out)
#         out = rearrange(out, "b t h w c -> b c t h w")

#         out = self.conv2(out)
#         out = rearrange(out, "b c t h w -> b t h w c")
#         out = self.norm2(out)
#         out = self.act2(out)
#         out = rearrange(out, "b t h w c -> b c t h w")

#         return out + self.skip(x)


# class WanFlowLineAdapter(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 仿照 ResNet50 的 stage 堆叠：3 + 4 + 6 + 3 = 16 blocks
#         # 每个 block 内有两层 conv，总层数接近 50
#         stage_blocks = [3, 4, 6, 3]
#         #5B:48,stage_channels = [128, 256, 512, 1024,2048]  # 通道逐步扩展
#         #14B:[256, 512, 1024, 2048,4096]
#         stage_channels = [256, 512, 1024, 2048,4096]  # 通道逐步扩展

#         blocks = []
#         in_ch = 16
#         for stage_idx, num_blocks in enumerate(stage_blocks):
#             out_ch = stage_channels[stage_idx]
#             for b in range(num_blocks):
#                 blocks.append(Residual3DBlock(in_ch, out_ch))
#                 in_ch = out_ch

#         self.flow_line_blocks = nn.ModuleList(blocks)

#         # 最终投影到 14B:5120,5B:3072
#         self.flow_line_patch_embedding = nn.Conv3d(in_ch, 5120, kernel_size=(1,2,2), stride=(1,2,2))

#     def after_patch_embedding(self, x: torch.Tensor, flow_line_latents: torch.Tensor):
#         for block in self.flow_line_blocks:
#             flow_line_latents = block(flow_line_latents)
#         flow_line_latents = self.flow_line_patch_embedding(flow_line_latents)

#         x[:, :, 1:] += flow_line_latents[:, :, 1:]
#         return x

#     @staticmethod
#     def state_dict_converter():
#         return WanFlowLineAdapterStateDictConverter()



class WanFlowLineAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # 最终投影到 14B:5120,5B:3072
        in_ch = 16
        self.flow_line_patch_embedding = nn.Conv3d(in_ch, 5120, kernel_size=(1,2,2), stride=(1,2,2))

    def after_patch_embedding(self, x: torch.Tensor, flow_line_latents: torch.Tensor):
        # for block in self.flow_line_blocks:
        #     flow_line_latents = block(flow_line_latents)
        flow_line_latents = self.flow_line_patch_embedding(flow_line_latents)

        x[:, :, 1:] += flow_line_latents[:, :, 1:]
        return x

    @staticmethod
    def state_dict_converter():
        return WanFlowLineAdapterStateDictConverter()

class WanFlowLineAdapterStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict

    # def from_civitai(self, state_dict):
    #     state_dict_ = {}
    #     for name, param in state_dict.items():
    #         if name.startswith("flow_line_blocks") or name.startswith("flow_line_patch_embedding"):
    #             state_dict_[name] = param
    #     return state_dict_

    def from_civitai(self, state_dict):
        state_dict_ = {}
        for name, param in state_dict.items():
            if  name.startswith("flow_line_patch_embedding"):
                state_dict_[name] = param
        return state_dict_