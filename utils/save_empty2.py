import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from einops import rearrange
from safetensors.torch import save_file as save_safetensors
from safetensors.torch import load_file as load_safetensors
from diffsynth.models.wan_video_flow_line_adapter import WanFlowLineAdapter

# -------------------------- 保存函数（支持 BFloat16） --------------------------
def save_state_dict(
    state_dict: dict,
    file_path: str,
    torch_dtype: Optional[torch.dtype] = None,
    device: str = "cpu"
):
    """
    保存权重为 .safetensors 或 .bin 格式，支持指定 dtype（如 bfloat16）
    Args:
        state_dict: 模型权重字典（model.state_dict()）
        file_path: 保存路径（以 .safetensors 或 .bin 结尾）
        torch_dtype: 可选，强制转换权重 dtype（如 torch.bfloat16）
        device: 可选，强制迁移权重到指定设备
    """
    # 1. 预处理：转换dtype和设备
    processed_state = {}
    for k, v in state_dict.items():
        # 转换为指定dtype（核心：bfloat16 转换）
        if torch_dtype is not None:
            # 避免 inplace 操作，先复制再转换
            v = v.clone().to(dtype=torch_dtype)
        # 迁移到指定设备
        if device is not None:
            v = v.to(device=device)
        processed_state[k] = v

    # 2. 按后缀选择保存格式
    if file_path.endswith(".safetensors"):
        save_safetensors(processed_state, file_path)
        print(f"权重已以 {torch_dtype} 格式保存为 safetensors：{file_path}")
    elif file_path.endswith(".bin"):
        torch.save(processed_state, file_path)
        print(f"权重已以 {torch_dtype} 格式保存为 bin：{file_path}")
    else:
        raise ValueError("文件后缀必须是 .safetensors 或 .bin")


# -------------------------- 加载函数（兼容 BFloat16） --------------------------
def load_state_dict(
    file_path: str,
    torch_dtype: Optional[torch.dtype] = None,
    device: str = "cpu"
):
    """加载 .safetensors 或 .bin 格式的权重，支持指定加载后的 dtype"""
    if file_path.endswith(".safetensors"):
        # 加载时指定设备，再转换dtype
        state_dict = load_safetensors(file_path, device=device)
        if torch_dtype is not None:
            state_dict = {k: v.clone().to(dtype=torch_dtype) for k, v in state_dict.items()}
        return state_dict
    elif file_path.endswith(".bin"):
        # map_location 同时指定设备和dtype
        map_location = torch.device(device)
        state_dict = torch.load(
            file_path,
            map_location=map_location,
            weights_only=True  # 安全加载，避免恶意代码
        )
        if torch_dtype is not None:
            state_dict = {k: v.clone().to(dtype=torch_dtype) for k, v in state_dict.items()}
        return state_dict
    else:
        raise ValueError("文件后缀必须是 .safetensors 或 .bin")


# -------------------------- 示例：以 BFloat16 保存/加载模型 --------------------------
if __name__ == "__main__":
    # 1. 创建模型并初始化
    model = WanFlowLineAdapter()
    print("原始模型权重 dtype：", model.flow_line_patch_embedding.weight.dtype)  # 默认 float32

    # 2. 核心：以 BFloat16 格式保存为 safetensors（推荐）
    safetensors_bf16_path = "models/FlowLineAdapter/flow_line_adapter_ablation.safetensors"
    save_state_dict(
        state_dict=model.state_dict(),
        file_path=safetensors_bf16_path,
        torch_dtype=torch.bfloat16,  # 指定 BFloat16
        device="cpu"
    )

    # # 3. 以 BFloat16 格式保存为 bin
    # bin_bf16_path = "flow_line_adapter_bf16.bin"
    # save_state_dict(
    #     state_dict=model.state_dict(),
    #     file_path=bin_bf16_path,
    #     torch_dtype=torch.bfloat16,  # 指定 BFloat16
    #     device="cpu"
    # )

    # 4. 关键修复：先将模型转换为 BFloat16，再加载权重
    model = model.to(dtype=torch.bfloat16)  # 核心：修改模型本身的dtype
    print("模型转换为 BFloat16 后的 dtype：", model.flow_line_patch_embedding.weight.dtype)  # 输出 bfloat16

    # 加载 BFloat16 权重并验证 dtype
    loaded_bf16_safe_dict = load_state_dict(
        safetensors_bf16_path,
        torch_dtype=torch.bfloat16,  # 加载后保持 BFloat16
        device="cpu"
    )


    # 加载到模型
    model.load_state_dict(loaded_bf16_safe_dict)
    print("加载后模型权重 dtype：", model.flow_line_patch_embedding.bias.dtype)  # 输出 torch.bfloat16
    print("BFloat16 Safetensors 权重加载成功")

    # # 5. 加载 BFloat16 bin 权重并验证
    # loaded_bf16_bin_dict = load_state_dict(
    #     bin_bf16_path,
    #     torch_dtype=torch.bfloat16,
    #     device="cpu"
    # )
    # model.load_state_dict(loaded_bf16_bin_dict)
    # print("BFloat16 Bin 权重加载成功")

    # # 6. 转换 Civitai 格式并以 BFloat16 保存
    # converter = model.state_dict_converter()
    # # 模拟Civitai权重（包含无关键）
    # mock_civitai_dict = {
    #     "flow_line_patch_embedding.weight": model.flow_line_patch_embedding.weight,
    #     "flow_line_patch_embedding.bias": model.flow_line_patch_embedding.bias,
    #     "unrelated.key1": torch.randn(10),
    #     "unrelated.key2": torch.randn(20)
    # }
    # # 转换并以 BFloat16 保存
    # converted_dict = converter.from_civitai(mock_civitai_dict)
    # save_state_dict(
    #     converted_dict,
    #     "flow_line_adapter_civitai_bf16.safetensors",
    #     torch_dtype=torch.bfloat16
    # )
    # print("Civitai格式转换后以 BFloat16 保存成功，权重键：", list(converted_dict.keys()))