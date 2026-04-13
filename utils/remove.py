import argparse
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def remove_prefix_from_ckpt(input_path, output_path, prefixes):
    """
    从 safetensors checkpoint 中移除指定前缀。
    
    Args:
        input_path (str): 输入 .safetensors 文件路径
        output_path (str): 输出 .safetensors 文件路径
        prefixes (list of str): 要移除的前缀列表，按顺序尝试匹配
    """
    print(f"Loading checkpoint from: {input_path}")
    state_dict = {}
    
    # 读取 safetensors 文件
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            new_key = key
            
            # 尝试移除每一个前缀（按顺序）
            for prefix in prefixes:
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    print(f"Renamed: '{key}' → '{new_key}'")
                    break  # 只移除第一个匹配的前缀
            
            state_dict[new_key] = tensor

    print(f"\nSaving cleaned checkpoint to: {output_path}")
    save_file(state_dict, output_path)
    print("Done!")

if __name__ == "__main__":
    input_path="models/train/Wan2.2-TI2V-5B_lora/epoch-2.safetensors"
    output_path="WanFlow/Wan2.1-Fun-V1.1-14B-InP_lora_8gpu/epoch-3.safetensors"
    output_path=input_path
    parser = argparse.ArgumentParser(description="Remove prefixes from safetensors checkpoint keys.")
    parser.add_argument("--input_ckpt", type=str, default=input_path, help="Input .safetensors file (e.g., epoch-0.safetensors)")
    parser.add_argument("--output_ckpt", type=str, default=output_path, help="Output file path. Default: <input>_clean.safetensors")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.flow_line_adapter.,pipe.dit.",
                        help='Comma-separated prefixes to remove, e.g., "pipe.flow_line_adapter.,pipe.dit."')
    
    args = parser.parse_args()
    
    # 解析前缀列表
    prefixes = [p.strip() for p in args.remove_prefix_in_ckpt.split(",") if p.strip()]
    if not prefixes:
        raise ValueError("No valid prefixes provided!")
    
    # 设置输出路径
    output_path = args.output_ckpt
    if output_path is None:
        if args.input_ckpt.endswith(".safetensors"):
            output_path = args.input_ckpt.replace(".safetensors", "-clean.safetensors")
        else:
            output_path = args.input_ckpt + "-clean.safetensors"
    
    # 执行清理
    remove_prefix_from_ckpt(args.input_ckpt, output_path, prefixes)