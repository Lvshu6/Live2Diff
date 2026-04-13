import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from pathlib import Path
from safetensors.torch import save_file
from diffsynth.utils.yxd.remove import remove_prefix_from_ckpt

path="models/train/Wan2.2-TI2V-5B_lora/0115-1635/epoch-30.safetensors"
remove_prefix_from_ckpt(path, path, ["pipe.flow_line_adapter.", "pipe.dit."])
state_dict = load_state_dict(path)
# 分离属于 dit 和 flow_line_adapter 的参数
dit_state_dict = {}
flow_line_adapter_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("flow_line_patch_embedding.") or key.startswith("flow_line_blocks."):
        # 属于 flow_line_adapter 的参数
        flow_line_adapter_state_dict[key] = value
    else:
        # 属于 dit 的参数
        dit_state_dict[key] = value
        

def save_model(base_path, name):
    # base_path = Path("models/train/Wan2.1-Fun-1.3B-InP_full/1205-1226")
    global dit_state_dict,flow_line_adapter_state_dict
    base_path = Path(base_path)

    base_path.mkdir(parents=True, exist_ok=True)
    # dit_path = base_path / "diffusion_pytorch_model.safetensors"
    dit_path = base_path / name

    flow_line_adapter_path = base_path / "flow_line_adapter.safetensors"

    save_file(dit_state_dict, dit_path)
    save_file(flow_line_adapter_state_dict, flow_line_adapter_path)

    print("模型已分别保存为 safetensors 格式")
    
save_model("models/train/Wan2.2-TI2V-5B_lora/0115-1635", "lora_5B.safetensors")
