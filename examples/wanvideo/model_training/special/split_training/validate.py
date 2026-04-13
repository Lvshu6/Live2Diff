import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from pathlib import Path
import os
DEVISE = "cuda" 
# 1. 限制CUDA可见设备为GPU 2（物理编号）
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=DEVISE,
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    vram_limit=40.0
)
pipe.vram_management_enabled = True
# state_dict = load_state_dict("models/train/Wan2.2-TI2V-5B_full/epoch-1.safetensors")
flow_state_dict = load_state_dict("models/train/Wan2.1-Fun-V1.1-14B-InP_full/epoch-0.safetensors")

# pipe.dit.load_state_dict(state_dict)
pipe.flow_line_adapter.load_state_dict(flow_state_dict)
pipe.flow_line_adapter.to(DEVISE)
# lora_state_dict = load_state_dict("models/lora/lora_5B.safetensors")
# pipe.load_lora(pipe.dit, alpha=1,state_dict=lora_state_dict)

base=Path("data/track")
flow=Path("flow_line")

video_path = Path("猫猫_clip_3_clip_002.mp4")
flow_path = base / flow / video_path

video = VideoData(base / video_path)
flow=VideoData(flow_path)

img_width, img_height = video[0].size

# First and last frame to video
video = pipe(
    prompt="move",
    negative_prompt="",
    input_image=video[0], flow_line=flow,
    seed=0, tiled=True,
    height=img_height, width=img_width, num_inference_steps=20, num_frames=len(video)
)
save_video(video, Path("valid") / video_path , fps=15, quality=5)
