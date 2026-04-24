import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from pathlib import Path
import os
import itertools  # 导入 itertools 用于生成组合

# ================== 配置 ==================
DEVISE = "cuda:0"
input_folder = Path("data/nuero/f2l")          # 原始视频所在目录
output_folder = input_folder / "output"    # 输出目录
# 确保输出目录存在
output_folder.mkdir(exist_ok=True, parents=True)

video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

# ================== 加载模型 ==================
print("Loading model...")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=DEVISE,
    model_configs=[
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="PAI/Wan2.1-Fun-V1.1-14B-InP", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    vram_limit=16.0
)
pipe.vram_management_enabled = True

# 加载 flow_line_adapter 微调权重
flow_state_dict = load_state_dict("WanFlow/Wan2.1-Fun-V1.1-14B-InP/full/0309/epoch-3.safetensors")
pipe.flow_line_adapter.load_state_dict(flow_state_dict)
pipe.flow_line_adapter.to(DEVISE)

print("Model loaded successfully.")

# ================== 辅助函数 ==================
def get_video_last_frame(video_path):
    """
    获取视频的最后一帧图像。
    """
    video_data = VideoData(video_path)
    if len(video_data) > 0:
        return video_data[len(video_data)-1]
    else:
        print(f"⚠️ Warning: Video is empty, skipping: {video_path.name}")
        return None


# ================== 批量处理 (修改部分) ==================

# 1. 首先，收集所有符合条件的视频文件
video_files = [f for f in input_folder.iterdir() if f.suffix.lower() in video_extensions]

# 2. 使用 itertools.combinations 生成所有两两配对 (C(n, 2))
# 这会生成一个包含元组的迭代器，例如 [(video1, video2), (video1, video3), ...]
video_pairs = itertools.combinations(video_files, 2)

print(f"\nFound {len(video_files)} videos. Starting to process {len(video_files) * (len(video_files) - 1) // 2} pairs...")

for video_a, video_b in video_pairs:
    print(f"\n--- Processing Pair ---")
    print(f"Video A (Start Frame): {video_a.name}")
    print(f"Video B (End Frame):   {video_b.name}")

    # 3. 获取配对视频的尾帧
    # 视频A的最后一帧作为生成视频的首帧
    start_frame = get_video_last_frame(video_a)
    # 视频B的最后一帧作为生成视频的尾帧
    end_frame = get_video_last_frame(video_b)

    # 如果任一尾帧获取失败，则跳过此配对
    if start_frame is None or end_frame is None:
        print(f"⚠️ Skipping pair due to failed frame extraction.")
        continue

    try:
        # 加载 flo

        img_width, img_height = start_frame.size
        # 帧数可以根据需要设定，这里沿用 flow 数据的长度

        # 5. 推理生成视频
        # 关键修改：使用 input_image 和 end_image 参数
        generated_video = pipe(
            prompt="move",
            negative_prompt="",
            input_image=start_frame,  # 传入视频A的尾帧
            end_image=end_frame,      # 传入视频B的尾帧
            seed=0,
            tiled=True,
            height=img_height,
            width=img_width,
            num_inference_steps=20,
            num_frames=13
        )

        # 6. 保存结果
        # 输出文件名清晰地表明是哪两个视频生成的
        output_filename = f"{video_a.stem}_to_{video_b.stem}{video_a.suffix}"
        output_path = output_folder / output_filename
        
        save_video(generated_video, output_path, fps=15, quality=5)
        print(f"✅ Saved: {output_path}")

    except Exception as e:
        print(f"❌ Error processing pair ({video_a.name}, {video_b.name}): {e}")
        continue

print("\n🎉 Batch processing for all pairs completed.")