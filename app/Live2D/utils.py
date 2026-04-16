import os
from pathlib import Path
import imageio.v3 as iio
from tqdm import tqdm
import glob
import numpy as np
import torch
from cotracker.utils.visualizer import read_video_from_path
from cotracker.utils.DirectionVisualizer import DirectionVisualizer

# 自动选择设备
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def visualize(pred_tracks, video, save_dir, filename="direction_tracks"):
    """
    可视化轨迹。
    为了节省显存，绘图操作在 CPU 上进行。
    """
    visualizer = DirectionVisualizer(
        save_dir=save_dir,
        linewidth=3,
        fps=10,
        history=3,
        save_pure_trajectory=True
    )
    visualizer.visualize(
        video,
        pred_tracks,
        filename=filename,
        save_video=True
    )

def gen_tracks(base_dir):
    # --- 1. 配置变量 ---
    input_dir = os.path.join(base_dir, "videos")
    output_dir = os.path.join(input_dir, "track")
    flow_dir = os.path.join(base_dir, "flow_line")
    
    # 模型运行参数
    grid_size = 48            # 跟踪网格大小
    grid_query_frame = 0      # 查询帧索引
    backward_tracking = False # 是否开启反向跟踪

    # --- 2. 准备目录 ---
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)

    # --- 3. 获取视频列表 ---
    # 提前获取列表，以便 tqdm 计算总长度
    video_paths = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    if not video_paths:
        print(f"❌ No .mp4 files found in {input_dir}")
        return

    print(f"Found {len(video_paths)} videos. Loading CoTracker3 model...")
    
    # --- 4. 加载模型 ---
    try:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        model = model.to(DEVICE)
        model.eval()
        print(f"Model loaded on {DEVICE}.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
        
    # --- 5. 批量处理循环 (使用 tqdm 包裹) ---
    # 使用 tqdm 包裹 video_paths，显示整体进度
    for video_path in tqdm(video_paths, desc="Processing Videos", unit="video"):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        npy_path = os.path.join(output_dir, f"{filename}.npy")

        try:
            # --- 步骤 A: 轨迹预测 ---
            if os.path.exists(npy_path):
                # 如果文件存在，更新进度条描述为 "Skipped"，不打印额外日志
                tqdm.write(f"⏭️ {filename} (Skipped)")
            else:
                video_np = read_video_from_path(video_path)
                if video_np is None:
                    tqdm.write(f"⚠️ {filename} (Read Failed)")
                    continue
                
                # 预处理: (S, H, W, C) -> (1, S, C, H, W)
                video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float().to(DEVICE)

                with torch.no_grad():
                    pred_tracks, pred_visibility = model(
                        video_tensor,
                        grid_size=grid_size,
                        grid_query_frame=grid_query_frame,
                        backward_tracking=backward_tracking,
                    )
                
                # 保存轨迹
                np.save(npy_path, pred_tracks.cpu().numpy())
                # 仅在处理新文件时打印简单提示
                tqdm.write(f"✅ {filename} (Processed)")
            
            # --- 步骤 B: 可视化 ---
            # 加载轨迹 (CPU)
            # tracks_np = np.load(npy_path)
            # # 读取视频 (CPU)
            # vis_video_np = read_video_from_path(video_path)
            # if vis_video_np is None: continue
            
            # vis_video_tensor = torch.from_numpy(vis_video_np).permute(0, 3, 1, 2)[None].float()
            # vis_tracks_tensor = torch.from_numpy(tracks_np)

            # visualize(vis_tracks_tensor, vis_video_tensor, flow_dir, filename=filename)

        except Exception as e:
            tqdm.write(f"❌ Error processing {filename}: {e}")
            continue

    print("\n🎉 All processing done!")

def v2i(base_dir):
    base_dir = Path(base_dir)
    video_dir = base_dir / "videos"
    output_root = base_dir / "images"
    
    output_root.mkdir(parents=True, exist_ok=True)

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    video_paths = [
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]

    for video_path in tqdm(video_paths, desc="视频转图片"):
        folder_name = video_path.stem
        output_folder = output_root / folder_name
        
        if output_folder.exists():
            continue # 静默跳过
            
        output_folder.mkdir(parents=True, exist_ok=True)

        frames = iio.imread(video_path)
            
        for idx, frame in enumerate(frames):
            img_path = output_folder / f"{idx:06d}.png"
            iio.imwrite(img_path, frame)

if __name__ == "__main__":
    base="/home/suat/yxd/DiffSynth-Studionew/app/Live2D/pet"
    v2i(base)
    gen_tracks(base)