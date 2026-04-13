import os
import glob
import numpy as np
import imageio.v3 as iio
from pathlib import Path
from tqdm import tqdm

def backward_pad_frames(full_frames, clip_frames, target_len=5):
    """从原序列中向前回溯补齐到 target_len 帧，保持顺序"""
    cur = len(clip_frames)
    if cur >= target_len:
        return clip_frames[:target_len]
    need = target_len - cur

    # 找 clip 起始位置
    start_idx = None
    for i in range(len(full_frames)):
        if np.array_equal(full_frames[i], clip_frames[0]):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("Clip start frame not found in full video")

    pad_start = max(0, start_idx - need)
    pad = full_frames[pad_start:start_idx]
    result = np.concatenate([pad, clip_frames], axis=0)
    return result[:target_len]  # 确保正好 target_len

def process_pair_video(track_path: Path, flow_path: Path, output_base: Path):
    try:
        track_frames = iio.imread(str(track_path))
        flow_frames = iio.imread(str(flow_path))
    except Exception as e:
        print(f"⚠️ 读取失败 {track_path.name} 或 {flow_path.name}: {e}")
        return

    if len(track_frames) != len(flow_frames):
        print(f"⚠️ 帧数不一致！{track_path.name}: {len(track_frames)} vs {flow_path.name}: {len(flow_frames)}")
        return

    total = len(track_frames)
    chunk_size = 5
    num_chunks = total// chunk_size + (1 if total % chunk_size else 0)  # 向上取整
    if total == chunk_size:
        save_name = track_path.name
        
        iio.imwrite(output_base / "track" / save_name, track_frames, fps=5, codec="h264", quality=10)
        iio.imwrite(output_base / "flow_line" / save_name, flow_frames, fps=5, codec="h264", quality=10)

        return

    for idx in range(num_chunks):
        start = idx * chunk_size
        end = start + chunk_size
        t_clip = track_frames[start:end]
        f_clip = flow_frames[start:end]

        # 向前补齐至 5 帧
        t_padded = backward_pad_frames(track_frames, t_clip, target_len=5)
        f_padded = backward_pad_frames(flow_frames, f_clip, target_len=5)

        # 保存
        stem = track_path.stem
        save_name = f"{stem}_clip_{idx + 1:03d}.mp4"
        iio.imwrite(output_base / "track" / save_name, t_padded, fps=5, codec="h264", quality=10)
        iio.imwrite(output_base / "flow_line" / save_name, f_padded, fps=5, codec="h264", quality=10)

def main():
    base = Path("data/newtrack")
    flow = Path("flow_line")
    flow=base / flow.name
    output_dir = base / "f5"

    (output_dir / "track").mkdir(parents=True, exist_ok=True)
    (output_dir / "flow_line").mkdir(parents=True, exist_ok=True)

    # 收集 track 中所有视频
    video_formats = ("mp4", "avi", "mov", "mkv", "webm")
    track_files = []
    for fmt in video_formats:
        track_files.extend(base.glob(f"*.{fmt}"))

    if not track_files:
        raise FileNotFoundError(f"在 {base} 中未找到视频文件")

    print(f"📁 找到 {len(track_files)} 个视频，开始处理...")

    for track_path in tqdm(track_files, desc="处理视频对"):
        flow_path = flow / track_path.name
        if not flow_path.exists():
            print(f"⚠️ 对应的 flow 文件不存在: {flow_path}")
            continue
        process_pair_video(track_path, flow_path, output_dir)

    print(f"\n✅ 处理完成！结果保存至：{output_dir}")

if __name__ == "__main__":
    main()