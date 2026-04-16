import os
from pathlib import Path
import imageio.v3 as iio

# 设置你的视频根目录（包含多个视频文件）
video_dir = Path(r"/home/suat/yxd/DiffSynth-Studionew/app/Live2D/pet/videos")  # 修改为你自己的路径
output_root =  video_dir / "images" # 输出的根目录

# 确保输出根目录存在
output_root.mkdir(parents=True, exist_ok=True)

# 支持的视频扩展名
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

for video_path in video_dir.iterdir():
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        continue

    # 创建对应输出文件夹：output_root / video_name_without_ext
    folder_name = video_path.stem  # 不含扩展名的文件名
    output_folder = output_root / folder_name
    if output_folder.exists():
        print(f"⚠️ 跳过已存在的文件夹: {output_folder}")
        continue
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"正在处理视频: {video_path.name} → 保存到 {output_folder}")

    # 使用 iio.imread 读取视频的所有帧
    try:
        # iio.imread 会返回一个包含所有帧的 NumPy 数组
        frames = iio.imread(video_path)
        
        # 遍历所有帧并保存
        for idx, frame in enumerate(frames):
            img_path = output_folder / f"{idx:06d}.png"  # 000000.png, 000001.png, ...
            iio.imwrite(img_path, frame)
            
    except Exception as e:
        print(f"⚠️ 处理 {video_path.name} 时出错: {e}")
        continue

    # 注意：如果视频为空，frames将为空数组，idx将未定义
    if frames.size > 0:
        print(f"✅ 完成: {video_path.name} 共提取 {len(frames)} 帧")
    else:
        print(f"✅ 完成: {video_path.name} (视频为空)")