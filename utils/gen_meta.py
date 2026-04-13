import os
from pathlib import Path

import pandas as pd

# -------------------------- 配置参数 --------------------------
#example_video_dataset
BASE_PATH = "data/near_model/f5_17"  # 数据集根目录
config = Path(BASE_PATH) / "config"
config.mkdir(parents=True, exist_ok=True)

FOLDER_PATH = ""  # 媒体文件夹（相对于 BASE_PATH，仅处理该目录下的文件）
METADATA_SAVE_PATH =BASE_PATH + "/config/metadata.csv"  # Metadata 保存路径

# 预处理参数
TARGET_H = 512
TARGET_W = 512
NUM_FRAMES = 32
MAX_PIXELS = 1920 * 1080


# -------------------------- 1. 生成 Metadata（匹配 flow_line 下同名视频文件）--------------------------
def generate_metadata(base_path, relative_folder, save_path):
    # 计算目标目录的绝对路径
    target_dir_abs = Path(base_path) / relative_folder
    if not target_dir_abs.exists():
        raise FileNotFoundError(f"目标目录不存在：{target_dir_abs}（请检查 BASE_PATH 和 FOLDER_PATH 配置）")

    # 支持的媒体文件后缀
    supported_ext = {"jpg", "jpeg", "png", "webp", "gif", "mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"}
    metadata = []

    # 遍历目标目录下的文件（不递归）
    for file in target_dir_abs.iterdir():
        if file.is_dir():
            continue  # 跳过子文件夹

        # 检查文件后缀
        ext = file.suffix.lstrip(".").lower()
        if ext not in supported_ext:
            continue

        # 视频文件相对路径（相对于 BASE_PATH）
        video_rel_path = str(file.relative_to(base_path))

        # 寻找 flow_line 下的同名视频文件
        flow_video_name = file.name  # 保留扩展名的完整文件名
        flow_video_abs = target_dir_abs / "flow_line" / flow_video_name
        flow_line_rel_path = str(flow_video_abs.relative_to(base_path)) if flow_video_abs.exists() else None
        if flow_line_rel_path is None:
            print(f"⚠️ 未找到光流视频：{flow_video_name} 对应的 flow_line 文件")
        # 收集元信息
        metadata.append({
            "video": video_rel_path,
            "flow_line": flow_line_rel_path , # 光流视频文件相对路径（无则为 None）
            "prompt":"move"
        })

    # 保存 Metadata
    pd.DataFrame(metadata).to_csv(save_path, index=False)
    print(f"✅ Metadata 生成完成！")
    print(f" - 保存路径：{save_path}")
    print(f" - 处理目录（绝对路径）：{target_dir_abs}")
    print(f" - 共找到 {len(metadata)} 个媒体文件")
    # 统计匹配到的光流视频数量
    flow_count = sum(1 for item in metadata if item["flow_line"] is not None)
    print(f" - 其中 {flow_count} 个文件匹配到 flow_line 下的同名视频")


# 执行 Metadata 生成
generate_metadata(BASE_PATH, FOLDER_PATH, METADATA_SAVE_PATH)