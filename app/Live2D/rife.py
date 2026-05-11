import os
import cv2
import torch
import argparse
import hashlib
from torch.nn import functional as F
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# 全局设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def load_rife_model(model_dir='train_log'):
    """ 加载RIFE模型，兼容v1/v2/v3版本 """
    try:
        try:
            from model.RIFE_HDv2 import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(model_dir, -1)
            print("Loaded v3.x HD model.")
    except:
        from model.RIFE_HD import Model
        model = Model()
        model.load_model(model_dir, -1)
        print("Loaded v1.x HD model")
    if not hasattr(model, 'version'):
        model.version = 0
    model.eval()
    model.device()
    return model


def get_path_hash(path1: str, path2: str) -> str:
    """ 提取路径的【最近两级路径】(父目录/文件名) 进行拼接并计算 MD5。
    例如: 'dfadfasf/fadf/000000.png' -> 提取 'fadf/000000.png'
    使用 .as_posix() 确保跨平台路径分隔符统一为 '/'，保证 Hash 一致性。
    """
    p1 = Path(path1)
    p2 = Path(path2)
    recent_path_1 = (Path(p1.parent.name) / p1.name).as_posix()
    recent_path_2 = (Path(p2.parent.name) / p2.name).as_posix()
    # 2. 拼接字符串并计算 Hash
    concat_str = recent_path_1 + recent_path_2
    # print(f"生成 Hash 的路径片段: concat_str='{concat_str}'")
    md5 = hashlib.md5(concat_str.encode('utf-8')).hexdigest()
    return md5


def interpolate_images(
        img_path1: str,
        img_path2: str,
        model,
        exp: int = 4,
        cache_root: str = "./cache"
) -> list:
    """ 核心补帧函数（带缓存）
    输入：两张图片路径、RIFE模型、插值倍数、缓存根目录
    输出：[img1路径, 补帧路径..., img2路径]
    """
    # 1. 生成双向哈希文件夹名
    hash_ab = get_path_hash(img_path1, img_path2)
    hash_ba = get_path_hash(img_path2, img_path1)
    cache_dir_ab = os.path.join(cache_root, hash_ab)
    cache_dir_ba = os.path.join(cache_root, hash_ba)

    # 2. 优先读取缓存（任意一个存在都直接读取）
    cache_dir = None
    if os.path.exists(cache_dir_ab):
        cache_dir = cache_dir_ab
    elif os.path.exists(cache_dir_ba):
        cache_dir = cache_dir_ba

    if cache_dir is not None:
        print(f"读取缓存: {cache_dir}")
        frame_files = sorted([f for f in os.listdir(cache_dir) if f.endswith(('.png', '.exr'))])
        frame_paths = [os.path.join(cache_dir, f) for f in frame_files]
        expected_frames = 2 ** exp + 1
        if len(frame_paths) == expected_frames:
            return frame_paths
        else:
            print(f"缓存帧数不匹配，删除重建：{cache_dir}")
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)

    # 3. 无缓存 → 执行补帧
    print(f"无缓存，开始补帧: {img_path1} ↔ {img_path2}")
    os.makedirs(cache_dir_ab, exist_ok=True)
    os.makedirs(cache_dir_ba, exist_ok=True)

    # ------------------- 原图像读取/预处理 -------------------
    # 读取原始图片以获取尺寸
    img0_raw = cv2.imread(img_path1, cv2.IMREAD_UNCHANGED)
    img1_raw = cv2.imread(img_path2, cv2.IMREAD_UNCHANGED)

    # --- 记录原始分辨率 ---
    orig_h, orig_w = img0_raw.shape[:2]

    # --- 强制缩放到模型输入尺寸 (448x256) ---
    model_h, model_w = 256, 448
    img0 = cv2.resize(img0_raw, (model_w, model_h))
    img1 = cv2.resize(img1_raw, (model_w, model_h))

    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    # ------------------- RIFE 插值生成帧列表 -------------------
    img_list = [img0, img1]
    for i in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = model.inference(img_list[j], img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp

    # ------------------- 保存到两个缓存文件夹 -------------------
    output_paths = []

    # 正序保存 → cache_dir_ab
    for i, tensor_img in enumerate(img_list):
        # 1. 转回 numpy 并去除 padding
        img_np = (tensor_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
        # 2. 【关键步骤】将模型尺寸 (448x256) 还原为 原始尺寸 (orig_w, orig_h)
        img_resized = cv2.resize(img_np, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        # 3. 保存
        save_path = os.path.join(cache_dir_ab, f"{i:06d}.png")
        cv2.imwrite(save_path, img_resized)
        output_paths.append(save_path)

    # 倒序保存 → cache_dir_ba （关键修改）
    for i, tensor_img in enumerate(reversed(img_list)):
        # 1. 转回 numpy 并去除 padding
        img_np = (tensor_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
        # 2. 【关键步骤】将模型尺寸 (448x256) 还原为 原始尺寸 (orig_w, orig_h)
        img_resized = cv2.resize(img_np, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        # 3. 保存
        save_path = os.path.join(cache_dir_ba, f"{i:06d}.png")
        cv2.imwrite(save_path, img_resized)

    return output_paths


if __name__ == "__main__":
    # 命令行参数
    # --- 变量控制区域 ---
    # 在这里直接修改参数
    # img_paths = [r'D:\Downloads\Live2Diff\app\Live2D\temp\I0_0.png',
    #              r'D:\Downloads\Live2Diff\app\Live2D\temp\I0_1.png']  # 两张图片路径
    img_paths = [r'D:\Downloads\Live2Diff\app\Live2D\temp\000000.png',
                 r'D:\Downloads\Live2Diff\app\Live2D\temp\000012.png']  # 两张图片路径
    exp = 2  # 插值倍数 2^exp
    model_dir = 'train_log'  # 模型文件夹
    # ---------------------

    # 加载模型
    rife_model = load_rife_model(model_dir)

    # 执行补帧
    result_frames = interpolate_images(
        img_path1=img_paths[0],
        img_path2=img_paths[1],
        model=rife_model,
        exp=exp,
        cache_root="./cache"
    )

    # 输出结果
    print("\n✅ 补帧完成，返回帧列表：")
    for idx, p in enumerate(result_frames):
        print(f"[{idx}] {p}")