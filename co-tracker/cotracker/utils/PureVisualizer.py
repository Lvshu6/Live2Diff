import os
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw
import flow_vis


class PureVisualizer:
    def __init__(
        self,
        save_dir="./results",
        linewidth=4,
        fps=10,
        history=3,
        # 固定为True（因为只保留纯轨迹），无需再作为可选参数
        bg_color=(255, 255, 255)  # 新增：背景色，默认纯白
    ):
        self.save_dir = save_dir
        self.linewidth = linewidth
        self.fps = fps
        self.history = history
        self.bg_color = bg_color

    def visualize(
        self,
        tracks: torch.Tensor,  # 仅保留轨迹输入 (B,T,N,2)
        filename="direction_tracks",
        save_video=True,
        # 新增：手动指定画布尺寸（因为无视频输入）
        canvas_size=(640, 480)  # (W, H)，默认640x480
    ):
        # 处理轨迹数据：去掉batch维度，转numpy
        tracks = tracks[0].cpu().numpy().astype(np.float32)  # (T, N, 2)
        T, N = tracks.shape[:2]
        W, H = canvas_size  # 画布宽高

        # 初始化纯轨迹帧列表
        pure_frames = []      
        # 创建背景图（根据指定的背景色）
        bg = np.full((H, W, 3), self.bg_color, dtype=np.uint8)

        for t in range(T):
            # 每次都复制干净的背景图，避免轨迹叠加
            pure_frame = Image.fromarray(bg.copy())
            pure_draw = ImageDraw.Draw(pure_frame)

            # 绘制历史方向线（核心逻辑保留）
            for k in range(1, self.history + 1):
                if t - k < 0:
                    continue

                prev = tracks[t - k]      # (N, 2) 上一帧轨迹点
                curr = tracks[t - k + 1]  # (N, 2) 当前帧轨迹点

                # 计算动态线宽（近帧粗，远帧细）
                min_width = 2
                max_width = self.linewidth
                if self.history > 1:
                    current_width = int(
                        max_width - (k - 1) * (max_width - min_width) / (self.history - 1)
                    )
                else:
                    current_width = max_width
                current_width = max(min_width, current_width)

                # 遍历每个轨迹点绘制连线
                for i in range(N):
                    x1, y1 = prev[i]
                    x2, y2 = curr[i]

                    # 跳过无效点（坐标为0的点）
                    if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0:
                        continue

                    # 使用flow_vis根据运动方向着色
                    dx = x2 - x1
                    dy = y2 - y1
                    flow_vec = np.array([[[dx, dy]]], dtype=np.float32)
                    color_img = flow_vis.flow_to_color(flow_vec)
                    color = tuple(color_img[0, 0].tolist())

                    # 绘制轨迹线
                    pure_draw.line((x1, y1, x2, y2), fill=color, width=current_width)

            # 将绘制好的帧加入列表
            pure_frames.append(np.array(pure_frame))

        # 构造返回的tensor（保持原接口习惯，返回(B,T,C,H,W)格式）
        result_tensor = torch.from_numpy(
            np.stack(pure_frames)
        ).permute(0, 3, 1, 2)[None].byte()

        # 保存视频（核心功能保留）
        if save_video:
            self._save_video(pure_frames, filename)

        return result_tensor

    def _save_video(self, frames, filename):
        """保存视频的辅助函数（保留核心逻辑）"""
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, f"{filename}.mp4")
        writer = imageio.get_writer(save_path, fps=self.fps)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"Saved to {save_path}")


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    # 1. 创建可视化器实例
    visualizer = PureVisualizer(
        save_dir="./pure_trajectory_results",
        linewidth=6,
        fps=5,
        history=2,
        bg_color=(255, 255, 255)  # 纯白背景
    )

    # 2. 构造测试轨迹数据 (B=1, T=10, N=2, 2)
    # T=10帧，N=2个轨迹点，每个点是(x,y)坐标
    test_tracks = torch.rand(1, 10, 2, 2) * 500  # 坐标范围0-500
    # 给部分点设置无效值（测试跳过逻辑）
    test_tracks[0, 3, 0] = 0
    test_tracks[0, 7, 1] = 0

    # 3. 生成纯轨迹视频
    visualizer.visualize(
        tracks=test_tracks,
        filename="test_pure_trajectory",
        save_video=True,
        canvas_size=(600, 600)  # 画布尺寸600x600
    )