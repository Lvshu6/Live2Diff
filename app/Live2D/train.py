import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from pathlib import Path
from tqdm import tqdm  # 引入 tqdm
from utils import gen_tracks,v2i
# ================= 1. 数据集定义 =================
SAMPLE_GAP = 3

class TrackGridDataset(Dataset):
    """数据集类：负责加载 .npy 文件并构建正负样本索引"""
    def __init__(self, folder_path, grid_size=32):
        self.grid_size = grid_size
        self.samples = [] 
        self.file_paths = glob.glob(os.path.join(folder_path, "*.npy"))
        
        if not self.file_paths:
            raise ValueError(f"在路径 {folder_path} 中未找到任何 .npy 文件")
            
        print(f"正在扫描数据目录... 找到 {len(self.file_paths)} 个文件。")
        total_pos = 0
        total_neg = 0

        # 构建索引过程（通常只需做一次）
        for f_idx, f_path in enumerate(self.file_paths):
            try:
                data = np.load(f_path)
                if data.ndim == 3:
                    data = data[np.newaxis, ...] 
                B, T, N, _ = data.shape
                
                # 正样本 (t, t+1)
                for b in range(B):
                    for t in range(T - 1):
                        self.samples.append((f_idx, b, t, t + 1, 1.0))
                        total_pos += 1
                
                # 负样本 (t, t_neg)
                for b in range(B):
                    for t in range(T):
                        min_t_neg = t + SAMPLE_GAP
                        if min_t_neg < T:
                            # 步长为2采样以减少负样本数量，防止显存爆炸
                            for t_neg in range(min_t_neg, T, 2): 
                                self.samples.append((f_idx, b, t, t_neg, 0.0))
                                total_neg += 1
                del data
            except Exception as e:
                continue
                
        self.length = len(self.samples)
        print(f"数据索引构建完成 | 正样本: {total_pos}, 负样本: {total_neg}, 总计: {self.length}")

    def _get_scale(self, f_idx):
        if not hasattr(self, 'scales'):
            self.scales = {}
        if f_idx not in self.scales:
            data = np.load(self.file_paths[f_idx])
            if data.ndim == 3:
                data = data[np.newaxis, ...]
            all_coords = data.reshape(-1, 2)
            max_x = all_coords[:, 0].max()
            max_y = all_coords[:, 1].max()
            self.scales[f_idx] = (self.grid_size / max(max_x, 1.0), self.grid_size / max(max_y, 1.0))
            del data
        return self.scales[f_idx]

    def _rasterize_to_grid(self, points, displacements, scale_x, scale_y):
        grid = torch.zeros((2, self.grid_size, self.grid_size), dtype=torch.float32)
        xs = (points[:, 0] * scale_x).long().clamp(0, self.grid_size - 1)
        ys = (points[:, 1] * scale_y).long().clamp(0, self.grid_size - 1)
        dx = displacements[:, 0] * scale_x
        dy = displacements[:, 1] * scale_y
        
        flat_grid = grid.view(2, -1)
        indices = ys * self.grid_size + xs
        # 使用 scatter_ 进行光栅化
        flat_grid[0, :].scatter_(0, indices, dx)
        flat_grid[1, :].scatter_(0, indices, dy)
        return grid

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f_idx, b_idx, t1, t2, label = self.samples[idx]
        data = np.load(self.file_paths[f_idx])
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        tracks = torch.from_numpy(data).float()
        scale_x, scale_y = self._get_scale(f_idx)
        
        pts_t1 = tracks[b_idx, t1, :, :]
        pts_t2 = tracks[b_idx, t2, :, :]
        displacement = pts_t2 - pts_t1
        
        grid_tensor = self._rasterize_to_grid(pts_t1, displacement, scale_x, scale_y)
        del data, tracks
        return grid_tensor, label

# ================= 2. 模型定义 =================
class AdjacencyCNN(nn.Module):
    def __init__(self, grid_size, hidden_dim=128, n_block=4):
        super(AdjacencyCNN, self).__init__()
        self.n_block = n_block
        
        layers = []
        # 简单的 CNN 堆叠结构
        in_ch, out_ch = 2, 16
        for _ in range(3): # 3个阶段
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU()]
            for _ in range(n_block):
                layers += [nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU()]
            layers.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 64) # 最大通道数 64
            
        self.features = nn.Sequential(*layers)
        
        reduced_size = max(1, grid_size // 8)
        self.flatten_dim = 64 * reduced_size * reduced_size
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ================= 3. 训练逻辑 =================
def train_model(folder_path, grid_size=48, epochs=15, batch_size=64, lr=1e-3, root_path=Path("."), save_interval=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据
    dataset = TrackGridDataset(folder_path, grid_size=grid_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = AdjacencyCNN(grid_size=grid_size).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_loss = float('inf')
    
    print(f"🚀 开始训练 | 设备: {device} | 总样本数: {len(dataset)}")
    
    # --- 修改点：tqdm 仅包装 Epoch 循环 ---
    # 使用 tqdm 包装 range(epochs)，这样可以看到整体进度条
    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # --- 修改点：内层循环完全静默，不打印，不显示进度条 ---
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # --- 修改点：仅在 Epoch 结束时计算并打印一行日志 ---
        avg_loss = running_loss / len(loader)
        accuracy = correct / total
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        # 使用 tqdm.write 确保打印内容不会破坏进度条显示
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = root_path / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'grid_size': grid_size
            }, best_model_path)
            # 只有当 Loss 显著下降或需要提示时才打印，这里选择静默保存或简单提示
            # tqdm.write(f"✨ 更新最佳模型 (Loss: {best_loss:.4f})")
    
    print(f"\n✅ 训练完成！最佳模型已保存至: {root_path / 'best_model.pth'},(Loss: {best_loss:.4f})")
    return str(root_path / "best_model.pth")

# ================= 4. 主程序入口 =================
if __name__ == "__main__":
    ROOT = Path("pet")
    BASE = ROOT / "videos"
    DATA_FOLDER = BASE / "track"
    
    GRID_SIZE = 48
    EPOCHS = 200
    BATCH_SIZE = 64
    
    v2i(ROOT)  # 生成图数据
    gen_tracks(ROOT)  # 生成轨迹数据
    
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ 错误：训练路径 '{DATA_FOLDER}' 不存在。")
    else:
        final_model_path = train_model(
            folder_path=DATA_FOLDER,
            grid_size=GRID_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=1e-5,
            root_path=ROOT,
            save_interval=50
        )