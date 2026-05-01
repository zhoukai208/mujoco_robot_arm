import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 超参数配置 =====================
DATASET_PATH = "bc_reach_dataset.pkl"
MODEL_SAVE_PATH = "bc_reach_final_model.pth"      # 最终模型
BEST_MODEL_SAVE_PATH = "bc_reach_best_model.pth"  # 最优模型（重点用这个）
PLOT_SAVE_PATH = "bc_loss_curve.png"              # 损失曲线保存路径

# 训练参数
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 1e-3
VAL_LOSS_THRESHOLD = 0.0001  # 提前终止阈值：验证损失低于此值自动停止

# 观测/动作维度（严格匹配你的数据）
OBS_DIM = 10   # 7关节角度 + 3方块位置
ACT_DIM = 7    # 7维目标关节角

# ===================== 2. 自定义数据集类 =====================
class ReachDataset(Dataset):
    """加载你的BC演示数据集，适配PyTorch"""
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.obs_list = []
        self.act_list = []
        for sample in self.data:
            self.obs_list.append(sample["obs"])
            self.act_list.append(sample["act"])
        
        self.obs_list = np.array(self.obs_list, dtype=np.float32)
        self.act_list = np.array(self.act_list, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs_list[idx])
        act = torch.from_numpy(self.act_list[idx])
        return obs, act

# ===================== 3. BC模型（MLP）=====================
class BCModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, obs):
        return self.net(obs)

# ===================== 4. 训练主函数 =====================
def train():
    # 1. 加载数据集
    dataset = ReachDataset(DATASET_PATH)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 初始化
    model = BCModel(OBS_DIM, ACT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # ===================== 新增：训练监控变量 =====================
    best_val_loss = float('inf')    # 记录最优验证损失
    train_losses = []               # 收集每轮训练损失
    val_losses = []                 # 收集每轮验证损失
    early_stop = False              # 提前终止标志

    print("="*50)
    print(f"🚀 开始训练BC模型")
    print(f"📊 总样本数: {len(dataset)} | 训练集: {train_size} | 验证集: {val_size}")
    print(f"📦 输入: {OBS_DIM}维 | 输出: {ACT_DIM}维")
    print(f"⏹️  提前终止阈值: 验证损失 < {VAL_LOSS_THRESHOLD}")
    print("="*50)

    # 3. 训练循环
    for epoch in range(EPOCHS):
        # -------- 训练阶段 --------
        model.train()
        train_loss = 0.0
        for obs, act in train_loader:
            pred_act = model(obs)
            loss = criterion(pred_act, act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)
        train_loss /= train_size

        # -------- 验证阶段 --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, act in val_loader:
                pred_act = model(obs)
                loss = criterion(pred_act, act)
                val_loss += loss.item() * obs.size(0)
        val_loss /= val_size

        # ===================== 新增：保存损失历史 =====================
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ===================== 新增：保存最优模型 =====================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_SAVE_PATH)
            best_msg = "✅ 保存最优模型"
        else:
            best_msg = "⏸️  当前模型非最优"

        # -------- 打印日志 --------
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f} | {best_msg}")

        # ===================== 新增：提前终止训练 =====================
        if val_loss < VAL_LOSS_THRESHOLD:
            print(f"\n🎉 验证损失 {val_loss:.6f} 低于阈值 {VAL_LOSS_THRESHOLD}，自动提前结束训练！")
            early_stop = True
            break

    # 4. 保存最终模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("="*50)
    print(f"🏆 最优验证损失: {best_val_loss:.6f}")
    print(f"💾 最优模型: {BEST_MODEL_SAVE_PATH}")
    print(f"💾 最终模型: {MODEL_SAVE_PATH}")
    print(f"📈 损失曲线已保存: {PLOT_SAVE_PATH}")
    print("="*50)

    # ===================== 新增：绘制并保存损失曲线 =====================
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='#2E86AB', linewidth=1.5)
    plt.plot(val_losses, label='Val Loss', color='#A23B72', linewidth=1.5)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('BC Training Loss Curve', fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    plt.show()

if __name__ == "__main__":
    train()