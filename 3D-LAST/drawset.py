import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    'font.size': 5,          # 全局字体
    'axes.titlesize': 12,     # 标题
    'axes.labelsize': 9,     # 轴标签
    'xtick.labelsize': 7,     # x轴刻度
    'ytick.labelsize': 7,     # y轴刻度
    'legend.fontsize': 9,     # 图例
})

def plot_trajectories_from_npz(file_path, title="  "):
    """
    从 .npz 文件中加载数据集并绘制所有轨迹。
    
    参数:
    - file_path: .npz 文件的路径。
    - title: 图形的标题。
    """
    name = f"Figures/dataset/dataset_test.png"
    # 加载 .npz 文件
    data = np.load(file_path)
    trajectories = data['trajectories']
    #model_labels = data['labels']

    # 创建一个3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 定义颜色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))  # 为每条轨迹分配不同颜色

    # 绘制每条轨迹
    for positions, color in zip(trajectories, colors):
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]  # 提取x, y, z坐标
        ax.plot(x, y, z, color=color, alpha=0.7)  # 绘制轨迹

    # 设置图形属性
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(title)
    ax.grid(True)

    # 显示图形
    plt.show()
    plt.savefig(name)

# 调用函数绘制轨迹
plot_trajectories_from_npz("data/3D_mtdataset_try_true.npz")