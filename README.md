# 3D-LAST-dataset
A large-scale dataset of 3D maneuvering target trajectories to facilitate effective training and comprehensive testing of the data-driven MTT  algorithm./用于训练以及测试数据驱动机动目标跟踪算法的三维机动目标数据集


## Requirements/环境要求

- Python ≥ 3.8
- NumPy 
- Matplotlib

## Quick Start/快速开始

Run `createtar.py` directly. The script will generate two `.npz` files in the `data/` directory:
| File | Content | Dimensions | Description |
|:-----|:--------|:----------:|:------------|
| `data/3D_mtdataset.npz` | Trajectory measurements | `(N, T, 3)` | Noisy position observations `[x, y, z]` |
| | One-hot labels | `(N, T, 4)` | Motion model labels `[CV, CA, CT+, CT−]` |
| `data/3D_mtdataset_true.npz` | Ground truth trajectories | `(N, T, 9)` | True state vectors `[x, vx, ax, y, vy, ay, z, vz, az]` |

Where `N` is the number of trajectories and `T` is the time steps per trajectory.

**Configuration**: Modify generation parameters in lines 4–13 of `createtar.py`. Change the output path and filenames in lines 176–177.

After dataset generation, run `drawset.py` to visualize the trajectories. Adjust the dataset path in line 52.

在文件 "createtar.py" 下直接运行，文件将在文件夹 “data” 中生成两个 “.npz” 文件，其中 "data/3D_mtdataset.npz" 文件包含两个数组，分别为轨迹测量值：维度为(N, T, 3)、轨迹one-hot编码：维度为(N, T, 4)，"data/3D_mtdataset_true.npz" 文件包含对应轨迹运动真值，维度为(N, T, 9)，其中N为轨迹数量，T为每条轨迹运动时间。生成数据集相关参数在文件4-13行进行修改，保持数据集路径以及名称在176、177行修改。

数据集生成后文件 “drawset.py” 文件对数据集整体进行绘图，在文件52行调整数据集路径。

## Usage Example/调用示例

X, Y = load_data("data/3D_mtdataset_zongti1.npz")
X_true, _ = load_data("data/3D_mtdataset_zongti1_true.npz")

X: Trajectory measurements
Y: Trajectory labels
X_true: Ground truth trajectories

X 为轨迹观测，Y 为轨迹标签，X_true为轨迹真值
