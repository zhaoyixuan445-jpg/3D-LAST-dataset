import numpy as np

# 定义运动模型的参数
DT = 1  # 采样时间间隔 (s)
T_MAX = 400  # 最大仿真时间 (s)   训练使用100
NUM_TRAJECTORIES = 100000  # 生成的轨迹数量
MAX_OMEGA = 0.03  # 最大角速度
NM2M = 1852.0                       # 1 海里 = 1852 m
R_MIN = 0.5 * NM2M                  # 926 m
R_MAX = 30.0 * NM2M                 # 111120 m
Z_MAX = 2.0 * NM2M                 # 37040 m
OMEGA_MAX = 0.30          # rad/s
R_BUF     = 500.0         # m



# 运动模型的状态转移矩阵
def cv_model(dt):
    """匀速直线运动模型 (CV)"""
    F = np.array([
        [1, dt, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, dt, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, dt,0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    return F

def ca_model(dt):
    """匀加速直线运动模型 (CA)"""
    F = np.array([
        [1, dt, 0.5 * dt**2, 0, 0, 0, 0, 0, 0],
        [0, 1, dt, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, dt, 0.5 * dt**2, 0, 0, 0],
        [0, 0, 0, 0, 1, dt, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, dt, 0.5*dt**2],
        [0, 0, 0, 0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    return F

def ct_model(dt, omega):  # 假设旋转轴恒为z轴
    """匀速转弯运动模型 (CT)"""
    if omega == 0:
        return cv_model(dt)
    else:
        F = np.array([
            [1, np.sin(omega * dt) / omega, 0, 0, (1 - np.cos(omega * dt)) / omega, 0, 0, 0, 0],
            [0, np.cos(omega * dt), 0, 0, -np.sin(omega * dt), 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, (1 - np.cos(omega * dt)) / omega, 0, 1, np.sin(omega * dt)/omega, 0, 0, 0, 0],
            [0, np.sin(omega * dt), 0, 0, np.cos(omega * dt), 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, dt, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        return F

# 生成随机初始状态
def generate_initial_state():
    pos = np.random.uniform(-0.8*R_MAX, 0.8*R_MAX, size=3)  # 初始位置 (x, y, z)
    vel = np.random.uniform(-5.0, 5.0, size=3)    # 初始速度 (vx, vy, vz)
    acc = np.random.uniform(-0.3, 0.3, size=3)      # 初始加速度 (ax, ay, az)
    arr = np.array([pos[0], vel[0], acc[0],pos[1], vel[1], acc[1],pos[2], vel[2], acc[2]])
    return arr

# 生成轨迹
def generate_trajectory(initial_state, T, dt):
    # 随机选择两个运动模型
    models = np.random.choice(['CV', 'CA', 'CTmax', 'CTmin'], size=3, replace=False) #
    
    change_point_1 = np.random.randint(T // 4, 2 * T // 4)  # 随机选择切换点
    change_point_2 = np.random.randint(2 * T // 4, 3 * T // 4)
    
    # 定义运动模型的编码
    model_encoding = {
        'CV': [1, 0, 0, 0],
        'CA': [0, 1, 0, 0],
        'CTmax': [0, 0, 1, 0],
        'CTmin': [0, 0, 0, 1]
    }

    state = initial_state
    
    positions = [np.array([state[0], state[3], state[6]])]  # 只保存位置信息
    positions_true = [np.array([state[0], state[3], state[6]])]
    model_labels = [model_encoding[models[0]]]  # 初始模型标签
    state_lock = 0  #状态锁 状态越界，则恒为转弯模型
    noise = np.random.randint(1,7)/100
    for t in range(1, T):
        x, y   = state[0], state[3]
        
        r_pred = np.hypot(x , y )


        if (r_pred > R_MAX - R_BUF) :
            
            state_lock = 1

        if(state_lock == 0):
            if (t <= change_point_1):  # 切换运动模型
                model_type = models[0]
            elif (t <= change_point_2 and t>=change_point_1):
                model_type = models[1]
            else:
                model_type = models[2]
        else:
            model_type = 'CTmax'


        tampla = [1, 0, 0, 0]
        if model_type == 'CV':
            F = cv_model(dt)
            tampla = model_encoding[model_type]
        elif model_type == 'CA':
            F = ca_model(dt)
            tampla = model_encoding[model_type]
        elif model_type == 'CTmax':
            omega = np.random.uniform(MAX_OMEGA/2, MAX_OMEGA)  # 随机角速度
            F = ct_model(dt, omega)
            model_labels[-1][2] = omega / MAX_OMEGA  # 软编码
            tampla = [0, 0, (omega+MAX_OMEGA) / 2*MAX_OMEGA, 1-((omega+MAX_OMEGA) / 2*MAX_OMEGA)]
        elif model_type == 'CTmin':
            omega = np.random.uniform(-MAX_OMEGA, -MAX_OMEGA/2)  # 随机角速度
            F = ct_model(dt, omega)
            model_labels[-1][3] = -omega / MAX_OMEGA  # 软编码
            tampla = [0, 0, (omega+MAX_OMEGA) / 2*MAX_OMEGA, 1-((omega+MAX_OMEGA) / 2*MAX_OMEGA)]
        else:
            raise ValueError("Unknown model type")
                
        state = F @ state + np.random.normal(0, 0.2, size=state.shape)  # 添加过程噪声
        statenose = state + np.random.normal(0, noise, size=state.shape)  # 添加观测噪声
        temp = np.array([statenose[0], statenose[3], statenose[6]])
        temp_1 = np.array([state[0], state[3], state[6]])
        positions.append(temp)  # 只保存位置信息
        positions_true.append(temp_1)
        model_labels.append(tampla)  # 记录当前时间步的模型标签

    return np.array(positions), np.array(model_labels), np.array(positions_true)
# 生成数据集
def generate_dataset(num_trajectories, T, dt):
    dataset = []
    dataset_1 = []
    for _ in range(num_trajectories):
        initial_state = generate_initial_state()
        positions, model_labels, positions_true = generate_trajectory(initial_state, T, dt)
        dataset.append((positions, model_labels))
        dataset_1.append((positions_true, model_labels))
    return dataset, dataset_1



# 保存数据集
def save_dataset(dataset, filename):
     # 提取轨迹数据和标签
    trajectories = [item[0] for item in dataset]
    model_labels = [item[1] for item in dataset]
    
    # 转换为 NumPy 数组
    trajectories_array = np.array(trajectories)
    model_labels_array = np.array(model_labels)
    
    # 保存为一个 NumPy 文件
    np.savez(filename, trajectories=trajectories_array, labels=model_labels_array)

# 主函数
if __name__ == "__main__":
    dataset, dataset_true = generate_dataset(NUM_TRAJECTORIES, int(T_MAX / DT), DT)
    save_dataset(dataset, "data/3D_mtdataset.npz")  
    save_dataset(dataset_true, "data/3D_mtdataset_true.npz")
    print(f"Generated {NUM_TRAJECTORIES} trajectories and saved to '3D_mtdataset.npz'")