import numpy as np

# Training Parameters
MODEL: str = "matd3"  # options: 'maddpg', 'matd3', 'mappo', 'masac', 'random'
SEED: int = 1234  # random seed for reproducibility
STEPS_PER_EPISODE: int = 50  # total T
LOG_FREQ: int = 1 #10  # episodes
IMG_FREQ: int = 100  # steps
TEST_LOG_FREQ: int = 1  # episodes (for testing)
TEST_IMG_FREQ: int = 100  # steps (for testing)

# Simulation Parameters
NUM_UES: int = 5

# 频率
Hz: float = 1
kHz: float = 1000 * Hz
mHz: float = 1000 * kHz
GHz: float = 1000 * mHz
nor: float = 10**(-7)
nor1: float = 10**19

# 数据大小
BIT: float = 1
B: float = 8 * BIT
KB: float = 1024 * B
MB: float = 1024 * KB

#模拟参数
κ: float = 10 ** (-27)  # 芯片结构对cpu处理的影响因子
# κ_mec: float = 10 ** (-26) #mec服务器芯片结构对cpu处理的影响因子
r: float = 1  #运行语义提取任务的CPU周期数的参数1...."若设为2会导致语义提取的能耗显著增大，不合适。。。"
alpha: float = 1  #运行语义提取任务的CPU周期数的参数2
beta: float = 2  #运行语义提取任务的CPU周期数的参数3
transmission_bandwidth: float = 1 * mHz  # 传输带宽1MHz
# W: float = transmission_bandwidth/agent_num  #每个用户设备的带宽分配
# self.transmission_power = np.random.uniform(0.1, 0.5)  # 传输功率0.1W-0.5W
noise_power: float = 10**(-20) # 噪声功率-170dBm
#不考虑邻道干扰功率
MEC_f: float = 20 * GHz  # MEC的计算能力
# self.weight_factor = 0.7  # 能耗和公平性的权重


#UE Parameters
MIN_SEMANTIC_EXTRACTION_FACTOR: float = 0.3
MAX_SEMANTIC_EXTRACTION_FACTOR: float = 1.0 
MIN_TRANSMIT_POWER: float = 0.1  # in Watts
MAX_TRANSMIT_POWER: float = 0.5  # in Watts
MIN_RESOURCE_ALLOCATION: float = 0.1  # fraction of CPU cycles
MAX_RESOURCE_ALLOCATION: float = 1.0  # fraction of CPU cycles


POWER_MOVE: float = 80.0  # P_move in Watts
POWER_HOVER: float = 100.0  # P_hover in Watts

# Request Parameters
NUM_SERVICES: int = 50  # S
NUM_CONTENTS: int = 100  # K
NUM_FILES: int = NUM_SERVICES + NUM_CONTENTS  # S + K
CPU_CYCLES_PER_BYTE: np.ndarray = np.random.randint(2000, 4000, size=NUM_SERVICES)  # omega_s_m
FILE_SIZES: np.ndarray = np.random.randint(10**5, 5 * 10**5, size=NUM_FILES)  # in bytes
MIN_INPUT_SIZE: int = 1 * 10**5  # in bytes
MAX_INPUT_SIZE: int = 5 * 10**5  # in bytes
ZIPF_BETA: float = 0.6  # beta^Zipf
K_CPU: float = 1e-27  # CPU capacitance coefficient

# Caching Parameters
T_CACHE_UPDATE_INTERVAL: int = 10  # T_cache
GDSF_SMOOTHING_FACTOR: float = 0.5  # beta^gdsf

# Communication Parameters
G_CONSTS_PRODUCT: float = 2.2846 * 1.42 * 1e-4  # G_0 * g_0
TRANSMIT_POWER: float = 0.5  # P in Watts
AWGN: float = 1e-13  # sigma^2
BANDWIDTH_INTER: int = 30 * 10**6  # B^inter in Hz
BANDWIDTH_EDGE: int = 20 * 10**6  # B^edge in Hz
BANDWIDTH_BACKHAUL: int = 40 * 10**6  # B^backhaul in Hz

# Model Parameters

ALPHA_1 = 0.4  # for latency
ALPHA_2 = 0.4  # for energy
ALPHA_3 = 0.2  # for fairness
assert round(ALPHA_1 + ALPHA_2 + ALPHA_3, 3) == 1.0

# OBS_DIM_SINGLE: int = 2 + NUM_FILES + (MAX_UAV_NEIGHBORS * (2 + NUM_FILES)) + (MAX_ASSOCIATED_UES * (2 + 3))
# # own state: pos (2) + cache (NUM_FILES) + Neighbors: pos (2) + cache (NUM_FILES) + UEs: pos (2) + request_tuple (3)
OBS_DIM_SINGLE: int =  3  # 设置智能体的观测维度 ：task_size, computing_density, max_delay
ACTION_DIM: int = 4  # offloading decision, resource allocation, transmit power, SE factor from [-1, 1]
STATE_DIM: int = NUM_UES * OBS_DIM_SINGLE
MLP_HIDDEN_DIM: int = 128

#max-normalization
MAX_TASK_SIZE: float = 2 * MB  # in bytes
MAX_COMPUTING_DENSITY: float = 500  # in cycles/byte
MAX_DELAY: float = 5.0  # in seconds

ACTOR_LR: float = 5e-5 #0.002
CRITIC_LR: float = 5e-5 #0.001
DISCOUNT_FACTOR: float = 0.99  # gamma
UPDATE_FACTOR: float = 0.01  # tau
MAX_GRAD_NORM: float = 0.5  # maximum norm for gradient clipping to prevent exploding gradients
LOG_STD_MAX: float = 2  # maximum log standard deviation for stochastic policies
LOG_STD_MIN: float = -20  # minimum log standard deviation for stochastic policies
EPSILON: float = 1e-9  # small value to prevent division by zero

# Off-policy algorithm hyperparameters
REPLAY_BUFFER_SIZE: int = 10**6  # B
REPLAY_BATCH_SIZE: int = 128  # minibatchbatch size
INITIAL_RANDOM_STEPS: int = 4e3  #TODO 前期探索 steps of random actions for exploration
LEARN_FREQ: int = 5  # steps to learn after

# Gaussian Noise Parameters (for MADDPG and MATD3)
INITIAL_NOISE_SCALE: float = 0.1
MIN_NOISE_SCALE: float = 0.01
NOISE_DECAY_RATE: float = 0.995

# MATD3 Specific Hyperparameters
POLICY_UPDATE_FREQ: int = 1  # delayed policy update frequency
TARGET_POLICY_NOISE: float = 0.2  # standard deviation of target policy smoothing noise.
NOISE_CLIP: float = 0.5  # range to clip target policy smoothing noise

# MAPPO Specific Hyperparameters
PPO_ROLLOUT_LENGTH: int = 2048  # number of steps to collect per rollout before updating
PPO_GAE_LAMBDA: float = 0.95  # lambda parameter for GAE
PPO_EPOCHS: int = 10  # number of epochs to run on the collected rollout data
PPO_BATCH_SIZE: int = 64  # size of mini-batches to use during the update step
PPO_CLIP_EPS: float = 0.2  # clipping parameter (epsilon) for the PPO surrogate objective
PPO_ENTROPY_COEF: float = 0.01  # coefficient for the entropy bonus to encourage exploration

# MASAC Specific Hyperparameters
ALPHA_LR: float = 3e-4  # learning rate for the entropy temperature alpha
