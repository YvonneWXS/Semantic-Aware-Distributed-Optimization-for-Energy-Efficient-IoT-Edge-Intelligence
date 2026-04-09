import copy
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from env import ENV
from replay_buffer import ReplayBuffer
from MADDPG import Maddpg
# from DQN import Double_DQN
from D3QN import D3QN
import csv

learning_start_step = 200
learning_fre = 5
batch_size = 64
gamma = 0.9
lr = 0.01
max_grad_norm = 0.5
save_model = 40
save_dir = "models/simple_adversary"
save_fer = 400
tao = 0.01
memory_size = 1024
EPOCH = 1
STEP = 50

write = SummaryWriter(log_dir="logs")

def train(ue, mec, k, lam):
    """step1:create the environment"""
    u = ue
    m = mec
    k = k
    lam = lam
    env = ENV(u, m, k, lam)    # UE: MEC:, k:
    # maddpg = Maddpg()
    # dqn = Double_DQN(env)
    d3qn = D3QN(env)


    print('=============================')
    print('=1 Env {} is right ...')
    print('=============================')

    """step2:create agent"""
    obs_shape_n = [env.n_features for i in range(env.UEs)]  #列表存储每个用户设备的观测值特征数量
    action_shape_n = [env.n_actions for i in range(env.UEs)]
    # actors_cur, critic_cur, actors_tar, critic_tar, optimizers_a, optimizers_c = \
    #     maddpg.get_train(env, obs_shape_n, action_shape_n)
    # memory_dpg = ReplayBuffer(memory_size)
    memory_dqn = ReplayBuffer(memory_size)

    print('=2 The {} agents are inited ...'.format(env.UEs))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    episode_energy_local, episode_energy_ran, episode_energy_mec = [0.0], [0.0], [0.0]
    epoch_energy_local, epoch_energy_ran, epoch_energy_mec = [], [], []

    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print('=3 starting iterations ...')
    print('=============================')

    np.random.seed(1)
    obs = env.reset() #初始化观测值

    # """创建CSV文件并写入表头"""
    # with open(f'episode_rewards_and_energy_{u}_UEs.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     # writer.writerow(['Episode', 'MADDPG_Reward', 'DQN_Reward', 'D3QN_Reward', 'MADDPG_Energy', 'DQN_Energy', 'D3QN_Energy', 'Local_Energy', 'MEC_Energy', 'Random_Energy'])
    #     writer.writerow(['Episode','local_energy','mec_energy','random_energy'])

    for epoch in range(EPOCH):
        episode_violation_num = 0
        for time_1 in range(STEP):
            # action_prob = [agent(torch.from_numpy(observation).to(torch.float)).detach().cpu().numpy() \
            #             for agent, observation in zip(actors_cur, obs)]
            # action_dqn = dqn.choose_action(obs)
            action_d3qn = d3qn.choose_action(obs)

            o1 = copy.deepcopy(obs)
            o2 = copy.deepcopy(obs)
            obs_old = copy.deepcopy(obs)
            # obs_, rew, energy_dpg, energy_local, energy_mec, energy_ran, info_dpg = env.step(obs, action_prob,epoch,time_1)
            # obs_dqn, rew_dqn, energy_dqn, info_dqn = env.step(o1, action_dqn, epoch, time_1,is_prob=False, is_compared=False)
            obs_d3qn, rew_d3qn, energy_d3qn, energy_local, energy_mec, energy_ran, info_d3qn = env.step(o2, action_d3qn, epoch, time_1, is_prob=False, is_compared=True)

            episode_energy_local[-1] += np.sum(energy_local)
            episode_energy_mec[-1] += np.sum(energy_mec)
            episode_energy_ran[-1] += np.sum(energy_ran)

            game_step += 1
            obs = obs_d3qn

        epoch_energy_local.append(episode_energy_local[-1] / STEP)
        epoch_energy_mec.append(episode_energy_mec[-1] / STEP)
        epoch_energy_ran.append(episode_energy_ran[-1] / STEP)

        episode_energy_local.append(0)
        episode_energy_mec.append(0)
        episode_energy_ran.append(0)

        write.add_scalars("energy", {
                                    'Local': epoch_energy_local[epoch],
                                    'Mec': epoch_energy_mec[epoch],
                                    'random': epoch_energy_ran[epoch]}, epoch)

        
        writer.writerow([
             u,
            epoch,
            epoch_energy_local[epoch], epoch_energy_mec[epoch], epoch_energy_ran[epoch]  # 能耗数据
        ])


if __name__ == '__main__':
        """创建CSV文件并写入表头"""
with open(f'local_random_energy_UEs.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # writer.writerow(['Episode', 'MADDPG_Reward', 'DQN_Reward', 'D3QN_Reward', 'MADDPG_Energy', 'DQN_Energy', 'D3QN_Energy', 'Local_Energy', 'MEC_Energy', 'Random_Energy'])
    writer.writerow(['UE_number','Episode','local_energy','mec_energy','random_energy'])

    for UEs in range(5, 31, 5): 
         train(ue = UEs, mec =1, k=100, lam=0.1)
