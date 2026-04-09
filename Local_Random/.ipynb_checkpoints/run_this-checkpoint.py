import copy
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from env import ENV
from replay_buffer import ReplayBuffer
from MADDPG import Maddpg
from DQN import Double_DQN
from D3QN import D3QN

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
memory_size = 2000  
EPOCH = 5000
STEP = 100

write = SummaryWriter(log_dir="logs")

# C:/Users/CCHONG/Desktop/Science/Edge Computing/codes/my_codes/baseline0/Multi-Agent-Mec-DRL/
def train(ue=3, mec=1, k=100, lam=0.5):
    """step1:create the environment"""
    u = ue
    m = mec
    k = k
    lam = lam
    env = ENV(u, m, k, lam)    # UE: MEC:, k:
    maddpg = Maddpg()
    dqn = Double_DQN(env)
    d3qn = D3QN(env)


    print('=============================')
    print('=1 Env {} is right ...')
    print('=============================')

    """step2:create agent"""
    obs_shape_n = [env.n_features for i in range(env.UEs)]  #列表存储每个用户设备的观测值特征数量
    action_shape_n = [env.n_actions for i in range(env.UEs)]
    actors_cur, critic_cur, actors_tar, critic_tar, optimizers_a, optimizers_c = \
        maddpg.get_train(env, obs_shape_n, action_shape_n)
    memory_dpg = ReplayBuffer(memory_size)
    memory_dqn = ReplayBuffer(memory_size)

    print('=2 The {} agents are inited ...'.format(env.UEs))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    update_cnt = 0
    episode_rewards, episode_dqn, episode_d3qn, episode_local,  episode_mec, episode_ran = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0] # sum of rewards for all agents
    episode_time_dpg,  episode_time_dqn, episode_time_d3qn, episode_time_local, episode_time_ran, episode_time_mec = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    episode_energy_dpg, episode_energy_dqn, episode_energy_d3qn, episode_energy_local, episode_energy_ran, episode_energy_mec = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    epoch_energy_dpg, epoch_energy_dqn, epoch_energy_d3qn, epoch_energy_local, epoch_energy_ran, epoch_energy_mec = [], [], [], [], [], []
    episode_total_cost = [0.0]
    episode_rewards_dpg, episode_rewards_dqn, episode_rewards_d3qn = [0.0], [0.0], [0.0]
    epoch_rewards_dpg, epoch_rewards_dqn, epoch_rewards_d3qn = [], [], []

    # agent_rewards = [[0.0] for _ in range(env.UEs)]  # individual agent reward
    # episode_fairness_dpg, episode_fairness_local, episode_fairness_mec, episode_fairness_ran,episode_fairness_dqn, episode_fairness_d3qn = [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
    # epoch_fairness_dpg, epoch_fairness_local, epoch_fairness_mec, epoch_fairness_ran,epoch_fairness_dqn,epoch_fairness_d3qn = [],[],[],[],[],[]
    epoch_reward = []
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

    # print(np.random.randint(1, 100))  # 先打印第一个随机数
    # print(np.random.uniform(0, 1))    # 再打印第二个随机数
    np.random.seed(1)
    obs = env.reset() #初始化观测值
    for epoch in range(EPOCH):
        for time_1 in range(STEP):
            action_prob = [agent(torch.from_numpy(observation).to(torch.float)).detach().cpu().numpy() \
                        for agent, observation in zip(actors_cur, obs)]
            action_dqn = dqn.choose_action(obs)
            action_d3qn = d3qn.choose_action(obs)

            o1 = copy.deepcopy(obs)
            o2 = copy.deepcopy(obs)
            obs_old = copy.deepcopy(obs)
            obs_, rew, energy_dpg, energy_local, energy_mec, energy_ran= env.step(obs, action_prob,epoch,time_1)
            obs_dqn, rew_dqn, energy_dqn = env.step(o1, action_dqn, epoch, time_1,is_prob=False, is_compared=False)
            obs_d3qn, rew_d3qn, energy_d3qn = env.step(o2, action_d3qn,epoch, time_1, is_prob=False, is_compared=False)


            # #计算公平性指标reward
            # def jain_fairness(energy_list):
            #     energy = np.array(energy_list)
            #     sum_sq = np.sum(energy)**2
            #     sum_energy_sq = np.sum(energy**2)
            #     return sum_sq / (len(energy) * sum_energy_sq)
            # rew_dqn= [jain_fairness(energy_dpg), jain_fairness(energy_dpg),jain_fairness(energy_dpg)]
            # rew_d3qn= [jain_fairness(energy_d3qn), jain_fairness(energy_d3qn),jain_fairness(energy_d3qn)]

            # save the experience
            memory_dpg.add(obs_old, np.concatenate(action_prob), rew, obs_)
            dqn.store_memory(obs_old, action_dqn, rew_dqn, obs_dqn)
            d3qn.store_memory(obs_old, action_d3qn, rew_d3qn, obs_d3qn)

            #REWARD
            episode_rewards_dpg[-1] += np.sum(rew)
            episode_rewards_dqn[-1]+=np.sum(rew_dqn)
            episode_rewards_d3qn[-1]+=np.sum(rew_d3qn)

            episode_energy_dpg[-1] += np.sum(energy_dpg)
            episode_energy_dqn[-1] += np.sum(energy_dqn)
            episode_energy_d3qn[-1] += np.sum(energy_d3qn)
            episode_energy_local[-1] += np.sum(energy_local)
            episode_energy_mec[-1] += np.sum(energy_mec)
            episode_energy_ran[-1] += np.sum(energy_ran)

            # train agent
            if game_step > 1000 and game_step % 100 == 0:
                update_cnt, actors_cur, actors_tar, critic_cur, critic_tar = maddpg.agents_train(
                    game_step, update_cnt, memory_dpg, obs_size, action_size,
                    actors_cur, actors_tar, critic_cur, critic_tar, optimizers_a, optimizers_c, write)
                dqn.learn(game_step, write)
                d3qn.learn(game_step, write)

            # update obs
            game_step += 1
            obs = obs_
        # print("epoch:{},MADDPG:{}".format(epoch, episode_rewards[-1]))    

        epoch_energy_d3qn.append(episode_energy_d3qn[-1] / STEP)
        epoch_energy_dpg.append(episode_energy_dpg[-1] / STEP)
        epoch_energy_dqn.append(episode_energy_dqn[-1] / STEP)
        epoch_energy_local.append(episode_energy_local[-1] / STEP)
        epoch_energy_mec.append(episode_energy_mec[-1] / STEP)
        epoch_energy_ran.append(episode_energy_ran[-1] / STEP)

        episode_energy_d3qn.append(0)
        episode_energy_dpg.append(0)
        episode_energy_dqn.append(0)
        episode_energy_local.append(0)
        episode_energy_mec.append(0)
        episode_energy_ran.append(0)

        write.add_scalars("energy", {'MADDPG': epoch_energy_dpg[epoch],
                                    'DQN': epoch_energy_dqn[epoch],
                                    'D3QN': epoch_energy_d3qn[epoch],
                                    'Local': epoch_energy_local[epoch],
                                    'Mec': epoch_energy_mec[epoch],
                                    'random': epoch_energy_ran[epoch]}, epoch)

        epoch_rewards_dpg.append(episode_rewards_dpg[-1] / STEP)
        epoch_rewards_dqn.append(episode_rewards_dqn[-1] / STEP)
        epoch_rewards_d3qn.append(episode_rewards_d3qn[-1] / STEP)
        episode_rewards_dpg.append(0)
        episode_rewards_dqn.append(0)
        episode_rewards_d3qn.append(0)

        write.add_scalars("reward", {'MADDPG': epoch_rewards_dpg[epoch],
                                    'DQN': epoch_rewards_dqn[epoch],
                                    'D3QN': epoch_rewards_d3qn[epoch]}, epoch)

        # epoch_fairness_dpg.append(episode_fairness_dpg[-1] / STEP)
        # epoch_fairness_local.append(episode_fairness_local[-1] / STEP)
        # epoch_fairness_mec.append(episode_fairness_mec[-1] / STEP)
        # epoch_fairness_ran.append(episode_fairness_ran[-1] / STEP)
        # epoch_fairness_dqn.append(episode_fairness_dqn[-1] / STEP)
        # epoch_fairness_d3qn.append(episode_fairness_d3qn[-1] / STEP)
        # epoch_reward.append(episode_rewards[-1] / STEP)

        # episode_fairness_dpg.append(0)
        # episode_fairness_local.append(0)
        # episode_fairness_mec.append(0)
        # episode_fairness_ran.append(0)
        # episode_fairness_dqn.append(0)
        # episode_fairness_d3qn.append(0)
        # episode_rewards.append(0)
        

        # write.add_scalars("fairness", {'MADDPG': epoch_fairness_dpg[epoch],
        #                            'Local': epoch_fairness_local[epoch],
        #                            'Mec': epoch_fairness_mec[epoch],
        #                            'random': epoch_fairness_ran[epoch],
        #                            'DQN': epoch_fairness_dqn[epoch],
        #                            'D3QN': epoch_fairness_d3qn[epoch]}, epoch)
        # write.add_scalars("reward", {'MADDPG': epoch_reward[epoch]}, epoch)



        # episode_rewards.append(0)   #初始化
        # episode_dqn.append(0)
        # episode_d3qn.append(0)
        # episode_local.append(0)
        # episode_mec.append(0)
        # episode_ran.append(0)

        # episode_time_dpg.append(0)
        # episode_time_dqn.append(0)
        # episode_time_d3qn.append(0)
        # episode_time_local.append(0)
        # episode_time_mec.append(0)
        # episode_time_ran.append(0)
        # episode_energy_dpg.append(0)
        # episode_energy_dqn.append(0)
        # episode_energy_d3qn.append(0)
        # episode_energy_local.append(0)
        # episode_energy_mec.append(0)
        # episode_energy_ran.append(0)

        # episode_total_cost.append(0)
        # # for a_r in agent_rewards:
        # #     a_r.append(0)
        # # print("------reset-------")
        # write.add_scalars("cost", {'MADDPG': epoch_average_total_cost[epoch],
        #                            'DQN': epoch_average_dqn[epoch],
        #                            'D3QN': epoch_average_d3qn[epoch],
        #                            'Local': epoch_average_local[epoch],
        #                            'Mec': epoch_average_mec[epoch],
        #                            'random': epoch_average_ran[epoch]}, epoch)
        # # write.add_scalars("cost", {'MADDPG': - episode_rewards[-1] /STEP,
        # #                            # 'DQN': epoch_average_dqn[epoch],
        # #                            'Local': episode_local[-1] / STEP,
        # #                            'Mec': episode_mec[-1] / STEP,
        # #                            'random': episode_ran[-1] / STEP}, epoch)
        # write.add_scalars("cost/energy", {'MADDPG': epoch_average_energy_reward[epoch],
        #                              'DQN': epoch_average_energy_dqn[epoch],
        #                              'D3QN': epoch_average_energy_d3qn[epoch],
        #                              'Local': epoch_average_energy_local[epoch],
        #                              'Mec': epoch_average_energy_mec[epoch],
        #                              'random': epoch_average_energy_ran[epoch]}, epoch)
        # write.add_scalars("cost/delay", {'MADDPG': epoch_average_time_reward[epoch],
        #                             'DQN': epoch_average_time_dqn[epoch],
        #                             'D3QN': epoch_average_time_d3qn[epoch],
        #                             'Local': epoch_average_time_local[epoch],
        #                             'Mec': epoch_average_time_mec[epoch],
        #                             'random': epoch_average_time_ran[epoch]}, epoch)

        # print("epoch:{},MADDPG:{}".format(epoch, epoch_average_total_cost[epoch]))
        # # print("epoch:{},DQN:{}".format(epoch, epoch_average_dqn[epoch]))
        # print("epoch:{},Local:{}".format(epoch, epoch_average_local[epoch]))
        # print("epoch:{},Mec:{}".format(epoch, epoch_average_mec[epoch]))
        # print("epoch:{},random:{}".format(epoch, epoch_average_ran[epoch]))
        # if epoch_average_mec[epoch] > epoch_average_reward[epoch]:
        #     print("True")
        # print("---------------------------------------")
    # return a
        print ("epoch", epoch, "is finished")
        print("epoch_rewards_dpg:",epoch_rewards_dpg[epoch])
        print("epoch_rewards_dqn:",epoch_rewards_dqn[epoch])
        print("epoch_rewards_d3qn:",epoch_rewards_d3qn[epoch])





if __name__ == '__main__':
    # for i in range(5):
    #     cost = train(i + 10)
    #     print(i + 10, "cost:", cost)
    #     write.add_scalar("cost", cost, i + 10)
    #     write.close()
    train()
