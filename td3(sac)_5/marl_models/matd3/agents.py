import config
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ActorNetwork(nn.Module): #input即next_obs_batch过于大需要归一化，
    # def __init__(self, obs_dim: int, action_dim: int) -> None:
    #     super(ActorNetwork, self).__init__()
    #     self.fc1: nn.Linear = nn.Linear(obs_dim, config.MLP_HIDDEN_DIM)
    #     self.fc2: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)
    #     self.out: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, action_dim)

    # def forward(self, input: torch.Tensor) -> torch.Tensor:
    #     x: torch.Tensor = F.relu(self.fc1(input)) 
    #     x = F.relu(self.fc2(x))
    #     return torch.tanh(self.out(x))
    
    'layer normalization不可用尺度差异太大  /max_value归一化试试'
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super(ActorNetwork, self).__init__()

        # ====== 加入 obs 最大值，用于归一化 ======
        self.obs_max = torch.tensor([
            config.MAX_TASK_SIZE,
            config.MAX_COMPUTING_DENSITY,
            config.MAX_DELAY
        ], dtype=torch.float32)

        self.fc1 = nn.Linear(obs_dim, config.MLP_HIDDEN_DIM)
        self.norm1 = nn.LayerNorm(config.MLP_HIDDEN_DIM)

        self.fc2 = nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)
        self.norm2 = nn.LayerNorm(config.MLP_HIDDEN_DIM)

        self.out = nn.Linear(config.MLP_HIDDEN_DIM, action_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # ====== ⭐ 第一步：对 obs 做归一化到 [-1, 1] ======
        # 为确保运行在 GPU 上，需要 to(input.device)
        input = input / self.obs_max.to(input.device)

        x = self.fc1(input)
        # print(torch.isnan(self.fc1.weight).any(), torch.isnan(self.fc1.bias).any())
        x = self.norm1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)

        return torch.tanh(self.out(x))



class CriticNetwork(nn.Module):
    def __init__(self, total_obs_dim: int, total_action_dim: int) -> None:
        super(CriticNetwork, self).__init__()
        self.fc1: nn.Linear = nn.Linear(total_obs_dim + total_action_dim, config.MLP_HIDDEN_DIM)
        self.fc2: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM)
        self.out: nn.Linear = nn.Linear(config.MLP_HIDDEN_DIM, 1)

    def forward(self, joint_obs: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = torch.cat([joint_obs, joint_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
