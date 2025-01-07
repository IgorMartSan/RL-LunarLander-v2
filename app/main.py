import copy
import gym
import torch

import numpy as np
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from IPython.display import HTML
from base64 import b64encode

from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)


#Create Deep Q Learning


class DQN (nn.Module):
    def __init__(self, n_actions, hidden_size, obs_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size), # Camada linear com 4 entradas e 8 saídas
            nn.ReLU(), # Função de ativação ReLU
            nn.Linear(hidden_size, hidden_size),# Camada linear com 4 entradas e 8 saídas
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self,x):
        return self.net(x.float())
    
## Create a Policy
#  técnica amplamente usada em Reinforcement Learning (RL) para balancear exploração e exploração durante o treinamento de agentes.
def epsilon_greedy(state, env, net, epsilon=0.0):
    if np.random.random() < epsilon:
        action = env.action_space.sample()  # Escolhe uma ação aleatória
    else:
        state = torch.tensor([state]).to(device)  # Converte o estado para tensor e move para o dispositivo (CPU/GPU)
        q_values = net(state)  # Calcula os valores Q(s, a) usando a rede neural
        _, action = torch.max(q_values, dim=1)  # Seleciona a ação com o maior valor Q
        action = int(action.item())  # Converte a ação para um inteiro
    return action  # Retorna a ação escolhida


#Create Buffer replay
#O principal motivo do uso do Replay Buffer em algoritmos de Reinforcement Learning (RL), especialmente no Deep Q-Learning (DQN), 
#é quebrar a correlação entre as amostras de treinamento, 
#o que leva a um aprendizado mais estável e eficiente. 
#Ele também permite o reaproveitamento de experiências, reduzindo a necessidade de interagir continuamente com o ambiente.

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)




class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size = 200): 
        self.buffer = buffer
        self.sample_size = sample_size

        
    

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience



##Create Enviroment

def create_enviroment(name):
    env = gym.make(name)
    env = RecordVideo(env, video='./videos')
    return env


env = create_enviroment("LunarLander-v2")

env.active_space



