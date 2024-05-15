import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import random
import torch.nn.functional as F
from solitaire_environment import SolitaireEnvironment
import math
from itertools import count
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import sys
#check the names of other folders in the runs directory
list_of_files = os.listdir('runs')
#check the last number of the folder name
last_number = int(list_of_files[-1][-1])
#add one to the last number
new_number = last_number + 1
folder_name = 'runs/solitaire_experiment_' + str(new_number)
writer = SummaryWriter(folder_name)
# torch.manual_seed(923817727448500)
# random.seed(0)
# np.random.seed(0)


# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
STATE_SIZE = 26624
TARGET_UPDATE = 10
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, STATE_SIZE, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x.view(-1, STATE_SIZE)  # reshape x to have a size of STATE_SIZE
        x = x.float()  # convert x to Float
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def get_state_representation(env):
    state_representation = np.zeros((STATE_SIZE // 4, 4))  # 4 positions for each card: tableau, foundation, waste, stock

    # Encode cards in the tableau
    for pile_index, pile in enumerate(env.table):
        for card in pile:
            if card.face_up:
                index = card.suit * 13 + card.value - 1
                state_representation[index][0] = pile_index + 1  # Use pile index to encode position in tableau

    # Encode cards in the foundation
    for pile_index, pile in enumerate(env.top):
        for card in pile:
            index = card.suit * 13 + card.value - 1
            state_representation[index][1] = pile_index + 1

    # Encode the waste pile
    for group in env.deck.faceup:
        for card in group:
            index = card.suit * 13 + card.value - 1
            state_representation[index] = 3

    # Encode the remaining deck
    for group in env.deck.facedown:
        index = card.suit * 13 + card.value - 1
        state_representation[index] = 4

    return state_representation.flatten()  # Flatten the array to 1D

def select_action(q_values, epsilon, num_actions):
    if random.random() > epsilon:
        return q_values.max() # Exploit: choose the best action
    else:
        return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long)  # Explore: choose a random action

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
env = SolitaireEnvironment()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
n_actions = 110  # Set this to the actual number of possible actions

policy_net = DQN(STATE_SIZE, n_actions).to(device)
target_net = DQN(STATE_SIZE, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Convert batch-array of Transitions to Transition of batch-arrays
    batch = Transition(*zip(*transitions))

 # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([torch.from_numpy(s) for s in batch.next_state if s is not None])
    state_batch = torch.cat([torch.from_numpy(s) for s in batch.state])
    action_batch = torch.stack([torch.tensor(a).view(1) for a in batch.action]).long()
    reward_batch = torch.stack([torch.tensor(r) for r in batch.reward])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    writer.add_scalar('Loss', loss.item(), steps_done)


num_episodes = 50
steps_done = 0
for episode in range(num_episodes):
    env.reset()
    state = get_state_representation(env)
    total_reward = 0
    for t in count():
        while True:
            # Select and perform an action
            action = select_action(state, EPS_START, n_actions)
            reward, done, return_dict = env.step(action.item())
            try:
                if return_dict["error"] == "Invalid move":
                    continue
                else:
                    break
            except:
                break
        next_state = get_state_representation(env)
        total_reward += reward
        writer.add_scalar('Total_reward', total_reward, steps_done)
        writer.add_scalar('Reward', reward, steps_done)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        steps_done += 1
        if t % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break
    # Update the DQN model
    optimize_model()

    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
writer.close()
