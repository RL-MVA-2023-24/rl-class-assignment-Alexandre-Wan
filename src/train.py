import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

config = {
    'nb_actions': env.action_space.n,
    'learning_rate': 0.001,
    'gamma': 0.98,
    'buffer_size': 100_000,
    'epsilon_min': 0.01,
    'epsilon_max': 1.0,
    'epsilon_decay_period': 50_000,
    'epsilon_delay_decay': 100,
    'batch_size': 512,
    'gradient_steps': 3,
    'update_target_strategy': 'replace',  # or 'ema'
    'update_target_freq': 300,
    'update_target_tau': 0.005,
    'criterion': nn.SmoothL1Loss(),
    'hidden_dim': 256,
    'depth': 6 ,
}

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super(DQN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NStepReplayBufferPER:
    def __init__(self, capacity, n_step=3, gamma=0.99, alpha=0.6):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            self.priorities.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sampling_probs = scaled_priorities / sum(scaled_priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probs)
        experiences = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * sampling_probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), indices, np.array(weights)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = np.power(error + 1e-6, self.alpha)
            self.max_priority = max(self.max_priority, self.priorities[idx])

    def __len__(self):
        return len(self.buffer)





# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self, config, input_dim, n_action):
        self.model = DQN(input_dim, config['hidden_dim'], n_action, config['depth']).to(device)
        self.target_model = DQN(input_dim, config['hidden_dim'], n_action, config['depth']).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.memory = NStepReplayBufferPER(config['buffer_size'])
        self.config = config
        self.n_action = n_action
        self.epsilon = config['epsilon_max']

    def greedy_action(self, state):
        with torch.no_grad():
            Q = self.model(torch.tensor(state, dtype=torch.float).to(device))
            return torch.argmax(Q).item()

    def act(self, observation, use_random=False):
        if use_random:
            return random.randrange(self.n_action)
        else:
            return self.greedy_action(observation)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
        }, path)

    def load(self):
        path = './trained_model_final.pth'
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])


def train(agent, env, max_episodes):
    episode_rewards = []
    steps_done = 0
    for episode in range(max_episodes):
        state, _  = env.reset()
        total_reward = 0
        done = False
        trunc = False
        while not (done or trunc):
            action = agent.act(state)
            next_state, reward, done, trunc, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            optimize_model(agent, steps_done)
            steps_done += 1

            agent.epsilon = max(agent.config['epsilon_min'], agent.config['epsilon_max'] -
                                (agent.config['epsilon_max']-agent.config['epsilon_min']) * steps_done / agent.config['epsilon_decay_period'])

            if steps_done % agent.config['update_target_freq'] == 0:
                update_target_model(agent)

        episode_rewards.append(total_reward)
        print(f'Episode {episode}, Total reward: {total_reward}, Epsilon: {agent.epsilon:.2f}')
    return episode_rewards



def optimize_model(agent, steps_done):
    if len(agent.memory) < agent.config['batch_size']:
        return

    beta = 0.4
    states, actions, rewards, next_states, dones, indices, weights = agent.memory.sample(agent.config['batch_size'], beta)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device).unsqueeze(-1)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    weights = torch.FloatTensor(weights).to(device)

    current_q_values = agent.model(states).gather(1, actions).squeeze(-1)
    next_actions = agent.model(next_states).argmax(dim=1, keepdim=True)
    next_q_values = agent.target_model(next_states).gather(1, next_actions).squeeze(-1)
    expected_q_values = rewards + agent.config['gamma'] * next_q_values * (1 - dones)

    loss = (weights * F.smooth_l1_loss(current_q_values, expected_q_values.detach(), reduction='none')).mean()
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    with torch.no_grad():
        td_errors = torch.abs(current_q_values - expected_q_values).cpu().numpy()
    agent.memory.update_priorities(indices, td_errors)

def update_target_model(agent):
    if agent.config['update_target_strategy'] == 'replace':
        agent.target_model.load_state_dict(agent.model.state_dict())
    elif agent.config['update_target_strategy'] == 'ema':
        for target_param, local_param in zip(agent.target_model.parameters(), agent.model.parameters()):
            target_param.data.copy_(agent.config['update_target_tau'] * local_param.data + (1.0 - agent.config['update_target_tau']) * target_param.data)

