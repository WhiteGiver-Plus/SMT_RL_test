import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# DQN网络结构
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# 使用与Q_table_train_roburst.py相同的CustomFrozenLake环境
class CustomFrozenLake(gym.Env):
    def __init__(self):
        super().__init__()
        self.base_env = gym.make('FrozenLake-v1', is_slippery=False).unwrapped
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.slippery_tiles = self._select_slippery_tiles()
        
    def _select_slippery_tiles(self):
        possible_tiles = [1, 2, 3, 4, 6, 8, 9, 10, 13, 14]
        return random.sample(possible_tiles, 2)
    
    def step(self, action):
        next_state, reward, done, truncated, info = self.base_env.step(action)
        
        if not done and next_state in self.slippery_tiles:
            rand = random.random()
            if rand < 1/3:
                if next_state % 4 != 0:
                    next_state -= 1
            elif rand < 2/3:
                if (next_state + 1) % 4 != 0:
                    next_state += 1
            
            if self.base_env.desc.flatten()[next_state] == b'H':
                done = True
                reward = 0
            elif self.base_env.desc.flatten()[next_state] == b'G':
                done = True
                reward = 1
                
            self.base_env.s = next_state
        
        return next_state, reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.slippery_tiles = self._select_slippery_tiles()
        return self.base_env.reset(**kwargs)

class DQNTrainer:
    def __init__(self, is_robust=False, learning_rate=0.001, gamma=0.96, 
                 epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = CustomFrozenLake() if is_robust else gym.make('FrozenLake-v1', is_slippery=False)
        
        self.policy_net = DQN(16, 4).to(self.device)  # 使用one-hot���码表示状态
        self.target_net = DQN(16, 4).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=10000)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 64
        
    def state_to_tensor(self, state):
        # 将状态转换为one-hot向量
        state_tensor = torch.zeros(16)
        state_tensor[state] = 1
        return state_tensor.to(self.device)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([self.state_to_tensor(s) for s, _, _, _, _ in batch])
        actions = torch.tensor([a for _, a, _, _, _ in batch], device=self.device)
        rewards = torch.tensor([r for _, _, r, _, _ in batch], device=self.device, dtype=torch.float32)
        next_states = torch.stack([self.state_to_tensor(s) for _, _, _, s, _ in batch])
        dones = torch.tensor([d for _, _, _, _, d in batch], device=self.device, dtype=torch.float32)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, episodes=20000):
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay()
            
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}/{episodes}")
    
    def save_q_table(self, filename):
        # 将DQN的Q值转换为Q表格式
        q_table = np.zeros((16, 4))
        with torch.no_grad():
            for state in range(16):
                state_tensor = self.state_to_tensor(state)
                q_values = self.policy_net(state_tensor).cpu().numpy()
                q_table[state] = q_values
        np.save(filename, q_table)
    
    def evaluate(self, num_episodes=1000):
        successes = 0
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            truncated = False
            
            while not (done or truncated):
                state_tensor = self.state_to_tensor(state)
                with torch.no_grad():
                    action = self.policy_net(state_tensor).argmax().item()
                state, reward, done, truncated, _ = self.env.step(action)
                if done and reward == 1:
                    successes += 1
        
        return successes / num_episodes

if __name__ == "__main__":
    # 训练普通版本
    print("Training normal version...")
    normal_trainer = DQNTrainer(is_robust=False)
    normal_trainer.train()
    normal_trainer.save_q_table("q_table_dqn_normal.npy")
    success_rate = normal_trainer.evaluate()
    print(f"Normal version success rate: {success_rate:.2%}")
    
    # 训练robust版本
    print("\nTraining robust version...")
    robust_trainer = DQNTrainer(is_robust=True)
    robust_trainer.train()
    robust_trainer.save_q_table("q_table_dqn_robust.npy")
    success_rate = robust_trainer.evaluate()
    print(f"Robust version success rate: {success_rate:.2%}")
