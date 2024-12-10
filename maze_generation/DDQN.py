import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

# 定义经验回放的数据结构
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

class FrozenLakeDQN:
    def __init__(self, difficulty='easy'):
        # 迷宫布局 - 两个难度使用相同的地图
        self.maze = np.array([
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])
        
        # 根据难度设置不同的目标位置
        if difficulty == 'easy':
            self.goals = [(6, 2)]  # 简单地图目标位置
            self.target_reward = 50  # 简单地图的目标奖励
        else:  # hard
            self.goals = [(6, 5)]  # 困难地图目标位置
            self.target_reward = 100  # 困难地图的目标奖励
        
        self.start = (0, 0)
        self.height, self.width = self.maze.shape
        self.n_actions = 4  # 上右下左
        self.difficulty = difficulty
        
        # DQN相关参数
        self.state_size = 2  # (x, y)坐标
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.policy_net = DQN(self.state_size, self.n_actions).to(self.device)
        self.target_net = DQN(self.state_size, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 根据难度调整学习参数
        if difficulty == 'easy':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
            self.memory = ReplayBuffer(5000)
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
            self.memory = ReplayBuffer(10000)
        
        # 动作对应的方向
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
    
    def get_state_tensor(self, state):
        """将状态转换为张量"""
        return torch.FloatTensor(state).to(self.device)
    
    def get_next_position(self, pos, action):
        """计算在给定位置采取动作后的下一个位置（滑行）"""
        dx, dy = self.actions[action]
        x, y = pos
        
        while True:
            new_x = x + dx
            new_y = y + dy
            
            if (new_x < 0 or new_x >= self.height or 
                new_y < 0 or new_y >= self.width or 
                self.maze[new_x][new_y] == 1):
                return (x, y)
            
            x, y = new_x, new_y
            
            if (x, y) in self.goals:
                return (x, y)
    
    def get_reward(self, pos):
        """获取奖励"""
        if pos in self.goals:
            return self.target_reward  # 使用难度对应的目标奖励
        return -1
    
    def select_action(self, state, epsilon):
        """选择动作"""
        if random.random() > epsilon:
            with torch.no_grad():
                state_tensor = self.get_state_tensor(state)
                return self.policy_net(state_tensor).max(0)[1].item()
        else:
            return random.randrange(self.n_actions)
    
    def optimize_model(self, batch_size, gamma):
        """优化模型"""
        if len(self.memory) < batch_size:
            return
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # 计算下一状态的最大Q值
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q = reward_batch + gamma * next_q * (1 - done_batch)
        
        # 计算Huber损失
        loss = nn.SmoothL1Loss()(current_q, expected_q.unsqueeze(1))
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, episodes=1000, batch_size=64, gamma=0.99, epsilon_start=0.9):
        """训练DQN"""
        # 根据难度调整参数
        if self.difficulty == 'hard':
            epsilon_start = 0.95
            epsilon_decay = 0.997  # 更慢的探索率衰减
        else:
            epsilon_decay = 0.995
        
        epsilon = epsilon_start
        running_reward = 0
        running_length = 100
        episode_rewards = []
        best_path = None
        best_path_length = float('inf')
        
        for episode in range(episodes):
            state = self.start
            path = [state]
            episode_reward = 0
            done = False
            
            while not done and len(path) < 100:
                action = self.select_action(state, epsilon)
                next_state = self.get_next_position(state, action)
                reward = self.get_reward(next_state)
                done = next_state in self.goals
                
                self.memory.push(state, action, reward, next_state, done)
                loss = self.optimize_model(batch_size, gamma)
                
                state = next_state
                path.append(state)
                episode_reward += reward
            
            # 更新running reward
            episode_rewards.append(episode_reward)
            if len(episode_rewards) > running_length:
                episode_rewards.pop(0)
            running_reward = np.mean(episode_rewards)
            
            # 每100个episode报告running reward
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}: Running reward: {running_reward:.2f}")
                print(f"Epsilon: {epsilon:.3f}")
                
                # 更新目标网络
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # 记录最短路径
            if done and len(path) < best_path_length:
                best_path = path
                best_path_length = len(path)
            
            # 根据难度使用不同的探索率衰减
            epsilon = max(0.01, epsilon * epsilon_decay)
        
        return best_path
    
    def visualize_path(self, path):
        """可视化路径"""
        maze_vis = np.array(self.maze, dtype=str)
        maze_vis[maze_vis == '0'] = '.'
        maze_vis[maze_vis == '1'] = '#'
        maze_vis[maze_vis == '2'] = 'G'
        
        for i, (x, y) in enumerate(path):
            if (x, y) not in self.goals:
                maze_vis[x][y] = str(i)
        
        print("\nOptimal path:")
        for row in maze_vis:
            print(' '.join(row))

def main():
    # 训练简单地图
    print("\nTraining on Easy Map (Target at (2,6))...")
    agent_easy = FrozenLakeDQN(difficulty='easy')
    best_path_easy = agent_easy.train(episodes=1000)
    
    print("\nEasy Map Results:")
    agent_easy.visualize_path(best_path_easy)
    print(f"Path length: {len(best_path_easy)-1} steps")
    
    # 训练困难地图
    print("\nTraining on Hard Map (Target at (5,6))...")
    agent_hard = FrozenLakeDQN(difficulty='hard')
    best_path_hard = agent_hard.train(episodes=2000)  # 困难地图多训练一些
    
    print("\nHard Map Results:")
    agent_hard.visualize_path(best_path_hard)
    print(f"Path length: {len(best_path_hard)-1} steps")

if __name__ == "__main__":
    main()
