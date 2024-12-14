import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

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
        # 迷宫布局 - 两个难度使用相同的
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
        
        # 根据难度���整学习参数
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
        
        # 如果已经在目标位置,则不能移动
        if (x, y) in self.goals:
            return (x, y)
        
        # 滑行直到撞墙或到达边界
        while True:
            new_x = x + dx
            new_y = y + dy
            
            # 检查是否出界或撞墙
            if (new_x < 0 or new_x >= self.height or 
                new_y < 0 or new_y >= self.width or 
                self.maze[new_x][new_y] == 1):
                # 如果最后位置是目标,则返回该位置
                if (x, y) in self.goals:
                    return (x, y)
                # 否则返回上一个位置
                return (x, y)
            
            x, y = new_x, new_y
    
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
    
    def train(self, episodes=2000, batch_size=64, gamma=0.99, epsilon_start=0.9):
        """训练DQN"""
        # 据难度调整参数
        if self.difficulty == 'hard':
            epsilon_start = 0.95
            epsilon_decay = 0.997  # 更慢的探索率衰减
        else:
            epsilon_decay = 0.995
        
        epsilon = epsilon_start
        running_reward = 0
        running_length = 100  # 改回100
        episode_rewards = []
        best_path = None
        best_path_length = float('inf')
        best_path_lengths = []  # 记录每100步的最优路径长度
        
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
            
            # 每100个episode记录并输出
            if (episode + 1) % 100 == 0:
                best_path_lengths.append(best_path_length)
                print(f"Episode {episode + 1}: Best path length: {best_path_length}")
                print(f"Running reward: {running_reward:.2f}")
                print(f"Epsilon: {epsilon:.3f}")
                
                # 更新目标网络
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # 记录最短路径
            if done and len(path) < best_path_length:
                best_path = path
                best_path_length = len(path)
            
            # 根据难度使用不同的探索率衰减
            epsilon = max(0.01, epsilon * epsilon_decay)
        
        # 绘制最优路径长度变化图
        plt.figure()
        plt.plot(range(100, episodes + 1, 100), best_path_lengths)
        plt.xlabel('Episodes')
        plt.ylabel('Best Path Length')
        plt.title(f'DQN Best Path Length vs Episodes ({self.difficulty})')
        plt.savefig(f'dqn_{self.difficulty}_path_lengths.png')
        plt.close()
        
        return best_path, best_path_lengths
    
    def visualize_path(self, path):
        """可视化路径"""
        # 创建图形
        plt.figure(figsize=(10, 10))
        
        # 绘制网格底色
        plt.imshow(self.maze, cmap='binary', alpha=0.2)
        
        # 绘制网格线
        for i in range(self.height + 1):
            plt.axhline(y=i - 0.5, color='gray', linestyle='-', alpha=0.3)
        for j in range(self.width + 1):
            plt.axvline(x=j - 0.5, color='gray', linestyle='-', alpha=0.3)
        
        # 绘制墙壁
        wall_positions = np.where(self.maze == 1)
        plt.scatter(wall_positions[1], wall_positions[0], 
                   color='black', marker='s', s=500)
        
        # 绘制路径箭头和数字
        path_coords = np.array(path)
        for i in range(len(path)-1):
            current = path[i]
            next_pos = path[i+1]
            
            # 只有当位置发生变化时才画箭头
            if current != next_pos:
                # 计算箭头方向
                dy = next_pos[1] - current[1]
                dx = next_pos[0] - current[0]
                
                # 绘制箭头
                plt.arrow(current[1], current[0], 
                         dy*0.45, dx*0.45,  # 缩短箭头长度
                         head_width=0.2, 
                         head_length=0.2, 
                         fc='blue', 
                         ec='blue',
                         alpha=0.8,
                         length_includes_head=True,
                         width=0.1)  # 加粗箭头
                
                # 在路径点上标记序号
                if current != self.start and current not in self.goals:
                    plt.text(current[1], current[0], str(i), 
                            ha='center', va='center', 
                            color='black',
                            fontweight='bold',
                            fontsize=12,
                            bbox=dict(facecolor='white', 
                                    edgecolor='blue',
                                    alpha=0.7))
        
        # 标记起点和终点
        plt.scatter(self.start[1], self.start[0], 
                   color='lime', marker='o', s=500, label='Start')
        plt.text(self.start[1], self.start[0], 'S', 
                ha='center', va='center', 
                color='white', 
                fontweight='bold',
                fontsize=14)
        
        for goal in self.goals:
            plt.scatter(goal[1], goal[0], 
                       color='red', marker='o', s=500, label='Goal')
            plt.text(goal[1], goal[0], 'G', 
                    ha='center', va='center', 
                    color='white', 
                    fontweight='bold',
                    fontsize=14)
        
        # 设置图形属性
        plt.grid(True, alpha=0.3)
        plt.title(f'DQN Optimal Path ({self.difficulty})', 
                 pad=20, fontsize=14, fontweight='bold')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=2, fontsize=12)
        
        # 调整坐标轴
        plt.xlim(-0.5, self.width - 0.5)
        plt.ylim(self.height - 0.5, -0.5)  # 翻转y轴使得原点在左上角
        
        # 移除坐标轴刻度
        plt.xticks([])
        plt.yticks([])
        
        # 保存图片,增加分辨率
        plt.savefig(f'dqn_{self.difficulty}_optimal_path.png', 
                    bbox_inches='tight', 
                    dpi=300,
                    pad_inches=0.2)
        plt.close()

def main():
    # 训练简单地图
    print("\nTraining on Easy Map...")
    agent_easy = FrozenLakeDQN(difficulty='easy')
    best_path_easy, history_easy = agent_easy.train(episodes=2000)  # 改为2000轮
    agent_easy.visualize_path(best_path_easy)
    print(f"Easy Map Path length: {len(best_path_easy)-1} steps")
    
    # 训练困难地图
    print("\nTraining on Hard Map...")
    agent_hard = FrozenLakeDQN(difficulty='hard')
    best_path_hard, history_hard = agent_hard.train(episodes=2000)  # 保持2000轮
    agent_hard.visualize_path(best_path_hard)
    print(f"Hard Map Path length: {len(best_path_hard)-1} steps")

if __name__ == "__main__":
    main()
