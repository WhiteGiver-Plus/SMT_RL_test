import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Dict

class DCEnvironment:
    def __init__(self, env_data: Dict):
        self.server_capacities = env_data['server_capacities']
        self.task_demands = env_data['task_demands']
        self.dependencies = env_data['dependencies']
        self.time_horizon = env_data['time_horizon']
        
        self.num_servers = len(self.server_capacities)
        self.num_tasks = len(self.task_demands)
        
        # 当前时间步
        self.current_time = 0
        # 任务分配状态
        self.task_placements = {}
        # 服务器当前资源使用情况
        self.server_usage = np.zeros_like(self.server_capacities)
        # 待处理任务队列
        self.pending_tasks = set(range(self.num_tasks))
        
    def reset(self):
        """重置环境状态"""
        self.current_time = 0
        self.task_placements = {}
        self.server_usage = np.zeros_like(self.server_capacities)
        self.pending_tasks = set(range(self.num_tasks))
        return self._get_state()
    
    def step(self, action: Tuple[int, int]):
        """执行一步调度动作
        action: (task_id, server_id)
        """
        task_id, server_id = action
        reward = 0
        done = False
        
        # 检查动作是否有效
        if not self._is_valid_action(task_id, server_id):
            reward = -100
            return self._get_state(), reward, True
        
        # 执行任务分配
        self.task_placements[task_id] = server_id
        self.pending_tasks.remove(task_id)
        
        # 更新服务器资源使用情况
        task = self.task_demands[task_id]
        self.server_usage[server_id] += task['resources']
        
        # 计算奖励
        reward = self._calculate_reward(task_id, server_id)
        
        # 检查是否完成所有任务
        if len(self.pending_tasks) == 0:
            done = True
            reward += 100  # 完成所有任务的奖励
            
        return self._get_state(), reward, done
    
    def _get_state(self):
        """获取当前环境状态"""
        state = np.concatenate([
            self.server_usage.flatten(),
            np.array([1 if i in self.pending_tasks else 0 
                     for i in range(self.num_tasks)])
        ])
        return state
    
    def _is_valid_action(self, task_id: int, server_id: int) -> bool:
        """检查动作是否有效"""
        if task_id not in self.pending_tasks:
            return False
            
        # 检查依赖是否满足
        for dep_task, target_task in self.dependencies:
            if target_task == task_id and dep_task in self.pending_tasks:
                return False
                
        # 检查资源是否足够
        task = self.task_demands[task_id]
        new_usage = self.server_usage[server_id] + task['resources']
        if np.any(new_usage > self.server_capacities[server_id]):
            return False
            
        return True
    
    def _calculate_reward(self, task_id: int, server_id: int) -> float:
        """计算奖励"""
        reward = 0
        
        # 资源利用率奖励
        utilization = np.mean(self.server_usage[server_id] / 
                            self.server_capacities[server_id])
        reward += utilization * 10
        
        # 负载均衡奖励
        std_util = np.std([np.mean(usage / capacity) 
                          for usage, capacity in 
                          zip(self.server_usage, self.server_capacities)])
        reward -= std_util * 5
        
        return reward

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                  lr=self.learning_rate)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
            
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([s[0] for s in batch]).to(self.device)
        actions = torch.LongTensor([s[1] for s in batch]).to(self.device)
        rewards = torch.FloatTensor([s[2] for s in batch]).to(self.device)
        next_states = torch.FloatTensor([s[3] for s in batch]).to(self.device)
        dones = torch.FloatTensor([s[4] for s in batch]).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, path: str):
        """保存模型权重"""
        torch.save(self.policy_net.state_dict(), path)
        
    def load_model(self, path: str):
        """加载模型权重"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_agent(env, agent, episodes=1000, batch_size=32):
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            task_id = action // env.num_servers
            server_id = action % env.num_servers
            
            next_state, reward, done = env.step((task_id, server_id))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay(batch_size)
            
        if episode % 10 == 0:
            agent.update_target_network()
            
        scores.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
    return scores

def main():
    from dc_generator import DataCenterGenerator
    
    # 创建环境
    generator = DataCenterGenerator()
    env_data = generator.generate_environment()
    env = DCEnvironment(env_data)
    
    # 创建智能体
    state_size = env.num_servers * 3 + env.num_tasks  # 服务器资源状态 + 任务状态
    action_size = env.num_servers * env.num_tasks     # 每个任务可以放置在每个服务器上
    
    agent = DQNAgent(state_size, action_size)
    
    # 训练智能体
    scores = train_agent(env, agent)
    
    # 保存训练好的模型
    agent.save_model("trained_agent.pth")
    
    # 绘制训练过程
    import matplotlib.pyplot as plt
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

if __name__ == "__main__":
    main() 