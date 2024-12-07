import gymnasium as gym
import numpy as np
import random

class CustomFrozenLake(gym.Env):
    def __init__(self):
        super().__init__()
        self.base_env = gym.make('FrozenLake-v1', is_slippery=False).unwrapped
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.slippery_tiles = self._select_slippery_tiles()
        
    def _select_slippery_tiles(self):
        # 选择2个非起点非终点的格子作为滑动块
        possible_tiles = [1, 2, 3, 4, 6, 8, 9, 10, 13, 14]  # 排除起点(0)和终点(15)
        return random.sample(possible_tiles, 2)
    
    def step(self, action):
        next_state, reward, done, truncated, info = self.base_env.step(action)
        
        if not done and next_state in self.slippery_tiles:
            # 1/3概率向左滑，1/3概率向右滑
            rand = random.random()
            if rand < 1/3:  # 向左滑
                if next_state % 4 != 0:  # 不在最左边
                    next_state -= 1
            elif rand < 2/3:  # 向右滑
                if (next_state + 1) % 4 != 0:  # 不在最右边
                    next_state += 1
            
            # 检查是否掉入水中
            if self.base_env.desc.flatten()[next_state] == b'H':
                done = True
                reward = 0
            # 检查是否到达目标
            elif self.base_env.desc.flatten()[next_state] == b'G':
                done = True
                reward = 1
                
            self.base_env.s = next_state
        
        return next_state, reward, done, truncated, info
    
    def reset(self, **kwargs):
        self.slippery_tiles = self._select_slippery_tiles()
        return self.base_env.reset(**kwargs)
    
    def render(self, mode='human'):
        return self.base_env.render(mode)

class QTableTrainer:
    def __init__(self, learning_rate=0.1, discount_factor=0.96, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01):
        # 使用自定义环境替换原始环境
        self.env = CustomFrozenLake()
        
        # 训练参数
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q表 (16个状态, 4个动作)
        self.q_table = np.zeros((16, 4))
    
    def get_action(self, state):
        # epsilon-greedy策略选择动作
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])
    
    def train(self, episodes=50000):
        for episode in range(episodes):
            state = self.env.reset()[0]  # 获取初始状态
            done = False
            truncated = False
            
            while not (done or truncated):
                # 选择动作
                action = self.get_action(state)
                
                # 执行动作
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Q-learning更新
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
                self.q_table[state, action] = new_value
                
                state = next_state
            
            # 衰减epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # 每1000轮打印一次进度
            if (episode + 1) % 1000 == 0:
                print(f"Episode {episode + 1}/{episodes}")
    
    def save_q_table(self, filename="q_table_roburst.npy"):
        np.save(filename, self.q_table)
    
    def evaluate(self, num_episodes=1000, render=False):
        # 使用自定义环境进行评估
        test_env = CustomFrozenLake()
        successes = 0
        
        for episode in range(num_episodes):
            state = test_env.reset()[0]
            done = False
            truncated = False
            
            while not (done or truncated):
                action = np.argmax(self.q_table[state])
                state, reward, done, truncated, _ = test_env.step(action)
                if done and reward == 1:  # 成功到达目标
                    successes += 1
                    
        success_rate = successes / num_episodes
        return success_rate

if __name__ == "__main__":
    # 创建训练器并训练
    trainer = QTableTrainer()
    trainer.train(episodes=20000)
    
    # 保存Q表
    trainer.save_q_table()
    print("训练完成，Q表已保存")
    
    # 评估模型性能
    success_rate = trainer.evaluate(num_episodes=1000)
    print(f"测试成功率: {success_rate:.2%}")
    
