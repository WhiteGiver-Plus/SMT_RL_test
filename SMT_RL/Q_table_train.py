import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

class QTableTrainer:
    def __init__(self, grid_size=4, num_holes=3, learning_rate=0.2, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.997, epsilon_min=0.01):
        """
        初始化Q表训练器
        
        Args:
            grid_size: 网格大小
            num_holes: 洞的数量
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
        """
        # 创建冰湖环境
        self.grid_size = grid_size
        self.total_states = grid_size * grid_size
        self.num_holes = num_holes
        
        # 生成随机洞的位置（不包括起点和终点）
        available_positions = list(range(1, self.total_states - 1))
        self.hole_positions = random.sample(available_positions, num_holes)
        
        self.env = gym.make('FrozenLake-v1', 
                           desc=self._generate_map(grid_size),
                           is_slippery=True)
        
        # 训练参数
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 初始化Q表 (states x 4个动作)
        self.q_table = np.zeros((self.total_states, 4))
    
    def _generate_map(self, size):
        """生成指定大小的地图描述，包含指定数量的洞
        
        Args:
            size: 网格大小
            
        Returns:
            list of str: 地图描述
        """
        # 创建一个空地图，所有位置都是F (冰面)
        map_desc = ['F' * size for _ in range(size)]
        
        # 设置起点(S)和终点(G)
        map_desc[0] = 'S' + map_desc[0][1:]
        map_desc[-1] = map_desc[-1][:-1] + 'G'
        
        # 添加洞(H)
        for hole in self.hole_positions:
            row = hole // size
            col = hole % size
            map_desc[row] = map_desc[row][:col] + 'H' + map_desc[row][col+1:]
        
        return map_desc
    
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
    
    def save_q_table(self, filename="q_table.npy"):
        np.save(filename, self.q_table)
        # 同时保存洞的位置，以便后续使用
        np.save(filename.replace('.npy', '_holes.npy'), self.hole_positions)
        print(f"洞的位置: {self.hole_positions}")
    
    def evaluate(self, num_episodes=1000, render=False):
        # 创建测试环境
        test_env = gym.make('FrozenLake-v1', 
                           desc=self._generate_map(self.grid_size),
                           is_slippery=True, 
                           render_mode='human' if render else None)
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
    
    def visualize_policy(self):
        """以字符串形式显示最优策略"""
        actions = np.argmax(self.q_table, axis=1)
        
        # 定义动作映射
        action_symbols = {
            0: '↑',  # 上
            1: '→',  # 右
            2: '↓',  # 下
            3: '←'   # 左
        }
        
        # 创建策略表格
        print("\n最优策略表格:")
        print("-" * (self.grid_size * 4 + 1))
        
        for i in range(self.grid_size):
            row = "|"
            for j in range(self.grid_size):
                state = i * self.grid_size + j
                if state in self.hole_positions:
                    symbol = " H "
                elif i == self.grid_size-1 and j == self.grid_size-1:
                    symbol = " G "
                elif i == 0 and j == 0:
                    symbol = " S "
                else:
                    symbol = f" {action_symbols[actions[state]]} "
                row += symbol + "|"
            print(row)
            print("-" * (self.grid_size * 4 + 1))

if __name__ == "__main__":
    # 创建训练器并训练
    trainer = QTableTrainer(grid_size=6, num_holes=4)  # 7x7网格，4个洞
    trainer.train(episodes=50000)
    
    # 保存Q表
    trainer.save_q_table()
    print("训练完成，Q表已保存")
    
    # 评估模型性能
    success_rate = trainer.evaluate(num_episodes=1000)
    print(f"测试成功率: {success_rate:.2%}")
    
    # 在评估之后添加可视化
    trainer.visualize_policy()
    
