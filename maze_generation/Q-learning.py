import numpy as np
import random
import matplotlib.pyplot as plt

class FrozenLakeQLearning:
    def __init__(self, difficulty='easy'):
        # 迷宫布局 (0:空地, 1:墙, 2:目标)
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
        
        # 保存难度设置
        self.difficulty = difficulty
        
        # 根据难度设置目标位置
        if difficulty == 'easy':
            self.goals = [(6, 2)]  # 简单例子终点
            self.target_reward = 50
        else:
            self.goals = [(6, 5)]  # 难例子终点
            self.target_reward = 100
        
        self.start = (0, 0)
        self.height, self.width = self.maze.shape
        self.n_states = self.height * self.width
        self.n_actions = 4  # 上右下左
        
        # Q表格
        self.q_table = np.zeros((self.n_states, self.n_actions))
        
        # 动作对应的方向
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
        
    def get_state_index(self, pos):
        """将位置转换为状态索引"""
        return pos[0] * self.width + pos[1]
    
    def get_next_position(self, pos, action):
        """计算在给定位置采取动作后的下一个位置（滑行）"""
        dx, dy = self.actions[action]
        x, y = pos
        
        # 滑行直到撞墙或到达边界
        while True:
            new_x = x + dx
            new_y = y + dy
            
            # 检查是否出界或撞墙
            if (new_x < 0 or new_x >= self.height or 
                new_y < 0 or new_y >= self.width or 
                self.maze[new_x][new_y] == 1):
                # 如果在撞墙时的位置是目标,则算作到达目标
                if (x, y) in self.goals:
                    return (x, y)
                return (x, y)
            
            x, y = new_x, new_y
    
    def get_reward(self, pos):
        """获取在当前位置的奖励"""
        if pos in self.goals:
            return self.target_reward
        return -1     # 移动惩罚
    
    def test_performance(self, num_tests=100):
        """测试当前策略的性能"""
        total_reward = 0
        total_steps = 0
        success_count = 0
        
        for _ in range(num_tests):
            state = self.start
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 100:
                state_idx = self.get_state_index(state)
                action = np.argmax(self.q_table[state_idx])
                next_state = self.get_next_position(state, action)
                reward = self.get_reward(next_state)
                
                episode_reward += reward
                state = next_state
                steps += 1
                done = state in self.goals
            
            if done:
                success_count += 1
            total_reward += episode_reward
            total_steps += steps
        
        avg_reward = total_reward / num_tests
        avg_steps = total_steps / num_tests
        success_rate = success_count / num_tests
        
        return avg_reward, avg_steps, success_rate
    
    def train(self, episodes=200, alpha=0.1, gamma=0.99, epsilon=0.1):
        """训练Q-learning agent"""
        best_path = None
        best_path_length = float('inf')
        best_path_lengths = []  # 记录每10步的最优路径长度
        
        # 记录训练过程
        training_history = []
        
        # 计算running reward
        running_reward = 0
        running_length = 10  # 计算最近10个episode的平均reward
        episode_rewards = []
        
        for episode in range(episodes):
            state = self.start
            path = [state]
            done = False
            episode_reward = 0
            
            while not done and len(path) < 100:
                if random.random() < epsilon:
                    action = random.randint(0, self.n_actions - 1)
                else:
                    state_idx = self.get_state_index(state)
                    action = np.argmax(self.q_table[state_idx])
                
                next_state = self.get_next_position(state, action)
                reward = self.get_reward(next_state)
                episode_reward += reward
                
                state_idx = self.get_state_index(state)
                next_state_idx = self.get_state_index(next_state)
                
                old_q = self.q_table[state_idx][action]
                next_max = np.max(self.q_table[next_state_idx])
                
                new_q = old_q + alpha * (reward + gamma * next_max - old_q)
                self.q_table[state_idx][action] = new_q
                
                state = next_state
                path.append(state)
                done = state in self.goals
            
            # 更新running reward
            episode_rewards.append(episode_reward)
            if len(episode_rewards) > running_length:
                episode_rewards.pop(0)
            running_reward = np.mean(episode_rewards)
            
            # 记录最短路径
            if done and len(path) < best_path_length:
                best_path = path
                best_path_length = len(path)
            
            # 每10个episode记录并输出
            if (episode + 1) % 10 == 0:
                best_path_lengths.append(best_path_length)
                print(f"Episode {episode + 1}: Best path length: {best_path_length}")
                print(f"Running reward: {running_reward:.2f}")
                print(f"Epsilon: {epsilon:.3f}")
                
                # 降低探索率
                epsilon = max(0.01, epsilon * 0.995)
        
        # 绘制最优路径长度变化图
        plt.figure()
        plt.plot(range(10, episodes + 1, 10), best_path_lengths)
        plt.xlabel('Episodes')
        plt.ylabel('Best Path Length')
        plt.title(f'Q-Learning Best Path Length vs Episodes ({self.difficulty})')
        plt.savefig(f'q_learning_{self.difficulty}_path_lengths.png')
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
        plt.title(f'Q-Learning Optimal Path ({self.difficulty})', 
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
        plt.savefig(f'q_learning_{self.difficulty}_optimal_path.png', 
                    bbox_inches='tight', 
                    dpi=300,
                    pad_inches=0.2)
        plt.close()

def main():
    # 训练简单例子
    print("\nTraining on Easy Map...")
    agent_easy = FrozenLakeQLearning(difficulty='easy')
    best_path_easy, history_easy = agent_easy.train(episodes=200)  # 改为200轮
    agent_easy.visualize_path(best_path_easy)
    
    # 训练难例子
    print("\nTraining on Hard Map...")
    agent_hard = FrozenLakeQLearning(difficulty='hard')
    best_path_hard, history_hard = agent_hard.train(episodes=200)  # 改为200轮
    agent_hard.visualize_path(best_path_hard)

if __name__ == "__main__":
    main()

# S # . . # . . .
# . . . . . . . #
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# # . . . # . # .
# . . . # . G . .
# . . . . . # . .

