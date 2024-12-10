import numpy as np
import random

class FrozenLakeQLearning:
    def __init__(self):
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
        
        # 设置目标位置 - 选择步数为19的位置作为目标
        self.goals = [(5, 5)]  # 步数为19的位置
        for x, y in self.goals:
            self.maze[x][y] = 2
            
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
                return (x, y)
            
            x, y = new_x, new_y
            
            # 如果到达目标，停止
            if (x, y) in self.goals:
                return (x, y)
    
    def get_reward(self, pos):
        """获取在当前位置的奖励"""
        if pos in self.goals:
            return 100  # 增加到达目标的奖励
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
    
    def train(self, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
        """训练Q-learning agent"""
        best_path = None
        best_path_length = float('inf')
        
        # 记录训练过程
        training_history = []
        
        # 计算running reward
        running_reward = 0
        running_length = 100  # 计算最近100个episode的平均reward
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
            
            # 每100个episode报告running reward
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}: Running reward: {running_reward:.2f}")
            
            # 记录最短路径
            if done and len(path) < best_path_length:
                best_path = path
                best_path_length = len(path)
            
            # 每1000个episode测试并记录性能
            if (episode + 1) % 1000 == 0:
                avg_reward, avg_steps, success_rate = self.test_performance()
                training_history.append({
                    'episode': episode + 1,
                    'avg_reward': avg_reward,
                    'avg_steps': avg_steps,
                    'success_rate': success_rate,
                    'epsilon': epsilon,
                    'running_reward': running_reward
                })
                print("------------------------")
                print(f"Performance at episode {episode + 1}:")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Average Steps: {avg_steps:.2f}")
                print(f"  Success Rate: {success_rate:.2%}")
                print(f"  Epsilon: {epsilon:.3f}")
                print(f"  Running Reward: {running_reward:.2f}")
                print("------------------------")
            
            # 降低探索率
            if episode % 1000 == 0:
                epsilon = max(0.01, epsilon * 0.995)
        
        return best_path, training_history
    
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
    # 创建并训练agent
    agent = FrozenLakeQLearning()
    print("Training Q-learning agent...")
    best_path, history = agent.train(episodes=1000)
    
    # 显示最优路径
    agent.visualize_path(best_path)
    print(f"\nPath length: {len(best_path)-1} steps")
    print(f"Path: {best_path}")
    
    # 显示最终性能
    print("\nFinal Performance:")
    avg_reward, avg_steps, success_rate = agent.test_performance()
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Success Rate: {success_rate:.2%}")

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

