import numpy as np
from collections import deque
import random

class FrozenMazeGenerator:
    def __init__(self, min_size, max_size, min_wall, max_wall, min_steps):
        """
        初始化迷宫生成器
        min_size: 最小地图边长
        max_size: 最大地图边长
        min_wall: 最小墙体数量
        max_wall: 最大墙体数量
        min_steps: 寻找大于多少步数的路径
        """
        # 随机生成地图大小
        self.width = random.randint(min_size, max_size)
        self.height = random.randint(min_size, max_size)
        
        # 确保墙的数量合理
        self.min_wall = min(min_wall, self.width * self.height - 2)
        self.max_wall = min(max_wall, self.width * self.height - 2)
        
        self.min_steps = min_steps
        self.start = (0, 0)  # 起点固定在左上角
        
    def generate_random_maze(self):
        """随机生成一个满足墙体数量约束的迷宫"""
        maze = np.zeros((self.height, self.width))
        # 随机选择位置放置墙
        available_positions = [(i, j) for i in range(self.height) for j in range(self.width)
                             if (i, j) != self.start]  # 除了起点外的所有位置
        
        num_walls = random.randint(self.min_wall, self.max_wall)
        wall_positions = random.sample(available_positions, num_walls)
        
        for i, j in wall_positions:
            maze[i][j] = 1
            
        return maze
    
    def get_next_positions(self, pos, maze):
        """获取从当前位置滑行可以到达的所有位置"""
        x, y = pos
        next_positions = []
        # 四个方向：上右下左
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        
        # 尝试四个方向的滑行
        for d in range(4):
            curr_x, curr_y = x, y
            while True:
                new_x = curr_x + dx[d]
                new_y = curr_y + dy[d]
                
                # 检查是否出界或撞墙
                if (new_x < 0 or new_x >= self.height or 
                    new_y < 0 or new_y >= self.width or 
                    maze[new_x][new_y] == 1):
                    # 将最后一个有效位置加入结果
                    if (curr_x, curr_y) != pos:
                        next_positions.append((curr_x, curr_y))
                    break
                
                curr_x, curr_y = new_x, new_y
                    
        return next_positions
    
    def calculate_min_steps(self, maze):
        """计算从起点到所���位置的最小步数"""
        # 初始化步数数组为无穷大
        steps = np.full((self.height, self.width), float('inf'))
        steps[self.start] = 0
        
        # 使用BFS计算最短路径
        queue = deque([self.start])
        visited = set([self.start])
        
        while queue:
            curr_pos = queue.popleft()
            
            # 获取所有可以滑到的位置
            next_positions = self.get_next_positions(curr_pos, maze)
            
            for next_pos in next_positions:
                if next_pos not in visited:
                    # 更新步数
                    steps[next_pos] = min(steps[next_pos], steps[curr_pos] + 1)
                    queue.append(next_pos)
                    visited.add(next_pos)
        
        return steps
    
    def visualize(self, maze, steps):
        """可视化迷宫和步数"""
        print(f"\nMaze size: {self.height}x{self.width}")
        print(f"Number of walls: {int(np.sum(maze))}")
        print(f"Looking for paths with steps > {self.min_steps}")
        
        print("\nMaze layout:")
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.start:
                    print('S', end=' ')
                elif maze[i][j] == 1:
                    print('#', end=' ')
                else:
                    print('.', end=' ')
            print()
            
        print("\nSteps to reach each position:")
        for i in range(self.height):
            for j in range(self.width):
                if steps[i][j] == float('inf'):
                    print('  #', end=' ')
                else:
                    print(f'{int(steps[i][j]):3}', end=' ')
            print()
            
        print(f"\nPositions with steps > {self.min_steps}:")
        long_paths = []
        for i in range(self.height):
            for j in range(self.width):
                if steps[i][j] > self.min_steps and steps[i][j] != float('inf'):
                    long_paths.append((i, j, int(steps[i][j])))
        
        if long_paths:
            for i, j, step in sorted(long_paths, key=lambda x: x[2], reverse=True):
                print(f"Position ({i},{j}): {step} steps")
        else:
            print("No positions found!")

def main():
    # 所有参数都可配置
    min_size = 8   # 最小地图大小
    max_size = 8    # 最大地图大小
    min_wall = 5    # 最小墙体数量
    max_wall = 8  # 最大墙体数量
    min_steps = 0  # 寻找大于多少步数的路径
    
    # 这些参数都可以从命令行或配置文件读取
    generator = FrozenMazeGenerator(min_size, max_size, min_wall, max_wall, min_steps)
    
    while True:
        maze = generator.generate_random_maze()
        steps = generator.calculate_min_steps(maze)
        
        # 检查是否有超过指定步数的位置
        has_long_path = False
        for i in range(generator.height):
            for j in range(generator.width):
                if steps[i][j] > generator.min_steps and steps[i][j] != float('inf'):
                    has_long_path = True
                    break
            if has_long_path:
                break
        
        # 如果找到符合条件的迷宫，打印并退出
        if has_long_path:
            generator.visualize(maze, steps)
            break

if __name__ == "__main__":
    main()
