from z3 import *
import random

class FrozenMazeGeneratorZ3:
    def __init__(self, width, height, min_wall, max_wall, min_steps):
        """
        初始化迷宫生成器
        min_wall: 最小墙体数量
        max_wall: 最大墙体数量
        min_steps: 寻找大于多少步数的路径
        """
        self.width = width   # 固定宽度
        self.height = height # 固定高度
        self.min_wall = min_wall
        self.max_wall = max_wall
        self.min_steps = min_steps
        self.start = (0, 0)

    def safe_maze_access(self, maze, i, j):
        """安全地访问迷宫位置，超出边界返回True（视为墙）"""
        return If(
            And(
                i >= 0,
                i < self.height,
                j >= 0,
                j < self.width
            ),
            maze[i][j],
            BoolVal(True)  # 超出边界时视为墙
        )

    def get_valid_moves(self, maze, i, j):
        """
        获取从位置(i,j)开始的所有有效移动
        只返回会撞墙或到达边界的移动
        
        Args:
            maze: 迷宫的Z3变量数组
            i, j: 当前位置的坐标
        
        Returns:
            list: 包含所有有效移动的列表，每个移动是一个元组 (move_condition, ni, nj, k, i, j)
        """
        valid_moves = []
        # 四个方向：上右下左
        for dx, dy, direction in [(-1, 0, "up"), (0, 1, "right"), (1, 0, "down"), (0, -1, "left")]:
            for k in range(1, max(self.height, self.width)):
                ni, nj = i + k * dx, j + k * dy
                
                # 首先检查当前位置是否在边界内
                if not (0 <= ni < self.height and 0 <= nj < self.width):
                    break
                
                next_i, next_j = ni + dx, nj + dy
                
                # 检查是否到达边界或撞墙
                if not (0 <= next_i < self.height and 0 <= next_j < self.width):
                    # 到达边界
                    path_clear = And([Not(self.safe_maze_access(maze, i + l * dx, j + l * dy)) 
                                       for l in range(1,k+1)])
                    move_condition = And(path_clear, Not(maze[i][j]), Not(maze[ni][nj]))
                    valid_moves.append((move_condition, ni, nj, k, i, j))
                    break
                
                # 检查是否撞墙
                path_clear = And([Not(self.safe_maze_access(maze, i + l * dx, j + l * dy)) 
                                   for l in range(1,k+1)])
                move_condition = And(
                    path_clear,
                    maze[next_i][next_j],  # 下一格是墙
                    Not(maze[i][j]),
                    Not(maze[ni][nj])
                )
                valid_moves.append((move_condition, ni, nj, k, i, j))  # 找到停止点后不再继续搜索该方向
                
        return valid_moves

    def generate_maze(self):
        """使用Z3生成满足条件的迷宫"""
        # 确保墙的数量合理
        self.min_wall = min(self.min_wall, self.width * self.height - 2)
        self.max_wall = min(self.max_wall, self.width * self.height - 2)
        num_walls = random.randint(self.min_wall, self.max_wall)

        # 定义Z3变量
        maze = [[Bool(f"maze_{i}_{j}") for j in range(self.width)] for i in range(self.height)]
        steps = [[Int(f"steps_{i}_{j}") for j in range(self.width)] for i in range(self.height)]

        s = Solver()

        # 约束：起点不是墙
        s.assert_and_track(Not(maze[self.start[0]][self.start[1]]), "start_not_wall")

        # 约束：墙的数量
        wall_count = Sum([If(maze[i][j], 1, 0) for i in range(self.height) for j in range(self.width)])
        s.assert_and_track(wall_count == num_walls, "wall_count")

        # 约束：步数计算
        # 起点步数为0
        s.assert_and_track(steps[self.start[0]][self.start[1]] == 0, "start_steps")

        # 约束：所有非墙位置的步数必须有效
        for i in range(self.height):
            for j in range(self.width):
                s.assert_and_track(
                    Implies(
                        Not(maze[i][j]),
                        And(
                            steps[i][j] >= 0,
                            steps[i][j] <= self.width * self.height
                        )
                    ),
                    f"valid_steps_{i}_{j}"
                )

        # 约束：滑行规则
        valid_moves = []
        for i in range(self.height):
            for j in range(self.width):
                valid_moves.extend(self.get_valid_moves(maze, i, j))
        for move_condition, ni, nj, k, prev_i, prev_j in valid_moves:
            # 1. 步数递增约束
            s.assert_and_track(
                Implies(
                    And(Not(maze[prev_i][prev_j]), move_condition),
                        steps[ni][nj] <= steps[prev_i][prev_j] + 1
                    ),
                    f"slide_rule_{prev_i}_{prev_j}_{k}_{ni}_{nj}"
                )

            # 2. 新增约束：每个非起点的可达位置必须由前一步通过有效移动到达
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) != self.start:
                    possible_origins = []
                    for move_condition, ni, nj, k, prev_i, prev_j in valid_moves:
                        if ni == i and nj == j:
                            possible_origins.append(And(Not(maze[prev_i][prev_j]), move_condition, steps[prev_i][prev_j] == steps[i][j] - 1))

                    s.assert_and_track(
                        Implies(
                            And(Not(maze[i][j]), steps[i][j] < self.width * self.height),
                            Or(*possible_origins)
                        ),
                        f"reachability_{i}_{j}"
                    )

        # 约束：墙的位置的步数必须是最大值(64)
        for i in range(self.height):
            for j in range(self.width):
                s.assert_and_track(
                    Implies(
                        maze[i][j],  # 如果是墙
                        steps[i][j] == self.width * self.height  # 则步数为最大值
                    ),
                    f"wall_steps_{i}_{j}"
                )

        # 约束：存在大于min_steps的路径
        s.assert_and_track(
            Or([And(steps[i][j] > self.min_steps, 
                   steps[i][j] < self.width * self.height-1) 
               for i in range(self.height) 
               for j in range(self.width)]),
            "min_steps_constraint"
        )

        # 求解
        if s.check() == sat:
            m = s.model()
            maze_result = [[is_true(m[maze[i][j]]) for j in range(self.width)] 
                          for i in range(self.height)]
            steps_result = [[
                m[steps[i][j]].as_long() if m[steps[i][j]] is not None and not maze_result[i][j]
                else float('inf')
                for j in range(self.width)
            ] for i in range(self.height)]
            return maze_result, steps_result
        else:
            print("Unsatisfiable constraints:", s.unsat_core())
            print("No solution found.")
            return None, None

    def visualize(self, maze, steps):
        """可视化迷宫和步数"""
        print(f"\nMaze size: {self.height}x{self.width}")
        print(f"Number of walls: {sum(row.count(True) for row in maze)}")
        print(f"Looking for paths with steps > {self.min_steps}")

        print("\nMaze layout:")
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.start:
                    print('S', end=' ')
                elif maze[i][j]:
                    print('#', end=' ')
                else:
                    print('.', end=' ')
            print()

        print("\nSteps to reach each position:")
        for i in range(self.height):
            for j in range(self.width):
                if maze[i][j]:
                    print('  #', end=' ')
                elif steps[i][j] == self.width * self.height:
                    print('  X', end=' ')
                else:
                    print(f'{steps[i][j]:3}', end=' ')
            print()

        print(f"\nPositions with steps > {self.min_steps}:")
        long_paths = []
        for i in range(self.height):
            for j in range(self.width):
                if steps[i][j] > self.min_steps and steps[i][j] != self.width * self.height:
                    long_paths.append((i, j, steps[i][j]))

        if long_paths:
            for i, j, step in sorted(long_paths, key=lambda x: x[2], reverse=True):
                print(f"Position ({i},{j}): {step} steps")
        else:
            print("No positions found!")

def main():
    # 固定地图大小为8x8
    width = 8
    height = 8
    min_wall = 5
    max_wall = 8
    min_steps = -1

    generator = FrozenMazeGeneratorZ3(width, height, min_wall, max_wall, min_steps)
    maze, steps = generator.generate_maze()

    if maze and steps:
        generator.visualize(maze, steps)

if __name__ == "__main__":
    main()
