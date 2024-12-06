from z3 import *
import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, grid_size=4):
        """
        初始化Q-learning代理
        
        Args:
            state_size: 状态空间大小
            action_size: 动作空间大小
            grid_size: 网格大小
        """
        # Z3 solver
        self.solver = Solver()
        self.grid_size = grid_size
        self.total_states = grid_size * grid_size
        
        # 初始化Q表
        self.q_table = np.zeros((state_size, action_size))
    
    def load_q_table(self, filename):
        # 从文件加载Q表
        self.q_table = np.load(filename)
    
    def get_action_z3(self, state):
        """获取状态对应的最优动作
        
        Args:
            state: 当前状态
            
        Returns:
            最优动作 (0:左, 1:下, 2:右, 3:上)
        """
        # 直接返回最大Q值对应的动作
        return int(np.argmax(self.q_table[state]))

    def visualize_action(self, state, action, actual_next_state, holes=[5, 7, 11]):
        """可视化4x4网格世界中的状态和动作，显示实际移动结果
        
        Args:
            state: 当前状态
            action: 执行的动作
            actual_next_state: 实际到达的下一个状态
            holes: 洞的位置列表，默认为[5, 7, 11]
        """
        # 定义4x4网格
        grid = ['.'] * 16
        
        # 设置特殊位置
        grid[state] = 'P'  # 当前位置
        grid[15] = 'G'     # 目标位置
        for hole in holes:
            grid[hole] = 'H'  # 洞
        
        # 显示动作箭头
        action_symbols = {
            0: '←',  # LEFT
            1: '',  # DOWN
            2: '→',  # RIGHT
            3: '↑'   # UP
        }
        intended_action = action_symbols[action]
        
        print("\n当前状态:", state)
        print("预期动作:", intended_action)
        print("实际到达:", actual_next_state, "\n")
        
        # 打4x4网格
        for i in range(0, 16, 4):
            print(' '.join(grid[i:i+4]))
        print()

    def simulate_episode(self, start_state=0, max_steps=100, holes=[5, 7, 11]):
        """模拟一个完整的回合，从起点到终点（或失败）
        
        Args:
            start_state: 起始状态
            max_steps: 最大步数，防止无限循环
            holes: 位置列表，默认为[5, 7, 11]
        """
        current_state = start_state
        step = 0
        
        print("\n=== 开始模拟回合 ===")
        print("起始位置:", start_state)
        print("目标位置: 15")
        print("危险位置:", holes)
        
        while step < max_steps:
            # 获取当前状态下的最优动作
            action = self.get_action_z3(current_state)
            
            # 可视化当前状态和动作
            actual_next_state = self.get_next_state(current_state, action)
            self.visualize_action(current_state, action, actual_next_state, holes)
            
            # 检查是否到达终点或掉入洞中
            if actual_next_state == 15:  # 到达目标
                print("🎉 成功到达目标！")
                break
            elif actual_next_state in holes:  # 掉入洞中
                print("💀 掉入洞中，游戏结束！")
                break
                
            current_state = actual_next_state
            step += 1
            
            # 添加暂停，便于观察
            input("按回键继续...")
        
        if step >= max_steps:
            print("达到最大步数限制，模拟结束")

    def get_next_state(self, state, action):
        """据前状态和动作计算下一个状态，考虑冰面滑动的情况
        
        Args:
            state: 当前状 (0-15)
            action: 动作 (0:左, 1:下, 2:右, 3:上) - 使用gym的动作定义
        
        Returns:
            可能的下一个状态，考虑滑动效果
        """
        # 滑动概率：按照原定方向移动的概率是1/3
        # 向左或向右滑动的概率各为1/3
        slide_prob = np.random.random()
        
        # 确定实际移动方向
        if slide_prob < 1/3:
            actual_action = action  # 按原定方向移动
        elif slide_prob < 2/3:
            actual_action = (action + 1) % 4  # 向右滑动（相对于当前朝向）
        else:
            actual_action = (action - 1) % 4  # 向左滑动（相对于当前朝向）
        
        row = state // 4
        col = state % 4
        
        # 使用正确的gym动作定义
        if actual_action == 0:    # 左
            col = max(0, col - 1)
        elif actual_action == 1:  # 下
            row = min(3, row + 1)
        elif actual_action == 2:  # 右
            col = min(3, col + 1)
        elif actual_action == 3:  # 上
            row = max(0, row - 1)
            
        return row * 4 + col

    def get_next_state_probability(self, state, action, target_state, straight_prob=None, right_slide_prob=None, left_slide_prob=None):
        """计算从当前状态执行动作到达目标状态的概率
        
        Args:
            state: 当前状态
            action: 执行的动作 (0:左, 1:下, 2:右, 3:上)
            target_state: 目标状态
            straight_prob, right_slide_prob, left_slide_prob: 移动概率变量，如果未提供则使用1/3
            
        Returns:
            Z3表达式，表示转移概率
        """
        # 如果未提供概率变量，使用默认值1/3
        if straight_prob is None:
            straight_prob = RealVal(1)/3
        if right_slide_prob is None:
            right_slide_prob = RealVal(1)/3
        if left_slide_prob is None:
            left_slide_prob = RealVal(1)/3
        
        row = state // self.grid_size
        col = state % self.grid_size
        
        # 计算三种滑动方向的下一个状态
        next_states = []
        for actual_action in [(action - 1) % 4, action, (action + 1) % 4]:
            new_row, new_col = row, col
            
            # 使用正确的gym动作定义
            if actual_action == 0:    # 左
                new_col = If(col > 0, col - 1, col)
            elif actual_action == 1:  # 下
                new_row = If(row < self.grid_size-1, row + 1, row)
            elif actual_action == 2:  # 右
                new_col = If(col < self.grid_size-1, col + 1, col)
            elif actual_action == 3:  # 上
                new_row = If(row > 0, row - 1, row)
            
            next_state = new_row * self.grid_size + new_col
            next_states.append(next_state)
        
        # 计算到达目标状态的概率
        prob = Sum([
            If(next_states[0] == target_state, left_slide_prob, 0),    # 左滑动概率
            If(next_states[1] == target_state, straight_prob, 0),      # 直行概率
            If(next_states[2] == target_state, right_slide_prob, 0)    # 右滑动概率
        ])
        
        return prob

    def calculate_success_probability(self, start_state=0, holes=None, straight_prob=None, right_slide_prob=None, left_slide_prob=None):
        """计算从起始状态到达终点的概率
        
        Args:
            start_state: 起始状态
            holes: 洞的位置列表
            straight_prob, right_slide_prob, left_slide_prob: 移动概率变量，如果未提供则使用1/3
        """
        # 为每个状态创建到达终点的概率变量
        success_probs = {}
        for state in range(self.total_states):
            success_probs[state] = Real(f'success_prob_{state}')
            
        # 添加终点约束
        self.solver.add(success_probs[self.total_states-1] == 1)  # 终点
        
        # 为每个状态添加概率转移方程
        for state in range(self.total_states):
            if state != self.total_states-1:  # 不是终点
                # 检查是否是洞
                if holes is not None:
                    # 使用传入的洞位置列表，将numpy数组转换为Z3整数常量
                    is_hole = Or([state == IntVal(int(h)) for h in holes])
                else:
                    # 如果没有提供洞的位置，则没有洞
                    is_hole = False
                
                # 直接使用gym的动作映射
                action = self.get_action_z3(state)
                
                # 计算转移概率（使用gym的动作定义）
                next_prob = 0
                for next_state in range(self.total_states):
                    trans_prob = self.get_next_state_probability(state, action, next_state,
                                                              straight_prob, right_slide_prob, left_slide_prob)
                    next_prob = next_prob + If(next_state == self.total_states-1,
                                             trans_prob,
                                             trans_prob * success_probs[next_state])
                
                # 添加约束：如果是洞则概率为0，否则使用计算的转移概率
                self.solver.add(success_probs[state] == If(is_hole, 0, next_prob))
                
                # 添加概率范围约束
                self.solver.add(success_probs[state] >= 0)
                self.solver.add(success_probs[state] <= 1)
        
        return success_probs[start_state]

    def print_optimal_actions(self, holes):
        """打印网格中每个位置的最优动作
        
        Args:
            holes: 洞的位置列表
        """
        action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑', None: 'X'}
        print(f"\n最优动作网格 (↑:上 ↓:下 ←:左 →:右 X:洞/终点):")
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = row * self.grid_size + col
                if state == self.total_states - 1:  # 终点
                    action = None
                elif state in holes:  # 洞
                    action = None
                else:
                    action = self.get_action_z3(state)
                print(f"{action_symbols[action]} ", end="")
            print()  # 换行

    def get_optimal_action_matrix(self):
        """返回网格中每个位置的最优动作矩阵
        
        Returns:
            numpy array: grid_size x grid_size的最优动作矩阵
        """
        action_matrix = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = row * self.grid_size + col
                if state == self.total_states - 1:  # 终点
                    action_matrix[row, col] = -1
                else:
                    action_matrix[row, col] = self.get_action_z3(state)
        return action_matrix

def safe_float_conversion(decimal_str):
    """安全地将Z3的decimal字符串转换为float"""
    try:
        # 移除可能的'?'后缀
        clean_str = decimal_str.split('?')[0]
        return float(clean_str)
    except:
        print("Warning: 转换失败")
        return 0.0  # 转换失败时返回0

# 使用示例
if __name__ == "__main__":
    agent = QLearningAgent(36, 4, grid_size=6)  # 使用6x6网格
    agent.load_q_table('q_table.npy')
    
    # 加载洞的位置
    holes = np.load('q_table_holes.npy')
    
    # 测试场景1：标准设置
    print("\n=== 测试场景1：标准设置 ===")
    agent.solver = Solver()
    
    # 设置具体参数
    # 定义滑动概率为Z3 Real变量
    straight_prob = Real('straight_prob')
    right_slide_prob = Real('right_slide_prob')
    left_slide_prob = Real('left_slide_prob')
    
    # 添加概率约束
    agent.solver.add(straight_prob == RealVal(1)/3)
    agent.solver.add(right_slide_prob == RealVal(1)/3)
    agent.solver.add(left_slide_prob == RealVal(1)/3)
    
    # 计算成功概率
    success_prob = agent.calculate_success_probability(start_state=0, holes=holes)
    
    # 求解并打印结果
    if agent.solver.check() == sat:
        model = agent.solver.model()
        print(f"洞的位置: {holes}")
        print(f"直行概率: {safe_float_conversion(model.eval(straight_prob).as_decimal(20))}")
        print(f"右滑概率: {safe_float_conversion(model.eval(right_slide_prob).as_decimal(20))}")
        print(f"左滑概率: {safe_float_conversion(model.eval(left_slide_prob).as_decimal(20))}")
        print(f"\n从起点(0)到终点({agent.total_states-1})的成功概率: {safe_float_conversion(model.eval(success_prob).as_decimal(20))}")
        
        # 打印最优动作
        agent.print_optimal_actions(holes)
        
        # 模拟一个回合
        # agent.simulate_episode(start_state=0, holes=holes)
    else:
        print("无解")
        print(agent.solver.unsat_core())
