from z3 import *
import numpy as np
from Q_table import QLearningAgent, safe_float_conversion

def is_concrete_value(expr):
    """检查Z3表达式是否为具体值"""
    try:
        return expr.as_long() is not None
    except:
        print("1")
        return False

def calculate_success_probability(action_matrix, state, holes, straight_prob, right_slide_prob, left_slide_prob):
    """计算从给定状态到达终点的概率的Z3约束
    
    Args:
        action_matrix: 4x4的最优动作矩阵
        state: 当前状态
        holes: 洞的位置列表 (Z3 Int变量)
        straight_prob, right_slide_prob, left_slide_prob: 移动概率 (Z3 Real变量)
    
    Returns:
        (success_prob, constraints): 成功概率变量和相关约束的列表
    """
    success_probs = {}
    for s in range(16):
        success_probs[s] = Real(f'success_prob_{s}')
    
    # 终点概率为1
    constraints = [success_probs[15] == 1]
    
    # 为每个状态添加概率转移方程
    for s in range(16):
        if s != 15:  # 不是终点
            row, col = s // 4, s % 4
            action = action_matrix[row, col]
            
            # 计算三个可能的下一个状态
            next_states = []
            for actual_action in [(action - 1) % 4, action, (action + 1) % 4]:
                new_row, new_col = row, col
                
                # 使用Z3的If表达式计算下一个状态
                if actual_action == 0:    # 左
                    new_col = If(col > 0, col - 1, col)
                elif actual_action == 1:  # 下
                    new_row = If(row < 3, row + 1, row)
                elif actual_action == 2:  # 右
                    new_col = If(col < 3, col + 1, col)
                elif actual_action == 3:  # 上
                    new_row = If(row > 0, row - 1, row)
                
                next_state = new_row * 4 + new_col
                next_states.append(next_state)
            
            # 计算转移概率
            next_prob = Sum([
                If(next_states[0] == 15, left_slide_prob, 
                   left_slide_prob * success_probs[ToInt(next_states[0])]),
                If(next_states[1] == 15, straight_prob, 
                   straight_prob * success_probs[ToInt(next_states[1])]),
                If(next_states[2] == 15, right_slide_prob, 
                   right_slide_prob * success_probs[ToInt(next_states[2])])
            ])
            
            # 如果是洞，概率为0
            is_hole = Or([s == h for h in holes])
            constraints.append(success_probs[s] == If(is_hole, 0, next_prob))
            constraints.append(success_probs[s] >= 0)
            constraints.append(success_probs[s] <= 1)
    
    return success_probs[state], constraints

def optimize_grid_parameters(num_mutants=50, mutants_dir='mutants', grid_size=4, 
                           optimize_holes=True, optimize_probs=False, 
                           fixed_holes=None, fixed_probs=None):
    """通用优化函数，可以优化洞的位置和/或滑动概率
    
    Args:
        num_mutants: 变异体数量
        mutants_dir: 存放变异体Q表的目录
        grid_size: 网格大小
        num_holes: 洞的数量
    """
    optimizer = Optimize()
    total_states = grid_size * grid_size
    # 初始化洞的位置变量
    holes = [Int(f'hole{i}') for i in range(3)]  # 固定为3个洞
    
    # 设置基本约束
    for i in range(3):  # 固定为3个洞
        optimizer.add(holes[i] > 0, holes[i] < total_states-1)  # 不能在起点和终点
        optimizer.add(holes[i] >= grid_size, holes[i] <= total_states-grid_size-1)  # 在中间区域
    
    # 确保洞的位置不重复且有序
    optimizer.add(Distinct(holes))
    for i in range(2):  # 固定为3个洞，所以是range(2)
        optimizer.add(holes[i] < holes[i+1])
    
    # 设置移动概率变量
    straight_prob = Real('straight_prob')
    right_slide_prob = Real('right_slide_prob')
    left_slide_prob = Real('left_slide_prob')
    
    # 添加概率约束
    optimizer.add(straight_prob == RealVal(1)/3)
    optimizer.add(right_slide_prob == RealVal(1)/3)
    optimizer.add(left_slide_prob == RealVal(1)/3)
    
    # 加载原始模型和变异体
    original_agent = QLearningAgent(total_states, 4)
    original_agent.load_q_table('q_table.npy')
    
    mutant_agents = []
    for i in range(num_mutants):
        agent = QLearningAgent(total_states, 4)
        agent.load_q_table(f'{mutants_dir}/q_table_{i}.npy')
        mutant_agents.append(agent)
    
    # 为原始模型和所有变异体创建概率变量
    success_probs = {}
    
    # 为原始模型创建概率变量
    for s in range(total_states):
        success_probs['original_state_{}'.format(s)] = Real(f'success_prob_original_{s}')
        if s == total_states-1:  # 终点概率为1
            optimizer.add(success_probs['original_state_{}'.format(s)] == 1)
    
    # 为每个变异体创建概率变量
    for i in range(num_mutants):
        for s in range(total_states):
            success_probs[f'mutant_{i}_state_{s}'] = Real(f'success_prob_mutant_{i}_{s}')
            if s == total_states-1:  # 终点概率为1
                optimizer.add(success_probs[f'mutant_{i}_state_{s}'] == 1)
    
    # 为原始模型添加转移概率约束
    action_matrix_original = original_agent.get_optimal_action_matrix()
    for s in range(total_states):
        if s != total_states-1:  # 非终点状态
            row, col = s // grid_size, s % grid_size
            action = action_matrix_original[row, col]
            
            # 计算三个可能的下一个状态
            next_states = []
            for actual_action in [(action - 1) % 4, action, (action + 1) % 4]:
                new_row, new_col = row, col
                if actual_action == 0:    # 左
                    new_col = max(0, col - 1)
                elif actual_action == 1:  # 下
                    new_row = min(grid_size-1, row + 1)
                elif actual_action == 2:  # 右
                    new_col = min(grid_size-1, col + 1)
                elif actual_action == 3:  # 上
                    new_row = max(0, row - 1)
                next_states.append(new_row * grid_size + new_col)
            
            # 添加转移概率约束
            is_hole = Or(s == holes[0], s == holes[1], s == holes[2])
            
            next_prob = Sum([
                left_slide_prob * success_probs[f'original_state_{next_states[0]}'],
                straight_prob * success_probs[f'original_state_{next_states[1]}'],
                right_slide_prob * success_probs[f'original_state_{next_states[2]}']
            ])
            
            optimizer.add(success_probs[f'original_state_{s}'] == If(is_hole, 0, next_prob))
            optimizer.add(success_probs[f'original_state_{s}'] >= 0)
            optimizer.add(success_probs[f'original_state_{s}'] <= 1)
    
    # 为每个变异体添加转移概率约束
    for i, agent in enumerate(mutant_agents):
        action_matrix = agent.get_optimal_action_matrix()
        
        for s in range(total_states):
            if s != total_states-1:  # 非终点状态
                row, col = s // grid_size, s % grid_size
                action = action_matrix[row, col]
                
                # 计算三个可能的下一个状态
                next_states = []
                for actual_action in [(action - 1) % 4, action, (action + 1) % 4]:
                    new_row, new_col = row, col
                    if actual_action == 0:    # 左
                        new_col = max(0, col - 1)
                    elif actual_action == 1:  # 下
                        new_row = min(grid_size-1, row + 1)
                    elif actual_action == 2:  # 右
                        new_col = min(grid_size-1, col + 1)
                    elif actual_action == 3:  # 上
                        new_row = max(0, row - 1)
                    next_states.append(new_row * grid_size + new_col)
                
                # 添加转移概率约束
                is_hole = Or(s == holes[0], s == holes[1], s == holes[2])
                
                next_prob = Sum([
                    left_slide_prob * success_probs[f'mutant_{i}_state_{next_states[0]}'],
                    straight_prob * success_probs[f'mutant_{i}_state_{next_states[1]}'],
                    right_slide_prob * success_probs[f'mutant_{i}_state_{next_states[2]}']
                ])
                
                optimizer.add(success_probs[f'mutant_{i}_state_{s}'] == If(is_hole, 0, next_prob))
                optimizer.add(success_probs[f'mutant_{i}_state_{s}'] >= 0)
                optimizer.add(success_probs[f'mutant_{i}_state_{s}'] <= 1)
    
    # 计算变异体的平均成功概率
    mutants_avg_prob = Sum([success_probs[f'mutant_{i}_state_0'] for i in range(num_mutants)]) / num_mutants
    
    # 最大化原始模型与变异体平均性能的差异
    optimizer.maximize(success_probs['original_state_0'] - mutants_avg_prob)
    
    # 设置求解器超时时间
    optimizer.set("timeout", 60000)  # 60秒超时
    
    # 求解和输出结果
    if optimizer.check() == sat:
        model = optimizer.model()
        
        # 获取洞的位置
        hole_positions = [model.eval(h).as_long() for h in holes]
        
        print("\n=== 优化结果 ===")
        print(f"最优洞位置: {hole_positions}")
        
        # 计算并打印原始模型和变异体的成功概率
        original_prob = model.eval(success_probs['original_state_0'])
        print(f"\n原始模型成功概率: {original_prob}")
        
        print("\n=== 变异体成功概率 ===")
        total_mutant_prob = RealVal(0)  # 使用Z3的RealVal初始化
        for i in range(num_mutants):
            prob = model.eval(success_probs[f'mutant_{i}_state_0'])
            print(f"变异体 {i} 成功概率: {prob}")
            total_mutant_prob += prob  # 直接使用Z3的数值相加
        
        avg_mutant_prob = total_mutant_prob / num_mutants
        print(f"\n变异体平均成功概率: {avg_mutant_prob}")
        print(f"性能差异: {original_prob - avg_mutant_prob}")
        
        # 可视化结果
        grid = ['.'] * total_states
        for pos in hole_positions:
            grid[pos] = 'H'
        grid[0] = 'S'
        grid[total_states-1] = 'G'
        
        print("\n网格布局:")
        for i in range(0, total_states, grid_size):
            print(' '.join(grid[i:i+grid_size]))
            
        # 打印原始模型的最优动作
        print("\n原始模型的最优动作:")
        original_agent.print_optimal_actions(hole_positions)
        
        # 打印第一个变异体的最优动作作为示例
        print("\n变异体示例(第一个)的最优动作:")
        mutant_agents[0].print_optimal_actions(hole_positions)
    else:
        print("无解")
    
    return optimizer, holes, (straight_prob, right_slide_prob, left_slide_prob)

def optimize_hole_positions(num_mutants=50, mutants_dir='mutants', grid_size=4):
    """优化洞的位置（固定概率）"""
    fixed_probs = (1/3, 1/3, 1/3)  # 固定的移动概率
    return optimize_grid_parameters(
        num_mutants=num_mutants,
        mutants_dir=mutants_dir,
        grid_size=grid_size,
        optimize_holes=True,
        optimize_probs=False,
        fixed_probs=fixed_probs
    )

def optimize_slide_probabilities(num_mutants=50, mutants_dir='mutants', grid_size=4, fixed_holes=None):
    """优化滑动概率（固定洞的位置）"""
    if fixed_holes is None:
        raise ValueError("必须提供固定的洞位置")
    
    return optimize_grid_parameters(
        num_mutants=num_mutants,
        mutants_dir=mutants_dir,
        grid_size=grid_size,
        optimize_holes=False,
        optimize_probs=True,
        fixed_holes=fixed_holes
    )

if __name__ == "__main__":
    # 示例：优化洞的位置
    optimize_hole_positions(grid_size=4)
    
    # 示例：优化滑动概率
    # fixed_holes = [5, 6, 9]  # 这些值需要根据实际情况设置
    # optimize_slide_probabilities(grid_size=4, fixed_holes=fixed_holes)
