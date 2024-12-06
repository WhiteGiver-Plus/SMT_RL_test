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

def optimize_hole_positions():
    # 初始化变量
    optimizer = Optimize()
    hole1 = Int('hole1')
    hole2 = Int('hole2')
    hole3 = Int('hole3')
    
    # 初始化两个agent并获取它们的动作矩阵
    agent1 = QLearningAgent(16, 4)
    agent2 = QLearningAgent(16, 4)
    agent1.load_q_table('q_table1.npy')
    agent2.load_q_table('q_table2.npy')
    
    action_matrix1 = agent1.get_optimal_action_matrix()
    action_matrix2 = agent2.get_optimal_action_matrix()
    
    # 设置基本约束
    optimizer.add(hole1 > 0, hole1 < 15)
    optimizer.add(hole2 > 0, hole2 < 15)
    optimizer.add(hole3 > 0, hole3 < 15)
    optimizer.add(Distinct([hole1, hole2, hole3]))
    optimizer.add(hole1 >= 4, hole1 <= 11)
    optimizer.add(hole2 >= 4, hole2 <= 11)
    optimizer.add(hole3 >= 4, hole3 <= 11)
    optimizer.add(hole1 < hole2)
    optimizer.add(hole2 < hole3)
    
    # 设置移动概率
    straight_prob = RealVal(1)/3
    right_slide_prob = RealVal(1)/3
    left_slide_prob = RealVal(1)/3
    
    # 为每个状态创建概率变量
    success_probs1 = {}
    success_probs2 = {}
    for s in range(16):
        success_probs1[s] = Real(f'success_prob1_{s}')
        success_probs2[s] = Real(f'success_prob2_{s}')
    
    # 添加终点概率约束
    optimizer.add(success_probs1[15] == 1)
    optimizer.add(success_probs2[15] == 1)
    
    # 为每个状态添加转移概率约束
    for s in range(16):
        if s == 15:  # 终点
            optimizer.add(success_probs1[s] == 1)
            optimizer.add(success_probs2[s] == 1)
        else:  # 非终点状态
            row, col = s // 4, s % 4
            
            # 对agent1的约束
            action1 = action_matrix1[row, col]
            next_states1 = []
            for actual_action in [(action1 - 1) % 4, action1, (action1 + 1) % 4]:
                new_row, new_col = row, col
                if actual_action == 0:    # 左
                    new_col = max(0, col - 1)
                elif actual_action == 1:  # 下
                    new_row = min(3, row + 1)
                elif actual_action == 2:  # 右
                    new_col = min(3, col + 1)
                elif actual_action == 3:  # 上
                    new_row = max(0, row - 1)
                next_states1.append(new_row * 4 + new_col)
            
            # 对agent2的约束
            action2 = action_matrix2[row, col]
            next_states2 = []
            for actual_action in [(action2 - 1) % 4, action2, (action2 + 1) % 4]:
                new_row, new_col = row, col
                if actual_action == 0:    # 左
                    new_col = max(0, col - 1)
                elif actual_action == 1:  # 下
                    new_row = min(3, row + 1)
                elif actual_action == 2:  # 右
                    new_col = min(3, col + 1)
                elif actual_action == 3:  # 上
                    new_row = max(0, row - 1)
                next_states2.append(new_row * 4 + new_col)
            
            # 添加转移概率约束
            is_hole = Or(s == hole1, s == hole2, s == hole3)
            
            # agent1的转移概率约束
            next_prob1 = Sum([
                left_slide_prob * success_probs1[next_states1[0]],
                straight_prob * success_probs1[next_states1[1]],
                right_slide_prob * success_probs1[next_states1[2]]
            ])
            optimizer.add(success_probs1[s] == If(is_hole, 0, next_prob1))
            
            # agent2的转移概率约束
            next_prob2 = Sum([
                left_slide_prob * success_probs2[next_states2[0]],
                straight_prob * success_probs2[next_states2[1]],
                right_slide_prob * success_probs2[next_states2[2]]
            ])
            optimizer.add(success_probs2[s] == If(is_hole, 0, next_prob2))
            
            # 添加概率范围约束
            optimizer.add(success_probs1[s] >= 0)
            optimizer.add(success_probs1[s] <= 1)
            optimizer.add(success_probs2[s] >= 0)
            optimizer.add(success_probs2[s] <= 1)
    
    # 打印约束，帮助调试
    print("\n=== 约束示例 ===")
    print(f"State 14的约束: {optimizer.assertions()[-4]}")
    
    # 最大化差异
    optimizer.maximize((success_probs1[0] - success_probs2[0]))
    
    # 设置求解器超时时间
    optimizer.set("timeout", 30000)
    
    # 求解和输出结果
    if optimizer.check() == sat:
        model = optimizer.model()
        
        hole1_pos = model.eval(hole1).as_long()
        hole2_pos = model.eval(hole2).as_long()
        hole3_pos = model.eval(hole3).as_long()
        
        # 直接打印Z3的结果，不进行转换
        print("\n=== Agent 1 的状态概率 ===")
        for s in range(16):
            print(f"状态 {s}: {model.eval(success_probs1[s])}")
            
        print("\n=== Agent 2 的状态概率 ===")
        for s in range(16):
            print(f"状态 {s}: {model.eval(success_probs2[s])}")
        
        prob1 = model.eval(success_probs1[0])
        prob2 = model.eval(success_probs2[0])
        
        print("\n=== 优化结果 ===")
        print(f"最优洞位置: {hole1_pos}, {hole2_pos}, {hole3_pos}")
        print(f"Agent 1 成功概率: {prob1}")
        print(f"Agent 2 成功概率: {prob2}")
        print(f"最大概率差异: {model.eval((success_probs1[0] - success_probs2[0]))}")
        
        # 可视化结果
        grid = ['.'] * 16
        grid[hole1_pos] = 'H'
        grid[hole2_pos] = 'H'
        grid[hole3_pos] = 'H'
        grid[0] = 'S'
        grid[15] = 'G'
        
        print("\n网格布局:")
        for i in range(0, 16, 4):
            print(' '.join(grid[i:i+4]))
            
        print("\nAgent 1 的最优动作:")
        agent1.print_optimal_actions([hole1_pos, hole2_pos, hole3_pos])
        
        print("\nAgent 2 的最优动作:")
        agent2.print_optimal_actions([hole1_pos, hole2_pos, hole3_pos])
    else:
        print("无解")

if __name__ == "__main__":
    optimize_hole_positions()
