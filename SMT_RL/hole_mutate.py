from z3 import *
import numpy as np
from Q_table import QLearningAgent, safe_float_conversion

def optimize_hole_positions():
    # 初始化 z3 求解器和变量
    optimizer = Optimize()
    hole1 = Int('hole1')
    hole2 = Int('hole2')
    hole3 = Int('hole3')
    
    # 初始化两个 Q-learning agent
    agent1 = QLearningAgent(16, 4)
    agent2 = QLearningAgent(16, 4)
    
    # 加载两个不同的 Q 表
    agent1.load_q_table('q_table1.npy')
    agent2.load_q_table('q_table2.npy')
    
    # 使用 z3 变量设置基本约束
    optimizer.add(hole1 > 0, hole1 < 15)
    optimizer.add(hole2 > 0, hole2 < 15)
    optimizer.add(hole3 > 0, hole3 < 15)
    optimizer.add(Distinct([hole1, hole2, hole3]))
    
    # 减少搜索范围
    optimizer.add(hole1 >= 4, hole1 <= 11)
    optimizer.add(hole2 >= 4, hole2 <= 11)
    optimizer.add(hole3 >= 4, hole3 <= 11)
    optimizer.add(hole1 < hole2)
    optimizer.add(hole2 < hole3)
    
    # 使用 z3 变量计算成功概率
    success_prob1 = agent1.calculate_success_probability(start_state=0, holes=[hole1, hole2, hole3])
    success_prob2 = agent2.calculate_success_probability(start_state=0, holes=[hole1, hole2, hole3])
    
    # 最大化差异
    optimizer.maximize(Abs(success_prob1 - success_prob2))
    
    # 设置求解器超时时间（毫秒）
    optimizer.set("timeout", 30000)  # 30秒超时
    
    # 求解和结果输出
    if optimizer.check() == sat:
        model = optimizer.model()
        
        # 获取最优的洞位置
        hole1_pos = model.eval(hole1).as_long()
        hole2_pos = model.eval(hole2).as_long()
        hole3_pos = model.eval(hole3).as_long()
        
        # 获取两个agent的最优成功概率
        prob1_val = model.eval(success_prob1)
        prob2_val = model.eval(success_prob2)
        prob1 = safe_float_conversion(str(prob1_val))
        prob2 = safe_float_conversion(str(prob2_val))
        
        print("\n=== 优化结果 ===")
        print(f"最优洞位置: {hole1_pos}, {hole2_pos}, {hole3_pos}")
        print(f"Agent 1 成功概率: {prob1:.4f}")
        print(f"Agent 2 成功概率: {prob2:.4f}")
        print(f"最大概率差异: {abs(prob1 - prob2):.4f}")
        
        # 可视化网格
        grid = ['.'] * 16
        grid[hole1_pos] = 'H'
        grid[hole2_pos] = 'H'
        grid[hole3_pos] = 'H'
        grid[0] = 'S'
        grid[15] = 'G'
        
        print("\n网格布局:")
        for i in range(0, 16, 4):
            print(' '.join(grid[i:i+4]))
            
        # 添加最优动作可视化
        print("\nAgent 1 的最优动作:")
        agent1.print_optimal_actions()
        
        print("\nAgent 2 的最优动作:")
        agent2.print_optimal_actions()
        
        # 可选：模拟一个回合
        print("\n模拟 Agent 1 的一个回合:")
        agent1.simulate_episode()
        
        print("\n模拟 Agent 2 的一个回合:")
        agent2.simulate_episode()

if __name__ == "__main__":
    optimize_hole_positions()
