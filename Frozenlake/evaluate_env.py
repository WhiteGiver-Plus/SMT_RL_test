import numpy as np
from Q_table import QLearningAgent

def calculate_success_probability(slippery_positions, grid_size=4, holes=[5, 7, 11]):
    """计算给定滑块位置的成功概率
    
    Args:
        slippery_positions: 滑块位置列表
        grid_size: 网格大小
        holes: 洞的位置列表
    
    Returns:
        float: 从起点到终点的成功概率
    """
    total_states = grid_size * grid_size
    
    # 加载Q表并获取最优动作矩阵
    agent = QLearningAgent(total_states, 4)
    agent.load_q_table('q_table_robust.npy')
    action_matrix = agent.get_optimal_action_matrix()
    
    # 初始化状态转移概率
    success_probs = np.zeros(total_states)
    success_probs[total_states - 1] = 1.0  # 终点概率为1
    
    # 迭代计算直到收敛
    max_iterations = 1000
    convergence_threshold = 1e-6
    
    for _ in range(max_iterations):
        old_probs = success_probs.copy()
        
        for s in range(total_states):
            if s == total_states - 1 or s in holes:
                continue
                
            row, col = s // grid_size, s % grid_size
            action = action_matrix[row, col]
            
            # 计算可能的下一个状态
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
            
            # 根据是否是滑块计算转移概率
            if s in slippery_positions:
                # 滑块：每个方向1/3的概率
                success_probs[s] = sum(success_probs[ns] for ns in next_states) / 3
            else:
                # 普通块：确定性移动
                success_probs[s] = success_probs[next_states[1]]
        
        # 检查是否收敛
        if np.max(np.abs(success_probs - old_probs)) < convergence_threshold:
            break
    
    return success_probs[0]  # 返回起点的成功概率

def evaluate_test_cases():
    """评估测试用例文件中的所有环境"""
    test_cases = [
        [1, 2, 3],  # 测试用例1的滑块位置
        [6, 10, 14],  # 测试用例2的滑块位置
        [4, 6, 10],  # 测试用例3的滑块位置
    ]
    
    print("=== 评估结果 ===")
    total_prob = 0
    
    for i, slippery_pos in enumerate(test_cases, 1):
        prob = calculate_success_probability(slippery_pos)
        total_prob += prob
        print(f"\n测试用例 {i}:")
        print(f"滑块位置: {slippery_pos}")
        print(f"计算得到的成功概率: {prob:.4f}")
    
    print(f"\n平均成功概率: {total_prob/len(test_cases):.4f}")

if __name__ == "__main__":
    evaluate_test_cases() 