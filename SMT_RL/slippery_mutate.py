from z3 import *
import numpy as np
from Q_table import QLearningAgent

def safe_prob_conversion(prob_str):
    """安全地将Z3输出的概率字符串转换为浮点数
    
    Args:
        prob_str: Z3输出的概率字符串，可能是分数(如 "1/3")、小数或带问号的值
        
    Returns:
        float: 转换后的浮点数
    """
    # 移除可能的问号
    prob_str = str(prob_str).replace('?', '')
    
    try:
        # 尝试直接转换为浮点数
        return float(prob_str)
    except ValueError:
        try:
            # 尝试处理分数形式
            if '/' in prob_str:
                num, denom = prob_str.split('/')
                return float(num) / float(denom)
            else:
                return 0.0  # 无法转换时返回0
        except:
            return 0.0  # 出现任何错误时返回0

def optimize_multiple_slippery_positions(num_cases=3, num_slippery=3, grid_size=4, diversity_weight=0.1):
    """优化多个测试用例的滑块位置
    
    Args:
        num_cases: 要生成的测试用例数量
        num_slippery: 每个测试用例中滑块的数量
        grid_size: 网格大小
        diversity_weight: 多样性奖励的权重
    """
    optimizer = Optimize()
    total_states = grid_size * grid_size
    holes = [5, 7, 11]  # 固定洞的位置
    
    # 为每个测试用例初始化滑块的位置变量
    all_slippery_positions = []
    for case in range(num_cases):
        slippery_positions = [Int(f'slippery_{case}_{i}') for i in range(num_slippery)]
        all_slippery_positions.append(slippery_positions)
        
        # 设置基本约束
        for i in range(num_slippery):
            optimizer.add(slippery_positions[i] >= 0, slippery_positions[i] < total_states)
            optimizer.add(slippery_positions[i] != 0)  # 不能在起点
            optimizer.add(slippery_positions[i] != total_states-1)  # 不能在终点
            # 不能在洞的位置
            for hole in holes:
                optimizer.add(slippery_positions[i] != hole)
        
        # 确保滑块的位置不重复且有序
        optimizer.add(Distinct(slippery_positions))
        for i in range(num_slippery-1):
            optimizer.add(slippery_positions[i] < slippery_positions[i+1])
    
    # 确保不同测试用例之间的滑块位置组合不同
    for i in range(num_cases):
        for j in range(i + 1, num_cases):
            # 至少有一个位置不同
            different_positions = []
            for k in range(num_slippery):
                different_positions.append(all_slippery_positions[i][k] != all_slippery_positions[j][k])
            optimizer.add(Or(different_positions))
    
    # 加载原始模型
    agent = QLearningAgent(total_states, 4)
    agent.load_q_table('SMT_RL/q_table_dqn_robust.npy')
    action_matrix = agent.get_optimal_action_matrix()
    
    # 为每个测试用例创建成功概率变量
    all_success_probs = []
    for case in range(num_cases):
        success_probs = {}
        for s in range(total_states):
            success_probs[s] = Real(f'success_prob_{case}_{s}')
            if s == total_states-1:  # 终点概率为1
                optimizer.add(success_probs[s] == 1)
            elif s in holes:  # 洞的概率为0
                optimizer.add(success_probs[s] == 0)
        all_success_probs.append(success_probs)
        
        # 为每个状态加转移概率约束
        for s in range(total_states):
            if s != total_states-1 and s not in holes:  # 非终点且非洞状态
                row, col = s // grid_size, s % grid_size
                action = action_matrix[row, col]
                
                # 计算可��的下一个状态
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
                
                # 判断当前位置是否是滑块
                is_slippery = Or([s == pos for pos in all_slippery_positions[case]])
                
                # 如果是滑块，使用滑动概率；如果不是，直接使用预期的下一个状态
                next_prob = If(is_slippery,
                             # 滑块：每个方1/3的概率
                             Sum([success_probs[next_states[0]] / 3,
                                 success_probs[next_states[1]] / 3,
                                 success_probs[next_states[2]] / 3]),
                             # 普通块：100%按预期移动
                             success_probs[next_states[1]])
                
                optimizer.add(success_probs[s] == next_prob)
                optimizer.add(success_probs[s] >= 0)
                optimizer.add(success_probs[s] <= 1)
    
    # 创建多样性度量变量
    diversity_vars = []
    for i in range(num_cases):
        for j in range(i + 1, num_cases):
            # 计算每对测试用例之间的不同位置数量
            diff_count = Real(f'diff_count_{i}_{j}')
            different_positions = []
            for pos1 in all_slippery_positions[i]:
                for pos2 in all_slippery_positions[j]:
                    different_positions.append(If(pos1 == pos2, 0, 1))
            optimizer.add(diff_count == Sum(different_positions))
            diversity_vars.append(diff_count)
    
    # 计算总的多样性分数
    total_diversity = Sum(diversity_vars)
    
    # 计算平均成功概率
    avg_success_prob = Sum([all_success_probs[case][0] for case in range(num_cases)]) / num_cases
    
    # 最小化目标：平均成功概率 - 多样性奖励
    optimizer.minimize(avg_success_prob - diversity_weight * total_diversity / (num_cases * (num_cases - 1) / 2))
    
    # 设置求解器超时时间
    optimizer.set("timeout", 120000)  # 120秒超时
    
    # 求解和输出结果
    if optimizer.check() == sat:
        model = optimizer.model()
        
        all_results = []
        total_prob = 0
        
        print("\n=== 优化结果 ===")
        
        # 计算并打印多样性分数
        diversity_score = 0
        for i in range(num_cases):
            for j in range(i + 1, num_cases):
                diff_var = next(v for v in diversity_vars if str(v).startswith(f'diff_count_{i}_{j}'))
                diversity_score += safe_prob_conversion(model.eval(diff_var))
        
        # 归一化多样性分数
        max_possible_diversity = num_cases * (num_cases - 1) / 2 * num_slippery * num_slippery
        normalized_diversity = diversity_score / max_possible_diversity
        
        for case in range(num_cases):
            # 获取滑块的位置
            slippery_pos = [model.eval(pos).as_long() for pos in all_slippery_positions[case]]
            success_prob = model.eval(all_success_probs[case][0])
            prob_value = safe_prob_conversion(success_prob)
            all_results.append((slippery_pos, success_prob))
            total_prob += prob_value
            
            print(f"\n测试用例 {case + 1}:")
            print(f"滑块位置: {slippery_pos}")
            print(f"成功概率: {success_prob} ({prob_value:.4f})")
            
            # 可视化结果
            grid = ['.'] * total_states
            for pos in slippery_pos:
                grid[pos] = 'S'  # S表示滑块
            for hole in holes:
                grid[hole] = 'H'  # H表示洞
            grid[0] = 'T'  # Start
            grid[total_states-1] = 'G'  # Goal
            
            print("\n网格布局:")
            print("T: 起点, G: 终点, S: 滑块, H: 洞, .: 普通块")
            for i in range(0, total_states, grid_size):
                print(' '.join(grid[i:i+grid_size]))
        
        avg_prob = total_prob / num_cases
        print("\n=== 优化指标 ===")
        print(f"平均成功概率: {avg_prob:.4f}")
        print(f"多样性分数: {normalized_diversity:.4f}")
        print(f"总目标值: {avg_prob - diversity_weight * normalized_diversity:.4f}")
        
        return all_results, avg_prob, normalized_diversity
    else:
        print("无解")
        return None, None, None

if __name__ == "__main__":
    # 生成3个不同的测试用例，每个包含4个滑块
    optimize_multiple_slippery_positions(
        num_cases=3, 
        num_slippery=3, 
        grid_size=4,
        diversity_weight=0.1  # 可以调整这个权重来平衡成功概率和多样性
    )
