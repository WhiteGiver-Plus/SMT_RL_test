from z3 import *
import numpy as np
from pathlib import Path
import logging
from Q_table import QLearningAgent, safe_float_conversion

class SlipperyOptimizer:
    """优化滑块位置以最大化原始Q表和变异体的性能差异"""
    
    def __init__(self, grid_size=4, holes=[5, 7, 11]):
        """
        初始化优化器
        
        Args:
            grid_size: 网格大小
            holes: 固定的洞位置列表
        """
        self.grid_size = grid_size
        self.total_states = grid_size * grid_size
        self.holes = holes
        self._setup_logging()
        
    def _setup_logging(self):
        """配置日志输出"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def optimize_slippery_positions(self, original_path, mutants_dir, 
                                  num_slippery=3, timeout=120000):
        """
        优化滑块位置
        
        Args:
            original_path: 原始Q表路径
            mutants_dir: 变异体目录
            num_slippery: 滑块数量
            timeout: 求解超时时间(毫秒)
            
        Returns:
            tuple: (最优滑块位置, 原始模型性能, 变异体平均性能)
        """
        optimizer = Optimize()
        
        # 初始化滑块位置变量
        slippery_positions = [Int(f'slippery_{i}') for i in range(num_slippery)]
        
        # 设置基本约束
        for i in range(num_slippery):
            optimizer.add(slippery_positions[i] >= 0)
            optimizer.add(slippery_positions[i] < self.total_states)
            optimizer.add(slippery_positions[i] != 0)  # 不能在起点
            optimizer.add(slippery_positions[i] != self.total_states-1)  # 不能在终点
            # 不能在洞的位置
            for hole in self.holes:
                optimizer.add(slippery_positions[i] != hole)
        
        # 确保滑块位置不重复且有序
        optimizer.add(Distinct(slippery_positions))
        for i in range(num_slippery-1):
            optimizer.add(slippery_positions[i] < slippery_positions[i+1])
        
        # 加载原始模型和变异体
        original_agent = QLearningAgent(self.total_states, 4)
        original_agent.load_q_table(original_path)
        
        # 加载所有变异体
        mutant_agents = []
        mutants_path = Path(mutants_dir)
        for mutant_file in mutants_path.glob("mutant_*.npy"):
            agent = QLearningAgent(self.total_states, 4)
            agent.load_q_table(str(mutant_file))
            mutant_agents.append(agent)
        
        num_mutants = len(mutant_agents)
        self.logger.info(f"加载了 {num_mutants} 个变异体")
        
        # 为原始模型和变异体创建成功概率变量
        success_probs = {}
        
        # 为原始模型创建概率变量
        for s in range(self.total_states):
            success_probs[f'original_{s}'] = Real(f'success_prob_original_{s}')
            if s == self.total_states-1:  # 终点概率为1
                optimizer.add(success_probs[f'original_{s}'] == 1)
            elif s in self.holes:  # 洞的概率为0
                optimizer.add(success_probs[f'original_{s}'] == 0)
        
        # 为每个变异体创建概率变量
        for i in range(num_mutants):
            for s in range(self.total_states):
                success_probs[f'mutant_{i}_{s}'] = Real(f'success_prob_mutant_{i}_{s}')
                if s == self.total_states-1:
                    optimizer.add(success_probs[f'mutant_{i}_{s}'] == 1)
                elif s in self.holes:
                    optimizer.add(success_probs[f'mutant_{i}_{s}'] == 0)
        
        # 添加转移概率约束
        self._add_transition_constraints(optimizer, original_agent, mutant_agents,
                                      success_probs, slippery_positions)
        
        # 计算目标：最大化原始模型与变异体的性能差异
        original_prob = success_probs['original_0']  # 起点成功概率
        mutants_avg_prob = Sum([success_probs[f'mutant_{i}_0'] 
                              for i in range(num_mutants)]) / num_mutants
        
        # 最大化差异
        optimizer.maximize(original_prob - mutants_avg_prob)
        
        # 设置超时
        optimizer.set("timeout", timeout)
        
        # 求解和处理结果
        return self._process_optimization_results(optimizer, slippery_positions,
                                               success_probs, num_mutants)
    
    def _add_transition_constraints(self, optimizer, original_agent, mutant_agents,
                                  success_probs, slippery_positions):
        """添加状态转移概率约束"""
        # 为原始模型添加约束
        self._add_agent_constraints(optimizer, original_agent, 'original',
                                  success_probs, slippery_positions)
        
        # 为每个变异体添加约束
        for i, agent in enumerate(mutant_agents):
            self._add_agent_constraints(optimizer, agent, f'mutant_{i}',
                                     success_probs, slippery_positions)
    
    def _add_agent_constraints(self, optimizer, agent, prefix, success_probs, slippery_positions):
        """为单个智能体添加转移概率约束"""
        action_matrix = agent.get_optimal_action_matrix()
        
        for s in range(self.total_states):
            if s != self.total_states-1 and s not in self.holes:
                row, col = s // self.grid_size, s % self.grid_size
                action = action_matrix[row, col]
                
                # 计算可能的下一个状态
                next_states = self._get_next_states(row, col, action)
                
                # 判断是否是滑块
                is_slippery = Or([s == pos for pos in slippery_positions])
                
                # 计算转移概率
                next_prob = If(is_slippery,
                             Sum([success_probs[f'{prefix}_{next_states[0]}'] / 3,
                                 success_probs[f'{prefix}_{next_states[1]}'] / 3,
                                 success_probs[f'{prefix}_{next_states[2]}'] / 3]),
                             success_probs[f'{prefix}_{next_states[1]}'])
                
                optimizer.add(success_probs[f'{prefix}_{s}'] == next_prob)
                optimizer.add(success_probs[f'{prefix}_{s}'] >= 0)
                optimizer.add(success_probs[f'{prefix}_{s}'] <= 1)
    
    def _get_next_states(self, row, col, action):
        """计算可能的下一个状态"""
        next_states = []
        for actual_action in [(action - 1) % 4, action, (action + 1) % 4]:
            new_row, new_col = row, col
            if actual_action == 0:    # 左
                new_col = max(0, col - 1)
            elif actual_action == 1:  # 下
                new_row = min(self.grid_size-1, row + 1)
            elif actual_action == 2:  # 右
                new_col = min(self.grid_size-1, col + 1)
            elif actual_action == 3:  # 上
                new_row = max(0, row - 1)
            next_states.append(new_row * self.grid_size + new_col)
        return next_states
    
    def _process_optimization_results(self, optimizer, slippery_positions,
                                   success_probs, num_mutants):
        """处理优化结果"""
        if optimizer.check() == sat:
            model = optimizer.model()
            
            # 获取最优滑块位置
            optimal_positions = [model.eval(pos).as_long() 
                              for pos in slippery_positions]
            
            # 计���性能指标
            original_prob = safe_float_conversion(
                model.eval(success_probs['original_0']).as_decimal(10))
            
            mutant_probs = [safe_float_conversion(
                model.eval(success_probs[f'mutant_{i}_0']).as_decimal(10))
                for i in range(num_mutants)]
            avg_mutant_prob = sum(mutant_probs) / num_mutants
            
            # 输出结果
            self._print_results(optimal_positions, original_prob, 
                              mutant_probs, avg_mutant_prob)
            
            return optimal_positions, original_prob, avg_mutant_prob
        else:
            self.logger.error("优化求解失败")
            return None, None, None
    
    def _print_results(self, positions, original_prob, mutant_probs, avg_mutant_prob):
        """打印优化结果"""
        self.logger.info("\n=== 优化结果 ===")
        self.logger.info(f"最优滑块位置: {positions}")
        self.logger.info(f"原始模型成功概率: {original_prob:.4f}")
        self.logger.info(f"变异体平均成功概率: {avg_mutant_prob:.4f}")
        self.logger.info(f"性能差异: {original_prob - avg_mutant_prob:.4f}")
        
        # 可视化网格
        grid = ['.'] * self.total_states
        for pos in positions:
            grid[pos] = 'S'
        for hole in self.holes:
            grid[hole] = 'H'
        grid[0] = 'T'
        grid[-1] = 'G'
        
        self.logger.info("\n网格布局:")
        self.logger.info("T: 起点, G: 终点, S: 滑块, H: 洞, .: 普通块")
        for i in range(0, self.total_states, self.grid_size):
            self.logger.info(' '.join(grid[i:i+self.grid_size]))

if __name__ == "__main__":
    # 使用示例
    optimizer = SlipperyOptimizer(grid_size=4, holes=[5, 7, 11])
    
    results = optimizer.optimize_slippery_positions(
        original_path='q_table_robust.npy',
        mutants_dir='mutants/q_table_robust',
        num_slippery=3,
        timeout=120000
    )
    
    if results[0] is not None:
        optimal_positions, original_prob, avg_mutant_prob = results
        print(f"\n优化完成！")
        print(f"最优滑块位置: {optimal_positions}")
        print(f"性能差异: {original_prob - avg_mutant_prob:.4f}") 