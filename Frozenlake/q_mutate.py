import numpy as np
import os
from pathlib import Path
import logging

class QTableMutator:
    """Q表变异器类，用于生成Q表的变异体"""
    
    def __init__(self, mutation_rate=0.1, mutation_strength=0.2):
        """
        初始化变异器
        
        Args:
            mutation_rate: 需要修改的值的比例 (0-1)
            mutation_strength: 修改的强度 (0-1)
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志输出"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def mutate_q_table(self, original_path, mutated_path):
        """
        对Q表进行随机修改
        
        Args:
            original_path: 原始Q表的路径
            mutated_path: 修改后Q表的保存路径
            
        Returns:
            bool: 变异是否成功
        """
        try:
            # 加载原始Q表
            q_table = np.load(original_path)
            
            # 创建修改后的Q表副本
            mutated_q_table = q_table.copy()
            
            # 获取Q表的形状
            total_elements = q_table.size
            
            # 计算需要修改的元素数量
            num_mutations = int(total_elements * self.mutation_rate)
            
            # 随机选择要修改的位置
            mutation_indices = np.random.choice(total_elements, num_mutations, replace=False)
            
            # 将Q表展平以便修改
            flat_q_table = mutated_q_table.flatten()
            
            # 对选中的位置进行修改
            for idx in mutation_indices:
                # 生成随机修改值 (-mutation_strength 到 +mutation_strength)
                mutation = (np.random.random() * 2 - 1) * self.mutation_strength
                flat_q_table[idx] *= (1 + mutation)
            
            # 恢复Q表形状
            mutated_q_table = flat_q_table.reshape(mutated_q_table.shape)
            
            # 确保保存路径的目录存在
            os.makedirs(os.path.dirname(mutated_path), exist_ok=True)
            
            # 保存修改后的Q表
            np.save(mutated_path, mutated_q_table)
            
            # 计算并记录修改统计信息
            diff = np.abs(q_table - mutated_q_table)
            self.logger.info(f"\n=== Q表修改统计 ===")
            self.logger.info(f"原始文件: {original_path}")
            self.logger.info(f"变异文件: {mutated_path}")
            self.logger.info(f"修改的元素数量: {num_mutations}")
            self.logger.info(f"平均修改幅度: {np.mean(diff):.4f}")
            self.logger.info(f"最大修改幅度: {np.max(diff):.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"变异过程出错: {str(e)}")
            return False

def generate_mutants(source_files, output_base_dir='mutants', num_mutants=100, 
                    mutation_rate=0.1, mutation_strength=0.2):
    """
    为多个Q表文件生成变异体
    
    Args:
        source_files: Q表文件路径列表或单个文件路径
        output_base_dir: 变异体保存的基础目录
        num_mutants: 每个源文件要生成的变异体数量
        mutation_rate: 需要修改的值的比例 (0-1)
        mutation_strength: 修改的强度 (0-1)
    
    Returns:
        dict: 每个源文件的变异结果统计
    """
    # 确保source_files是列表
    if isinstance(source_files, (str, Path)):
        source_files = [source_files]
    
    # 创建变异器实例
    mutator = QTableMutator(mutation_rate, mutation_strength)
    
    results = {}
    
    for source_file in source_files:
        source_path = Path(source_file)
        if not source_path.exists():
            mutator.logger.error(f"源文件不存在: {source_file}")
            continue
            
        # 为每个源文件创建专门的输出目录
        output_dir = Path(output_base_dir) / source_path.stem
        os.makedirs(output_dir, exist_ok=True)
        
        # 复制原始文件作为基准
        base_mutant_path = output_dir / f"original_{source_path.name}"
        np.save(base_mutant_path, np.load(source_file))
        
        # 生成变异体
        success_count = 0
        for i in range(1, num_mutants + 1):
            mutant_path = output_dir / f"mutant_{i}.npy"
            if mutator.mutate_q_table(source_file, mutant_path):
                success_count += 1
        
        results[source_file] = {
            'total': num_mutants,
            'success': success_count,
            'output_dir': str(output_dir)
        }
        
        mutator.logger.info(f"\n=== {source_path.name} 变异完成 ===")
        mutator.logger.info(f"成功生成变异体: {success_count}/{num_mutants}")
        mutator.logger.info(f"保存位置: {output_dir}")
    
    return results

if __name__ == "__main__":
    # 示例用法
    source_files = [
        'SMT_RL\q_table_no_slippery.npy',
        'SMT_RL\q_table_roburst.npy',
    ]
    
    # 为每个文件生成变异体
    results = generate_mutants(
        source_files=source_files,
        output_base_dir='mutants',
        num_mutants=50,
        mutation_rate=0.2,
        mutation_strength=0.1
    )
    
    # 打印总结果
    print("\n=== 变异总结 ===")
    for source, result in results.items():
        print(f"\n文件: {source}")
        print(f"成功率: {result['success']}/{result['total']}")
        print(f"输出目录: {result['output_dir']}")