import numpy as np

def mutate_q_table(original_path='q_table.npy', mutated_path='q_table2.npy', mutation_rate=0.1, mutation_strength=0.2):
    """
    对Q表进行随机修改
    
    Args:
        original_path: 原始Q表的路径
        mutated_path: 修改后Q表的保存路径
        mutation_rate: 需要修改的值的比例 (0-1)
        mutation_strength: 修改的强度 (0-1)
    """
    # 加载原始Q表
    q_table = np.load(original_path)
    
    # 创建修改后的Q表副本
    mutated_q_table = q_table.copy()
    
    # 获取Q表的形状
    total_elements = q_table.size
    
    # 计算需要修改的元素数量
    num_mutations = int(total_elements * mutation_rate)
    
    # 随机选择要修改的位置
    mutation_indices = np.random.choice(total_elements, num_mutations, replace=False)
    
    # 将Q表展平以便修改
    flat_q_table = mutated_q_table.flatten()
    
    # 对选中的位置进行修改
    for idx in mutation_indices:
        # 生成随机修改值 (-mutation_strength 到 +mutation_strength)
        mutation = (np.random.random() * 2 - 1) * mutation_strength
        flat_q_table[idx] *= (1 + mutation)
    
    # 恢复Q表形状
    mutated_q_table = flat_q_table.reshape(mutated_q_table.shape)
    
    # 保存修改后的Q表
    np.save(mutated_path, mutated_q_table)
    
    # 计算并打印修改统计信息
    diff = np.abs(q_table - mutated_q_table)
    print(f"\n=== Q表修改统计 ===")
    print(f"修改的元素数量: {num_mutations}")
    print(f"平均修改幅度: {np.mean(diff):.4f}")
    print(f"最大修改幅度: {np.max(diff):.4f}")
    print(f"修改后的Q表已保存到: {mutated_path}")

if __name__ == "__main__":
    # 复制原始Q表作为第一个Q表
    np.save('q_table1.npy', np.load('q_table.npy'))
    
    # 创建修改后的第二个Q表
    mutate_q_table(
        original_path='q_table.npy',
        mutated_path='q_table2.npy',
        mutation_rate=0.2,    # 修改10%的值
        mutation_strength=2  # 最大修改幅度为原值的±20%
    )