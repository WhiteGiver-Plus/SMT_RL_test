from z3 import *
import json
import os
import numpy as np

def generate_test_case(solver, constraints):
    """根据约束生成一个测试用例"""
    if solver.check() == sat:
        model = solver.model()
        return model
    return None

def create_traffic_scenario():
    """创建一个交通场景的约束求解器"""
    solver = Solver()
    
    # 为4个方向创建整数变量(初始等待车辆数)
    north = Int('north')
    south = Int('south')
    east = Int('east')
    west = Int('west')
    
    # 添加基本约束
    solver.add(north >= 0, north <= 50)
    solver.add(south >= 0, south <= 50)
    solver.add(east >= 0, east <= 50)
    solver.add(west >= 0, west <= 50)
    
    return solver, [north, south, east, west]

def generate_balanced_scenario():
    """生成一个相对平衡的交通场景"""
    solver, vars = create_traffic_scenario()
    north, south, east, west = vars
    
    # 添加平衡约束
    total = north + south + east + west
    solver.add(total >= 40, total <= 100)
    
    # 确保方向间的车辆数量差异不会太大
    for i in range(len(vars)):
        for j in range(i + 1, len(vars)):  # 避免重复比较
            solver.add(Abs(vars[i] - vars[j]) <= 10)
    
    model = generate_test_case(solver, vars)
    if model:
        return {
            'north': model[north].as_long(),
            'south': model[south].as_long(),
            'east': model[east].as_long(),
            'west': model[west].as_long(),
            'type': 'balanced',
            'optimal_sequence': calculate_optimal_sequence
        }
    return None

def generate_congested_scenario():
    """生成一个拥堵的交通场景"""
    solver, vars = create_traffic_scenario()
    north, south, east, west = vars
    
    # 添加拥堵约束
    solver.add(north + south + east + west >= 80)
    solver.add(north + south + east + west <= 120)
    solver.add(north >= 15, south >= 15, east >= 15, west >= 15)
    
    model = generate_test_case(solver, vars)
    if model:
        return {
            'north': model[north].as_long(),
            'south': model[south].as_long(),
            'east': model[east].as_long(),
            'west': model[west].as_long(),
            'type': 'congested',
            'optimal_sequence': calculate_optimal_sequence
        }
    return None

def generate_unbalanced_scenario():
    """生成一个不平衡的交通场景"""
    solver, vars = create_traffic_scenario()
    north, south, east, west = vars
    
    # 添加不平衡约束 - 某些方向特别拥堵
    solver.add(Or(
        And(north >= 40, south <= 10),
        And(east >= 40, west <= 10),
        And(south >= 40, north <= 10),
        And(west >= 40, east <= 10)
    ))
    
    model = generate_test_case(solver, vars)
    if model:
        return {
            'north': model[north].as_long(),
            'south': model[south].as_long(),
            'east': model[east].as_long(),
            'west': model[west].as_long(),
            'type': 'unbalanced',
            'optimal_sequence': calculate_optimal_sequence
        }
    return None

def calculate_optimal_sequence(scenario):
    """计算给定场景的理论最优动作序列和最大奖励"""
    vehicles = [
        scenario['north'],
        scenario['south'],
        scenario['east'],
        scenario['west']
    ]
    
    # 简单的贪心策略：优先处理等待车辆最多的方向
    sequence = []
    rewards = 0.0  # 使用Python float
    remaining_vehicles = vehicles.copy()
    
    for _ in range(10):  # 假设测试10个时间步
        # 找到等待车辆最多的方向
        max_idx = int(np.argmax(remaining_vehicles))  # 转换为Python int
        sequence.append(max_idx)
        
        # 计算该动作的奖励
        cleared_vehicles = min(8, remaining_vehicles[max_idx])
        remaining_vehicles[max_idx] = max(0, remaining_vehicles[max_idx] - cleared_vehicles)
        
        # 其他方向可能增加车辆
        for i in range(4):
            if i != max_idx:
                remaining_vehicles[i] = min(50, remaining_vehicles[i] + 2)
        
        # 计算奖励
        total_waiting = sum(remaining_vehicles)
        reward = -0.1 * total_waiting + cleared_vehicles
        rewards += reward
    
    return {
        'sequence': [int(x) for x in sequence],  # 确保序列中的所有数字都是Python int
        'max_reward': float(rewards)  # 确保奖励值是Python float
    }

def generate_test_suite(output_dir='test_cases'):
    """生成一套测试用例并保存到文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    scenarios = []
    
    # 生成多个不同类型的场景
    for _ in range(5):
        scenarios.extend([
            generate_balanced_scenario(),
            generate_congested_scenario(),
            generate_unbalanced_scenario()
        ])
    
    # 保存到文件
    for i, scenario in enumerate(scenarios):
        if scenario:
            optimal = calculate_optimal_sequence(scenario)
            scenario['optimal_sequence'] = optimal['sequence']
            scenario['max_reward'] = optimal['max_reward']
            
            filename = os.path.join(output_dir, f'scenario_{i}.json')
            with open(filename, 'w') as f:
                json.dump(scenario, f, indent=4)

if __name__ == '__main__':
    generate_test_suite()
