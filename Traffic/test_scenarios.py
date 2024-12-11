import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_test_scenarios(test_dir='test_cases'):
    """加载所有测试场景并按类型分类"""
    scenarios = {
        'balanced': [],
        'congested': [],
        'unbalanced': []
    }
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.json'):
            with open(os.path.join(test_dir, filename), 'r') as f:
                scenario = json.load(f)
                scenarios[scenario['type']].append(scenario)
    
    return scenarios

def analyze_scenario_characteristics(scenario):
    """分析单个场景的特征"""
    vehicles = [
        scenario['north'],
        scenario['south'],
        scenario['east'],
        scenario['west']
    ]
    
    return {
        'total_vehicles': sum(vehicles),
        'max_direction': max(vehicles),
        'min_direction': min(vehicles),
        'std_dev': np.std(vehicles),
        'imbalance_ratio': max(vehicles) / (min(vehicles) + 1),  # 避免除以0
        'direction_distribution': {
            'north': scenario['north'],
            'south': scenario['south'],
            'east': scenario['east'],
            'west': scenario['west']
        }
    }

def evaluate_scenario_performance(scenario):
    """评估场景的性能指标"""
    optimal_sequence = scenario['optimal_sequence']
    max_reward = scenario.get('max_reward', 0)
    
    vehicles = {
        'north': scenario['north'],
        'south': scenario['south'],
        'east': scenario['east'],
        'west': scenario['west']
    }
    
    initial_load = sum(vehicles.values())
    
    return {
        'sequence_length': len(optimal_sequence),
        'max_reward': max_reward,
        'initial_load': initial_load,
        'reward_per_vehicle': max_reward / initial_load if initial_load > 0 else 0
    }

def generate_type_report(scenarios):
    """生成每种类型的详细报告"""
    type_reports = {}
    
    for scenario_type, type_scenarios in scenarios.items():
        characteristics = []
        performances = []
        
        for scenario in type_scenarios:
            chars = analyze_scenario_characteristics(scenario)
            perf = evaluate_scenario_performance(scenario)
            characteristics.append(chars)
            performances.append(perf)
        
        # 计算统计数据
        type_reports[scenario_type] = {
            'sample_count': len(type_scenarios),
            'characteristics': {
                'avg_total_vehicles': np.mean([c['total_vehicles'] for c in characteristics]),
                'avg_imbalance_ratio': np.mean([c['imbalance_ratio'] for c in characteristics]),
                'avg_std_dev': np.mean([c['std_dev'] for c in characteristics]),
                'direction_averages': {
                    'north': np.mean([c['direction_distribution']['north'] for c in characteristics]),
                    'south': np.mean([c['direction_distribution']['south'] for c in characteristics]),
                    'east': np.mean([c['direction_distribution']['east'] for c in characteristics]),
                    'west': np.mean([c['direction_distribution']['west'] for c in characteristics])
                }
            },
            'performance': {
                'avg_max_reward': np.mean([p['max_reward'] for p in performances]),
                'avg_reward_per_vehicle': np.mean([p['reward_per_vehicle'] for p in performances])
            }
        }
    
    return type_reports

def visualize_type_comparison(reports):
    """可视化不同类型场景的对比"""
    plt.figure(figsize=(15, 12))
    
    # 1. 车辆总量对比
    plt.subplot(2, 2, 1)
    types = list(reports.keys())
    total_vehicles = [reports[t]['characteristics']['avg_total_vehicles'] for t in types]
    plt.bar(types, total_vehicles)
    plt.title('Average Total Vehicles by Type')
    plt.ylabel('Vehicles')
    
    # 2. 方向分布
    plt.subplot(2, 2, 2)
    directions = ['north', 'south', 'east', 'west']
    x = np.arange(len(types))
    width = 0.2
    for i, direction in enumerate(directions):
        values = [reports[t]['characteristics']['direction_averages'][direction] for t in types]
        plt.bar(x + i*width, values, width, label=direction)
    plt.xticks(x + width*1.5, types)
    plt.title('Direction Distribution by Type')
    plt.legend()
    
    # 3. 不平衡率
    plt.subplot(2, 2, 3)
    imbalance = [reports[t]['characteristics']['avg_imbalance_ratio'] for t in types]
    plt.bar(types, imbalance)
    plt.title('Average Imbalance Ratio by Type')
    plt.ylabel('Imbalance Ratio')
    
    # 4. 平均奖励
    plt.subplot(2, 2, 4)
    rewards = [reports[t]['performance']['avg_max_reward'] for t in types]
    plt.bar(types, rewards)
    plt.title('Average Max Reward by Type')
    plt.ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('type_comparison.png')
    plt.close()

def print_type_report(reports):
    """打印类型报告"""
    for scenario_type, report in reports.items():
        print(f"\n=== {scenario_type.upper()} 场景分析 ===")
        print(f"样本数量: {report['sample_count']}")
        
        print("\n特征统计:")
        print(f"平均车辆总量: {report['characteristics']['avg_total_vehicles']:.2f}")
        print(f"平均不平衡率: {report['characteristics']['avg_imbalance_ratio']:.2f}")
        print(f"方向标准差: {report['characteristics']['avg_std_dev']:.2f}")
        
        print("\n方向分布:")
        for direction, avg in report['characteristics']['direction_averages'].items():
            print(f"{direction}: {avg:.2f}")
        
        print("\n性能指标:")
        print(f"平均最大奖励: {report['performance']['avg_max_reward']:.2f}")
        print(f"每车辆平均奖励: {report['performance']['avg_reward_per_vehicle']:.2f}")

def run_type_analysis():
    """运行类型分析"""
    print("加载测试场景...")
    scenarios = load_test_scenarios()
    
    print("生成类型报告...")
    reports = generate_type_report(scenarios)
    
    print("生成可视化对比...")
    visualize_type_comparison(reports)
    
    print("\n=== 详细类型报告 ===")
    print_type_report(reports)
    print("\n可视化结果已保存到 'type_comparison.png'")

if __name__ == '__main__':
    run_type_analysis()