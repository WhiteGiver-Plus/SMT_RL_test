import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from traffic_agent import TrafficAgent
from traffic_env import TrafficControlEnv
import time

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

def simulate_episode(env, agent):
    """使用test_visualization.py中的方法模拟一个episode"""
    state = env.reset()
    total_reward = 0
    states = []
    actions = []
    rewards = []
    done = False
    step = 0
    
    while not done and step < 10:  # 限制最大步数为10
        # 选择动作
        action = agent.act(state)
        actions.append(action)
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        rewards.append(reward)
        total_reward += reward
        states.append(state)
        
        state = next_state
        step += 1
    
    return {
        'total_reward': total_reward,
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'final_vehicles': sum(env.waiting_vehicles)
    }

def evaluate_agent_performance(agent, scenario, num_episodes=50):
    """评估agent在特定场景下的表现"""
    rewards = []
    final_vehicles = []
    
    # 创建环境
    env = TrafficControlEnv()
    
    for _ in range(num_episodes):
        # 设置初始状态
        env.waiting_vehicles = np.array([
            scenario['north'],
            scenario['south'],
            scenario['east'],
            scenario['west']
        ])
        env.traffic_lights = np.zeros(4)
        
        # 运行一个episode
        result = simulate_episode(env, agent)
        rewards.append(result['total_reward'])
        final_vehicles.append(result['final_vehicles'])
    
    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_final_vehicles': np.mean(final_vehicles),
        'std_final_vehicles': np.std(final_vehicles),
        'optimal_reward': scenario['max_reward'],
        'reward_gap': np.mean(rewards) - scenario['max_reward']
    }

def generate_comprehensive_report(scenarios, agent):
    """生成综合报告，包含场景特征和agent性能"""
    reports = {}
    
    for scenario_type, type_scenarios in scenarios.items():
        print(f"\n评估 {scenario_type} 类型场景...")
        
        performances = []
        
        for i, scenario in enumerate(type_scenarios):
            print(f"场景 {i+1}/{len(type_scenarios)}", end='\r')
            
            # 评估agent性能
            perf = evaluate_agent_performance(agent, scenario)
            performances.append(perf)
        
        print()  # 换行
        
        reports[scenario_type] = {
            'sample_count': len(type_scenarios),
            'avg_reward': np.mean([p['avg_reward'] for p in performances]),
            'std_reward': np.mean([p['std_reward'] for p in performances]),
            'avg_final_vehicles': np.mean([p['avg_final_vehicles'] for p in performances]),
            'optimal_reward': np.mean([p['optimal_reward'] for p in performances]),
            'reward_gap': np.mean([p['reward_gap'] for p in performances])
        }
    
    return reports

def print_comprehensive_report(reports):
    """打印综合报告"""
    for scenario_type, report in reports.items():
        print(f"\n=== {scenario_type.upper()} 场景分析 ===")
        print(f"样本数量: {report['sample_count']}")
        print(f"平均奖励: {report['avg_reward']:.2f}")
        print(f"奖励标准差: {report['std_reward']:.2f}")
        print(f"平均最终车辆数: {report['avg_final_vehicles']:.2f}")
        print(f"理论最优奖励: {report['optimal_reward']:.2f}")
        print(f"与理论最优差距: {report['reward_gap']:.2f}")

def visualize_comprehensive_results(reports):
    """可视化综合结果"""
    plt.figure(figsize=(15, 10))
    
    # 1. 奖励对比
    plt.subplot(2, 2, 1)
    types = list(reports.keys())
    rewards = [reports[t]['avg_reward'] for t in types]
    optimal = [reports[t]['optimal_reward'] for t in types]
    
    x = np.arange(len(types))
    width = 0.35
    plt.bar(x - width/2, rewards, width, label='Agent Reward')
    plt.bar(x + width/2, optimal, width, label='Optimal Reward')
    plt.xticks(x, types)
    plt.title('Average Rewards by Scenario Type')
    plt.legend()
    
    # 2. 奖励差距
    plt.subplot(2, 2, 2)
    gaps = [reports[t]['reward_gap'] for t in types]
    plt.bar(types, gaps)
    plt.title('Reward Gap (Agent - Optimal)')
    
    # 3. 最终车辆数量
    plt.subplot(2, 2, 3)
    final_vehicles = [reports[t]['avg_final_vehicles'] for t in types]
    plt.bar(types, final_vehicles)
    plt.title('Average Final Vehicles')
    
    # 4. 奖励标准差
    plt.subplot(2, 2, 4)
    std_rewards = [reports[t]['std_reward'] for t in types]
    plt.bar(types, std_rewards)
    plt.title('Reward Standard Deviation')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png')
    plt.close()

def run_comprehensive_evaluation(model_path):
    """运行完整的综合评估"""
    print("初始化agent...")
    state_size = 8  # 4个方向的等待车辆数量 + 4个方向的信号灯状态
    action_size = 4  # 4个可能的动作
    
    # 创建agent并加载训练好的模型
    agent = TrafficAgent(state_size, action_size)
    
    print(f"加载模型 {model_path}...")
    agent.load(model_path)
    agent.epsilon = 0  # 评估时不需要探索
    
    print("加载测试场景...")
    scenarios = load_test_scenarios()
    
    print("开始综合评估...")
    reports = generate_comprehensive_report(scenarios, agent)
    
    print("\n生成可视化结果...")
    visualize_comprehensive_results(reports)
    
    print("\n=== 详细评估报告 ===")
    print_comprehensive_report(reports)
    print("\n可视化结果已保存到 'comprehensive_analysis.png'")

if __name__ == '__main__':
    model_path = 'traffic_agent.pth'
    run_comprehensive_evaluation(model_path)