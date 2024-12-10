import torch
import numpy as np
from dc_generator import DataCenterGenerator
from dc_dqn import DCEnvironment, DQNAgent
import matplotlib.pyplot as plt
from typing import Dict, List

class AgentEvaluator:
    def __init__(self, agent_path: str):
        """
        初始化评估器
        agent_path: 训练好的模型权重文件路径
        """
        self.agent_path = agent_path
        
    def load_agent(self, state_size: int, action_size: int) -> DQNAgent:
        """加载训练好的智能体"""
        agent = DQNAgent(state_size, action_size)
        agent.policy_net.load_state_dict(
            torch.load(self.agent_path)
        )
        agent.epsilon = 0  # 评估时不使用探索
        return agent
        
    def evaluate_scenario(self, env_data: Dict, num_episodes: int = 100) -> Dict:
        """评估智能体在特定场景下的表现"""
        env = DCEnvironment(env_data)
        state_size = env.num_servers * 3 + env.num_tasks
        action_size = env.num_servers * env.num_tasks
        
        agent = self.load_agent(state_size, action_size)
        
        metrics = {
            'rewards': [],
            'completion_rates': [],
            'resource_utilization': [],
            'load_balance': []
        }
        
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.act(state)
                task_id = action // env.num_servers
                server_id = action % env.num_servers
                
                next_state, reward, done = env.step((task_id, server_id))
                total_reward += reward
                state = next_state
                
            # 计算完成率
            completion_rate = (env.num_tasks - len(env.pending_tasks)) / env.num_tasks
            
            # 计算资源利用率
            utilization = np.mean([
                np.mean(usage / capacity) 
                for usage, capacity in zip(env.server_usage, env.server_capacities)
            ])
            
            # 计算负载均衡度
            load_balance = 1 - np.std([
                np.mean(usage / capacity)
                for usage, capacity in zip(env.server_usage, env.server_capacities)
            ])
            
            metrics['rewards'].append(total_reward)
            metrics['completion_rates'].append(completion_rate)
            metrics['resource_utilization'].append(utilization)
            metrics['load_balance'].append(load_balance)
            
        return {k: np.mean(v) for k, v in metrics.items()}
        
    def evaluate_all_scenarios(self, num_episodes: int = 100):
        """评估所有测试场景"""
        generator = DataCenterGenerator()
        scenarios = ['basic', 'resource_tight', 'complex_dependency', 'burst_arrival']
        
        results = {}
        for scenario in scenarios:
            print(f"\nEvaluating {scenario} scenario...")
            generator.scenario_type = scenario
            env_data = generator.generate_environment()
            
            metrics = self.evaluate_scenario(env_data, num_episodes)
            results[scenario] = metrics
            
            print(f"Results for {scenario}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
                
        self.plot_results(results)
        return results
        
    def plot_results(self, results: Dict):
        """可视化评估结果"""
        metrics = list(next(iter(results.values())).keys())
        scenarios = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[s][metric] for s in scenarios]
            axes[i].bar(scenarios, values)
            axes[i].set_title(metric)
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()

def main():
    # 假设模型保存在这个路径
    agent_path = "trained_agent.pth"
    
    # 创建评估器
    evaluator = AgentEvaluator(agent_path)
    
    # 评估所有场景
    results = evaluator.evaluate_all_scenarios()
    
    # 保存评估结果
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 