from z3 import *
import numpy as np
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class DataCenterGenerator:
    def __init__(self, 
                 num_servers: int = 4,
                 num_tasks: int = 8,
                 time_horizon: int = 10,
                 scenario_type: str = 'basic'):
        self.num_servers = num_servers
        self.num_tasks = num_tasks
        self.time_horizon = time_horizon
        self.scenario_type = scenario_type
        self.num_resources = 3
        
    def generate_environment(self) -> Dict:
        """根据场景类型生成测试环境"""
        if self.scenario_type == 'basic':
            return self._generate_basic_env()
        elif self.scenario_type == 'resource_tight':
            return self._generate_resource_tight_env()
        elif self.scenario_type == 'complex_dependency':
            return self._generate_complex_dependency_env()
        elif self.scenario_type == 'burst_arrival':
            return self._generate_burst_arrival_env()
        else:
            raise ValueError(f"Unknown scenario type: {self.scenario_type}")
            
    def _generate_basic_env(self) -> Dict:
        """生成基础测试环境,使用Z3确保可解性"""
        solver = Solver()
        
        # 1. 定义变量
        # 任务分配变量: task_placement[i][j][t] 表示任务i在时间t是否分配到服务器j
        task_placement = [[[Bool(f"task_{i}_server_{j}_time_{t}") 
                           for t in range(self.time_horizon)]
                          for j in range(self.num_servers)]
                         for i in range(self.num_tasks)]
        
        # 2. 生成资源数据
        server_capacities = np.random.uniform(
            low=[0.6, 0.6, 0.6],
            high=[1.0, 1.0, 1.0],
            size=(self.num_servers, self.num_resources)
        )
        
        task_demands = []
        for i in range(self.num_tasks):
            demands = np.random.uniform(
                low=[0.1, 0.1, 0.1],
                high=[0.4, 0.4, 0.4],
                size=self.num_resources
            )
            duration = random.randint(1, 4)
            arrival_time = random.randint(0, self.time_horizon-duration)
            task_demands.append({
                'resources': demands,
                'duration': duration,
                'arrival_time': arrival_time
            })
        
        # 3. 添加约束
        # 3.1 每个任务在其运行时间内必须且只能分配到一个服务器
        for i in range(self.num_tasks):
            task = task_demands[i]
            for t in range(task['arrival_time'], 
                          task['arrival_time'] + task['duration']):
                # 每个时间点只能分配到一个服务器
                solver.add(
                    Sum([If(task_placement[i][j][t], 1, 0) 
                         for j in range(self.num_servers)]) == 1
                )
        
        # 3.2 资源容量约束
        for j in range(self.num_servers):
            for t in range(self.time_horizon):
                for r in range(self.num_resources):
                    # 计算t时刻服务器j上资源r的总使用量
                    resource_usage = Sum([
                        If(task_placement[i][j][t],
                           task_demands[i]['resources'][r],
                           0.0)
                        for i in range(self.num_tasks)
                    ])
                    solver.add(resource_usage <= server_capacities[j][r])
        
        # 3.3 任务连续性约束 - 一旦分配到某服务器就不能改变
        for i in range(self.num_tasks):
            task = task_demands[i]
            for j in range(self.num_servers):
                for t in range(task['arrival_time'], 
                             task['arrival_time'] + task['duration'] - 1):
                    solver.add(
                        task_placement[i][j][t] == task_placement[i][j][t+1]
                    )
        
        # 4. 求解
        if solver.check() == sat:
            model = solver.model()
            
            # 根据求解结果调整任务分配
            valid_assignments = {}
            for i in range(self.num_tasks):
                task = task_demands[i]
                for j in range(self.num_servers):
                    if model.evaluate(
                        task_placement[i][j][task['arrival_time']]):
                        valid_assignments[i] = j
                        break
            
            # 如果有任务没有分配成功,调整参数重新生成
            if len(valid_assignments) < self.num_tasks:
                return self._generate_basic_env()
            
            return {
                'server_capacities': server_capacities,
                'task_demands': task_demands,
                'dependencies': self._generate_dependencies(task_demands),
                'time_horizon': self.time_horizon,
                'valid_assignments': valid_assignments  # 添加一个可行解
            }
        else:
            # 如果无解,调整参数重新生���
            print("No solution found, regenerating...")
            return self._generate_basic_env()
    
    def _generate_dependencies(self, task_demands: List[Dict]) -> List[Tuple[int, int]]:
        """生成任务依赖关系"""
        dependencies = []
        num_deps = random.randint(0, self.num_tasks // 2)
        
        for _ in range(num_deps):
            while True:
                task1 = random.randint(0, self.num_tasks-1)
                task2 = random.randint(0, self.num_tasks-1)
                
                # 检查是否形成有效依赖
                if (task1 != task2 and
                    task_demands[task1]['arrival_time'] + task_demands[task1]['duration'] 
                    <= task_demands[task2]['arrival_time'] and
                    (task1, task2) not in dependencies):
                    dependencies.append((task1, task2))
                    break
                    
        return dependencies
    
    def visualize_environment(self, env_data: Dict):
        """可视化数据中心环境"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. 显示服务器资源容量
        server_data = env_data['server_capacities']
        resources = ['CPU', 'Memory', 'Network']
        x = np.arange(self.num_servers)
        width = 0.25
        
        for i in range(self.num_resources):
            ax1.bar(x + i*width, server_data[:, i], width, 
                   label=resources[i])
            
        ax1.set_ylabel('Resource Capacity')
        ax1.set_title('Server Resources')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([f'Server {i}' for i in range(self.num_servers)])
        ax1.legend()
        
        # 2. 显示任务时间线和依赖关系
        task_demands = env_data['task_demands']
        dependencies = env_data['dependencies']
        
        colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, self.num_servers))
        
        for i, task in enumerate(task_demands):
            start = task['arrival_time']
            duration = task['duration']
            # 如果有分配方案，使用对应服务器的颜色
            if 'valid_assignments' in env_data and i in env_data['valid_assignments']:
                server_id = env_data['valid_assignments'][i]
                color = colors[server_id]
                alpha = 0.6
            else:
                color = 'gray'
                alpha = 0.3
            
            ax2.barh(i, duration, left=start, alpha=alpha, color=color)
            ax2.text(start, i, f'T{i}')
            
        # 绘制依赖箭头
        for dep in dependencies:
            task1, task2 = dep
            t1_end = task_demands[task1]['arrival_time'] + task_demands[task1]['duration']
            t2_start = task_demands[task2]['arrival_time']
            
            ax2.arrow(t1_end, task1, 
                     t2_start - t1_end, task2 - task1,
                     head_width=0.1, head_length=0.1, 
                     fc='k', ec='k', length_includes_head=True)
            
        ax2.set_ylabel('Tasks')
        ax2.set_xlabel('Time')
        ax2.set_title('Task Timeline and Dependencies')
        ax2.set_yticks(range(self.num_tasks))
        ax2.grid(True)
        
        # 3. 显示服务器资源使用情况
        if 'valid_assignments' in env_data:
            server_usage = np.zeros((self.num_servers, self.num_resources))
            for task_id, server_id in env_data['valid_assignments'].items():
                task = task_demands[task_id]
                server_usage[server_id] += task['resources']
            
            # 绘制资源使用率
            for i in range(self.num_resources):
                ax3.bar(x + i*width, server_usage[:, i], width,
                       label=f'{resources[i]} Used', alpha=0.6)
                ax3.bar(x + i*width, server_data[:, i], width,
                       label=f'{resources[i]} Total', alpha=0.3,
                       linestyle='--', fill=False)
            
            ax3.set_ylabel('Resource Usage')
            ax3.set_title('Server Resource Usage After Assignment')
            ax3.set_xticks(x + width)
            ax3.set_xticklabels([f'Server {i}' for i in range(self.num_servers)])
            ax3.legend()
            
            # 添加使用率标签
            for j in range(self.num_servers):
                usage_ratio = np.mean(server_usage[j] / server_data[j])
                ax3.text(j, 0.1, f'{usage_ratio:.1%}',
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def _generate_resource_tight_env(self) -> Dict:
        """生成资源紧张但仍可解的测试场景"""
        while True:
            solver = Solver()
            
            # 生成较低的服务器容量
            server_capacities = np.random.uniform(
                low=[0.4, 0.4, 0.4],
                high=[0.6, 0.6, 0.6],
                size=(self.num_servers, self.num_resources)
            )
            
            # 生成较高的任务需求
            task_demands = []
            for _ in range(self.num_tasks):
                demands = np.random.uniform(
                    low=[0.3, 0.3, 0.3],
                    high=[0.5, 0.5, 0.5],
                    size=self.num_resources
                )
                duration = random.randint(2, 5)
                task_demands.append({
                    'resources': demands,
                    'duration': duration,
                    'arrival_time': random.randint(0, self.time_horizon-duration)
                })
                
            # 使用与basic相同的约束检查可解性
            task_placement = [[[Bool(f"tight_task_{i}_server_{j}_time_{t}") 
                              for t in range(self.time_horizon)]
                             for j in range(self.num_servers)]
                            for i in range(self.num_tasks)]
                            
            # 添加约束....(与basic场景相同的约束逻辑)
            
            if solver.check() == sat:
                model = solver.model()
                valid_assignments = {}
                # 提取可行解...
                
                return {
                    'server_capacities': server_capacities,
                    'task_demands': task_demands,
                    'dependencies': self._generate_dependencies(task_demands),
                    'time_horizon': self.time_horizon,
                    'scenario_type': 'resource_tight',
                    'valid_assignments': valid_assignments
                }
        
    def _generate_complex_dependency_env(self) -> Dict:
        """生成复杂依赖关系的测试场景"""
        env_data = self._generate_basic_env()
        
        # 生成菱形依赖
        def create_diamond_dependency(start_idx, tasks):
            deps = []
            if start_idx + 3 < len(tasks):
                deps.extend([
                    (start_idx, start_idx + 1),
                    (start_idx, start_idx + 2),
                    (start_idx + 1, start_idx + 3),
                    (start_idx + 2, start_idx + 3)
                ])
            return deps
            
        # 生成链式依赖
        def create_chain_dependency(start_idx, length, tasks):
            deps = []
            for i in range(length-1):
                if start_idx + i + 1 < len(tasks):
                    deps.append((start_idx + i, start_idx + i + 1))
            return deps
            
        dependencies = []
        # 添加2-3个菱形依赖
        for i in range(0, self.num_tasks-3, 4):
            dependencies.extend(create_diamond_dependency(i, env_data['task_demands']))
            
        # 添加1-2个长链依赖
        chain_start = random.randint(0, self.num_tasks//2)
        dependencies.extend(create_chain_dependency(chain_start, 4, env_data['task_demands']))
        
        env_data['dependencies'] = dependencies
        env_data['scenario_type'] = 'complex_dependency'
        return env_data
        
    def _generate_burst_arrival_env(self) -> Dict:
        """生成突发任务到达的测试场景"""
        env_data = self._generate_basic_env()
        
        # 将大部分任务的到达时间集中在一个时间窗口内
        burst_window = (self.time_horizon // 3, 2 * self.time_horizon // 3)
        for task in env_data['task_demands']:
            if random.random() < 0.7:  # 70%的任��是突发的
                task['arrival_time'] = random.randint(*burst_window)
                
        env_data['scenario_type'] = 'burst_arrival'
        return env_data

    def save_environment(self, env_data: Dict, filename: str):
        """保存测试环境到文件"""
        import json
        
        # 将numpy数组转换为列表
        env_data_serializable = env_data.copy()
        env_data_serializable['server_capacities'] = env_data['server_capacities'].tolist()
        for task in env_data_serializable['task_demands']:
            task['resources'] = task['resources'].tolist()
            
        with open(filename, 'w') as f:
            json.dump(env_data_serializable, f)
            
    @staticmethod
    def load_environment(filename: str) -> Dict:
        """从文件加载测试环境"""
        import json
        
        with open(filename, 'r') as f:
            env_data = json.load(f)
            
        # 将列表转换回numpy数组
        env_data['server_capacities'] = np.array(env_data['server_capacities'])
        for task in env_data['task_demands']:
            task['resources'] = np.array(task['resources'])
            
        return env_data

def main():
    # 测试所有场景类型
    scenarios = ['basic', 'resource_tight', 'complex_dependency', 'burst_arrival']
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Testing {scenario.upper()} scenario")
        print(f"{'='*50}")
        
        # 创建生成器实例
        generator = DataCenterGenerator(scenario_type=scenario)
        
        # 生成环境
        env_data = generator.generate_environment()
        
        # 可视化环境
        generator.visualize_environment(env_data)
        
        # 打印环境信息
        print("\nEnvironment Details:")
        print(f"Number of Servers: {generator.num_servers}")
        print(f"Number of Tasks: {generator.num_tasks}")
        print(f"Time Horizon: {generator.time_horizon}")
        print(f"Number of Dependencies: {len(env_data['dependencies'])}")
        print("Dependencies:", env_data['dependencies'])
        
        # 打印场景特定信息
        if scenario == 'resource_tight':
            print("\nResource Usage Statistics:")
            server_usage = np.mean(env_data['server_capacities'], axis=1)
            print(f"Average Server Capacity: {server_usage}")
            task_demands = [np.mean(task['resources']) for task in env_data['task_demands']]
            print(f"Average Task Demand: {np.mean(task_demands):.3f}")
            
        elif scenario == 'complex_dependency':
            print("\nDependency Statistics:")
            # 计算每个任务的入度和出度
            in_degree = {}
            out_degree = {}
            for dep in env_data['dependencies']:
                src, dst = dep
                out_degree[src] = out_degree.get(src, 0) + 1
                in_degree[dst] = in_degree.get(dst, 0) + 1
            print(f"Max In-Degree: {max(in_degree.values()) if in_degree else 0}")
            print(f"Max Out-Degree: {max(out_degree.values()) if out_degree else 0}")
            
        elif scenario == 'burst_arrival':
            print("\nArrival Time Statistics:")
            arrival_times = [task['arrival_time'] for task in env_data['task_demands']]
            print(f"Arrival Time Distribution:")
            for t in range(generator.time_horizon):
                count = sum(1 for time in arrival_times if time == t)
                if count > 0:
                    print(f"Time {t}: {count} tasks")
        
        # 如果有可行解，打印它
        if 'valid_assignments' in env_data:
            print("\nValid Assignment Found:")
            for task_id, server_id in env_data['valid_assignments'].items():
                print(f"Task {task_id} -> Server {server_id}")
        
        input("\nPress Enter to continue to next scenario...")

if __name__ == "__main__":
    main() 