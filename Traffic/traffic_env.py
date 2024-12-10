import gym
import numpy as np
from gym import spaces

class TrafficControlEnv(gym.Env):
    def __init__(self):
        super(TrafficControlEnv, self).__init__()
        
        # 定义状态空间
        # 状态包括: [等待车辆数量(4个方向), 当前信号灯状态(4个方向)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([50, 50, 50, 50, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # 定义动作空间
        # 动作: 0-3 表示给予哪个方向的绿灯
        self.action_space = spaces.Discrete(4)
        
        # 初始化环境状态
        self.reset()
        
    def reset(self):
        # 随机初始化等待车辆数量
        self.waiting_vehicles = np.random.randint(0, 20, size=4)
        # 初始化信号灯状态 (0: 红灯, 1: 绿灯)
        self.traffic_lights = np.zeros(4)
        self.time_step = 0
        self.max_steps = 100
        
        return self._get_state()
        
    def step(self, action):
        self.time_step += 1
        
        # 更新信号灯状态
        self.traffic_lights = np.zeros(4)
        self.traffic_lights[action] = 1
        
        # 更新等待车辆数量
        # 绿灯方向车辆减少
        self.waiting_vehicles[action] = max(0, self.waiting_vehicles[action] - np.random.randint(3, 8))
        
        # 其他方向随机增加车辆
        for i in range(4):
            if i != action:
                self.waiting_vehicles[i] = min(50, self.waiting_vehicles[i] + np.random.randint(0, 4))
        
        # 计算奖励
        reward = self._calculate_reward(action)
        
        # 判断是否结束
        done = self.time_step >= self.max_steps
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        # 组合等待车辆数量和信号灯状态作为观察状态
        return np.concatenate([self.waiting_vehicles, self.traffic_lights])
    
    def _calculate_reward(self, action):
        # 计算奖励：减少等待车辆数量为正奖励，累积等待车辆为负奖励
        total_waiting = np.sum(self.waiting_vehicles)
        cleared_vehicles = self.waiting_vehicles[action]
        
        reward = -0.1 * total_waiting + cleared_vehicles
        return reward
    
    def render(self, mode='human'):
        if mode == 'human':
            print("\n=== Traffic Control Visualization ===")
            directions = ['North', 'South', 'East', 'West']
            for i in range(4):
                light_status = "🟢" if self.traffic_lights[i] == 1 else "🔴"
                print(f"{directions[i]}: {light_status} Waiting: {int(self.waiting_vehicles[i])}")
            print("================================")

# 注册环境
from gym.envs.registration import register

register(
    id='TrafficControl-v0',
    entry_point='traffic_env:TrafficControlEnv',
)
