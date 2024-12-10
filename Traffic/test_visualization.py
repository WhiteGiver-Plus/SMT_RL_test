import gym
import numpy as np
import matplotlib.pyplot as plt
from traffic_agent import TrafficAgent
from traffic_env import TrafficControlEnv
import time

def visualize_agent(env, agent, episodes=5, delay=0.5):
    """
    可视化agent的表现
    
    参数:
    env: 环境实例
    agent: 训练好的agent
    episodes: 测试轮数
    delay: 每步延迟时间(秒)
    """
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1} 开始")
        
        while not done:
            # 渲染环境
            env.render()
            time.sleep(delay)  # 添加延迟使可视化更容易观察
            
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            step += 1
            
            print(f"Step: {step}, Action: {action}, Reward: {reward:.2f}")
            
        print(f"Episode {episode + 1} 完成")
        print(f"总步数: {step}")
        print(f"总奖励: {total_reward:.2f}")
        
    env.close()

def plot_episode(rewards, episode_steps):
    """
    绘制单个episode的奖励和步数统计
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制奖励曲线
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Reward')
    
    # 绘制步数统计
    ax2.bar(['Total Steps'], [episode_steps])
    ax2.set_title('Episode Steps')
    ax2.set_ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 创建环境和agent
    env = TrafficControlEnv()
    state_size = 8  # 4个方向的等待车辆数量 + 4个方向的信号灯状态
    action_size = 4  # 4个可能的动作（给哪个方向绿灯）
    
    # 创建agent并加载训练好的模型
    agent = TrafficAgent(state_size, action_size)
    agent.load("traffic_agent.pth")
    
    # 设置agent为评估模式（关闭探索）
    agent.epsilon = 0
    
    # 可视化agent的表现
    visualize_agent(env, agent)
    
    # 收集一个完整episode的数据用于绘图
    state = env.reset()
    rewards = []
    done = False
    step = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        state = next_state
        step += 1
    
    # 绘制统计图
    plot_episode(rewards, step) 