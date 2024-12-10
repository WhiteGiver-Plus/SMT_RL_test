import gym
import numpy as np
from traffic_agent import TrafficAgent
import matplotlib.pyplot as plt
from traffic_env import TrafficControlEnv

def train_agent(env, agent, episodes=1000):
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 训练网络
            agent.replay()
            
            state = next_state
            total_reward += reward
            
        # 每10个episode更新目标网络
        if episode % 10 == 0:
            agent.update_target_network()
            
        rewards_history.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
    return rewards_history

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title('Training Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == "__main__":
    # 创建环境和agent
    env = TrafficControlEnv()
    state_size = 8  # 4个方向的等待车辆数量 + 4个方向的信号灯状态
    action_size = 4  # 4个可能的动作（给哪个方向绿灯）
    
    agent = TrafficAgent(state_size, action_size)
    
    # 训练agent
    rewards = train_agent(env, agent)
    
    # 保存训练好的模型
    agent.save("traffic_agent.pth")
    
    # 绘制训练过程中的奖励
    plot_rewards(rewards) 