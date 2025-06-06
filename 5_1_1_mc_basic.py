import sys
sys.path.append("..")
from src.grid_world import GridWorld # 假设 GridWorld 类在 src 目录下
import numpy as np
import matplotlib.pyplot as plt

def policy_iteration_algorithm(env, gamma=0.9, num_episodes=10, episode_length=10):
    """
    改进的策略迭代算法，每次评估完一个状态的价值后立即更新该状态的策略.
    这种方法被称为"广义策略迭代"的一种形式，它在评估-改进周期中交替进行，
    但不是等待完全的策略评估，而是在单次状态遍历后就进行策略更新，
    这通常能加速收敛。

    Args:
        env: GridWorld 环境.
        gamma: 折扣因子.
        num_episodes: 每个状态-动作对进行采样的episode数量.
        episode_length: 每个episode的最大步数.

    Returns:
        policy: 最优策略.
        V: 最优价值函数.
    """
    n_states = env.num_states
    n_actions = len(env.action_space)
    
    # 1. 初始化策略和Q值
    policy_matrix = np.zeros((n_states, n_actions))
    for s_idx in range(n_states):
        policy_matrix[s_idx, 0] = 1.0  # 初始时，对每个状态选择第一个动作
    
    action_values = np.zeros((n_states, n_actions))

    iteration = 0
    while True:
        iteration += 1
        print(f"Policy Iteration - Iteration: {iteration}")
        policy_stable = True

        # 遍历每个状态
        for s_idx in range(n_states):
            old_action_idx = np.argmax(policy_matrix[s_idx])
            
            initial_x = s_idx % env.env_size[0]
            initial_y = s_idx // env.env_size[0]
            initial_state_tuple = (initial_x, initial_y)

            # 为状态 s 评估所有动作的价值 Q(s,a)
            for a_idx in range(n_actions):
                first_action_to_take = env.action_space[a_idx]
                
                returns_sum_sa = 0
                for _ in range(num_episodes):
                    env.reset()
                    env.agent_state = initial_state_tuple
                    
                    current_rewards = []
                    
                    # 第一步: 执行选定的 first_action_to_take
                    current_env_state, reward, done, _ = env.step(first_action_to_take)
                    current_rewards.append(reward)
                    
                    # 后续步骤: 遵循当前策略 policy_matrix
                    for _ in range(1, episode_length):
                        if done:
                            break
                        
                        current_s_idx_for_policy = current_env_state[0] + current_env_state[1] * env.env_size[0]

                        action_probs = policy_matrix[current_s_idx_for_policy]
                        action_from_policy_idx = np.random.choice(n_actions, p=action_probs)
                        action_from_policy = env.action_space[action_from_policy_idx]
                        
                        next_env_state, reward, done, _ = env.step(action_from_policy)
                        current_rewards.append(reward)
                        current_env_state = next_env_state
                    
                    # 计算此 episode 的折扣回报 G
                    G = 0
                    for r_val in reversed(current_rewards):
                        G = r_val + gamma * G
                    
                    returns_sum_sa += G
                
                action_values[s_idx, a_idx] = returns_sum_sa / num_episodes
            
            # 策略改进: 为状态 s 找到最佳动作并立即更新策略
            best_action_idx = np.argmax(action_values[s_idx])
            
            if old_action_idx != best_action_idx:
                policy_stable = False
            
            # 更新策略
            policy_matrix[s_idx, :] = 0.0
            policy_matrix[s_idx, best_action_idx] = 1.0

        # 检查策略是否在整个状态空间上稳定
        if policy_stable:
            print(f"Policy converged after {iteration} iterations.")
            break
    
    # V*(s) = max_a Q*(s,a)
    V = np.max(action_values, axis=1)
    return policy_matrix, V

def show_policy(env, V, policy_matrix):
    """
    可视化价值函数和策略. (复用自 value_iteration)
    """
    env.render(animation_interval=0.01) 
    env.add_state_values(V)
    env.add_policy(policy_matrix)
    plt.draw() 
    plt.savefig("mc_basic_visualization.png", dpi=300, bbox_inches='tight')



def simulate(env, policy_matrix, max_steps=100):
    """
    从起点开始，根据策略模拟代理的路径. (复用自 value_iteration)
    """
    print("Starting simulation with the learned policy...")
    state, _ = env.reset() 
    env.render() 

    for step in range(max_steps):
        x, y = state
        s_idx = x + y * env.env_size[0]
        
        action_idx = 0 # 默认动作
        if np.sum(policy_matrix[s_idx]) > 0: 
            action_idx = np.argmax(policy_matrix[s_idx]) # 确定性策略：选择概率最高的动作
        
        action = env.action_space[action_idx]

        next_state, reward, done, _ = env.step(action)
        env.render() 
        
        state = next_state
        
        if done:
            print(f"Target reached in {step + 1} steps!")
            break
    else: 
        print(f"Simulation ended after {max_steps} steps without reaching the target.")

if __name__ == '__main__':
    env = GridWorld() # 使用默认参数初始化

    optimal_policy, optimal_V = policy_iteration_algorithm(env, gamma=0.9)
    # 重置环境并可视化
    env.reset() 
    show_policy(env, optimal_V, optimal_policy)
    
    # 模拟路径
    simulate(env, optimal_policy, max_steps=50)

    print("Algorithm finished. Close the plot window to exit.")
    plt.ioff() 
    plt.show()
