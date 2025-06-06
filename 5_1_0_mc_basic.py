import sys
sys.path.append("..")
from src.grid_world import GridWorld # 假设 GridWorld 类在 src 目录下
import numpy as np
import matplotlib.pyplot as plt

def policy_evaluation(policy_matrix, env, gamma=0.9, num_episodes=10, episode_length=10):
    """
    策略评估：使用朴素蒙特卡洛方法计算给定策略下的状态-动作价值函数 Q_pi.
    对于每个 (s,a) 对，从状态 s 开始，执行动作 a，然后遵循策略 policy_matrix 产生 num_episodes 个 episodes.
    计算这些 episodes 的平均回报作为 Q_pi(s,a) 的估计.

    Args:
        policy_matrix: 当前策略, (num_states, num_actions) 数组.
                       policy_matrix[s, a] 是在状态s时选择动作a的概率.
        env: GridWorld 环境.
        gamma: 折扣因子.
        num_episodes: 每个状态-动作对进行采样的episode数量.
        episode_length: 每个episode的最大步数.

    Returns:
        action_values: 状态-动作价值函数 Q_pi, (num_states, num_actions) 数组.
    """
    n_states = env.num_states
    n_actions = len(env.action_space)
    
    action_values = np.zeros((n_states, n_actions))
    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))

    for s_idx in range(n_states):
        initial_x = s_idx % env.env_size[0]
        initial_y = s_idx // env.env_size[0]
        initial_state_tuple = (initial_x, initial_y)

        for a_idx in range(n_actions):
            first_action_to_take = env.action_space[a_idx]
            
            for _ in range(num_episodes):
                env.reset()
                env.agent_state = initial_state_tuple
                
                current_rewards = []
                
                # 第一步: 执行选定的 first_action_to_take
                current_env_state, reward, done, _ = env.step(first_action_to_take)
                current_rewards.append(reward)
                
                # 后续步骤: 遵循策略 policy_matrix
                for _ in range(1, episode_length): # episode_length 是总步数
                    
                    # 将 current_env_state (元组) 转换为 s_idx 以用于策略查找
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
                
                returns_sum[s_idx, a_idx] += G
                returns_count[s_idx, a_idx] += 1
                
    # 计算平均 Q 值
    for s in range(n_states):
        for a in range(n_actions):
            if returns_count[s, a] > 0:
                action_values[s, a] = returns_sum[s, a] / returns_count[s, a]
    
    return action_values

def policy_improvement(action_values, env):
    """
    策略改进：根据状态价值函数 V 贪婪地更新策略.

    Args:
        action_values: 状态-动作价值函数, (num_states, num_actions) 数组.
        env: GridWorld 环境.
    Returns:
        new_policy_matrix: 改进后的确定性策略, (num_states, num_actions) 数组.
    """
    new_policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    for s_idx in range(env.num_states):
        best_action_idx = np.argmax(action_values[s_idx])
        new_policy_matrix[s_idx, best_action_idx] = 1.0  # 确定性策略
    return new_policy_matrix

def policy_iteration_algorithm(env, gamma=0.9, theta=1e-6):
    """
    策略迭代算法主循环.

    Args:
        env: GridWorld 环境.
        gamma: 折扣因子.
        theta:策略评估收敛的阈值.

    Returns:
        policy: 最优策略.
        V: 最优价值函数.
    """
    # 1. 初始化策略 (例如，所有状态下都选择第一个动作)
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    for s_idx in range(env.num_states):
        policy_matrix[s_idx, 0] = 1.0  # 初始时，对每个状态选择第一个动作

    iteration = 0
    while True:
        iteration += 1
        print(f"Policy Iteration - Iteration: {iteration}")
        
        action_values = policy_evaluation(policy_matrix, env, gamma)

        new_policy_matrix = policy_improvement(action_values, env)
        
        # 检查策略是否稳定
        if np.array_equal(new_policy_matrix, policy_matrix):
            print(f"Policy converged after {iteration} iterations.")
            break # 策略已稳定，找到最优策略
        
        policy_matrix = new_policy_matrix
        
        # V*(s) = max_a Q*(s,a)
        # action_values at this stage are Q_pi for the new_policy_matrix (which is now policy_matrix)
        # If the policy has converged, these action_values correspond to the optimal policy.
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

    optimal_policy, optimal_V = policy_iteration_algorithm(env, gamma=0.9, theta=1e-6)
    # 重置环境并可视化
    env.reset() 
    show_policy(env, optimal_V, optimal_policy)
    
    # 模拟路径
    simulate(env, optimal_policy, max_steps=50)

    print("Algorithm finished. Close the plot window to exit.")
    plt.ioff() 
    plt.show()
