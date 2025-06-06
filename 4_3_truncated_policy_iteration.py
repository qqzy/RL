import sys
sys.path.append("..")
from src.grid_world import GridWorld # 假设 GridWorld 类在 src 目录下
import numpy as np
import matplotlib.pyplot as plt

def policy_evaluation(policy_matrix, env, gamma=0.9, max_sweeps=5):
    """
    策略评估：计算给定策略下的状态价值函数 V_pi.
    如果提供了 max_sweeps，则最多执行 max_sweeps 次迭代.

    Args:
        policy_matrix: 当前策略, (num_states, num_actions) 数组.
                       policy_matrix[s, a] 是在状态s时选择动作a的概率.
                       对于确定性策略，一个动作的概率为1，其余为0.
        env: GridWorld 环境.
        gamma: 折扣因子.
        max_sweeps: 策略评估的最大迭代次数 (用于截断策略评估).

    Returns:
        V: 该策略下的状态价值函数, (num_states) 数组.
    """
    V = np.zeros(env.num_states)  # 初始化值函数为0
    sweeps_done = 0
    while True:
        for s_idx in range(env.num_states):
            x = s_idx % env.env_size[0]
            y = s_idx // env.env_size[0]
            current_state = (x, y)

            new_v_s = 0
            
            # 根据当前策略计算状态价值
            # 对于确定性策略，policy_matrix[s_idx, action_idx] 对于选定的动作是1，其他是0
            for action_idx, action in enumerate(env.action_space):
                if policy_matrix[s_idx, action_idx] > 0: # 如果策略选择这个动作
                    next_state, reward = env._get_next_state_and_reward(current_state, action)
                    next_state_idx = next_state[0] + next_state[1] * env.env_size[0]
                    # V(s) = sum_a pi(a|s) * (R + gamma * V(s'))
                    new_v_s += policy_matrix[s_idx, action_idx] * (reward + gamma * V[next_state_idx])
            
            V[s_idx] = new_v_s
        
        sweeps_done += 1
        
        if sweeps_done >= max_sweeps:
            break

            
    return V

def policy_improvement(V, env, gamma=0.9):
    """
    策略改进：根据状态价值函数 V 贪婪地更新策略.

    Args:
        V: 当前的状态价值函数, (num_states) 数组.
        env: GridWorld 环境.
        gamma: 折扣因子.

    Returns:
        new_policy_matrix: 改进后的确定性策略, (num_states, num_actions) 数组.
    """
    new_policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    for s_idx in range(env.num_states):
        x = s_idx % env.env_size[0]
        y = s_idx // env.env_size[0]
        current_state = (x, y)


        action_values = []
        for action_idx, action in enumerate(env.action_space):
            next_state, reward = env._get_next_state_and_reward(current_state, action)
            next_state_idx = next_state[0] + next_state[1] * env.env_size[0]
            action_values.append(reward + gamma * V[next_state_idx])

        best_action_idx = np.argmax(action_values)
        
        new_policy_matrix[s_idx, best_action_idx] = 1.0  # 确定性策略
    return new_policy_matrix

def truncated_policy_iteration_algorithm(env, gamma=0.9, k_sweeps_eval=5):
    """
    截断策略迭代算法主循环.

    Args:
        env: GridWorld 环境.
        gamma: 折扣因子.
        k_sweeps_eval: 每次策略评估阶段执行的迭代次数.

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
        print(f"Truncated Policy Iteration - Iteration: {iteration}, Eval Sweeps: {k_sweeps_eval}")
        
        # 策略评估 (截断)
        V = policy_evaluation(policy_matrix, env, gamma, max_sweeps=k_sweeps_eval)

        # 策略改进
        new_policy_matrix = policy_improvement(V, env, gamma)
        
        # 检查策略是否稳定
        if np.array_equal(new_policy_matrix, policy_matrix):
            print(f"Policy converged after {iteration} iterations.")
            break # 策略已稳定，找到最优策略
        
        policy_matrix = new_policy_matrix
        
            
    return policy_matrix, V

def show_policy(env, V, policy_matrix, filename="policy_iteration_visualization.png"):
    """
    可视化价值函数和策略.
    """
    env.render(animation_interval=0.01) 
    env.add_state_values(V)
    env.add_policy(policy_matrix)
    plt.draw() 
    plt.savefig(filename, dpi=300, bbox_inches='tight')



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


    optimal_policy, optimal_V = truncated_policy_iteration_algorithm(env, gamma=0.9,  k_sweeps_eval=3)
    
    # 重置环境并可视化
    env.reset() 
    show_policy(env, optimal_V, optimal_policy, filename="truncated_policy_iteration_visualization.png")
    
    # 模拟路径
    simulate(env, optimal_policy, max_steps=50)

    print("Algorithm finished. Close the plot window to exit.")
    plt.ioff() 
    plt.show()
