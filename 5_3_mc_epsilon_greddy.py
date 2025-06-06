import sys
sys.path.append("..")
from src.grid_world import GridWorld # 假设 GridWorld 类在 src 目录下
import numpy as np
import matplotlib.pyplot as plt

def policy_iteration_algorithm(env, gamma=0.9, num_episodes=8000, episode_length=1000, epsilon=0.1):
    """
    使用 ε-greedy 策略的蒙特卡洛控制算法 (Every-visit MC).
    对每一个 episode, 从一个随机的(状态, 动作)对开始, 然后遵循当前策略。
    episode 结束后，从后向前遍历，更新所有访问过的(状态, 动作)对的价值，并使用 ε-greedy 策略改进。

    Args:
        env: GridWorld 环境.
        gamma: 折扣因子.
        num_episodes: 每个状态-动作对进行采样的episode数量.
        episode_length: 每个episode的最大步数.
        epsilon: 探索概率，控制随机选择动作的概率。

    Returns:
        policy: 最优策略.
        V: 最优价值函数.
    """
    n_states = env.num_states
    n_actions = len(env.action_space)

    # 1. 初始化
    # 初始策略为均匀随机策略
    policy_matrix = np.ones((n_states, n_actions)) / n_actions

    action_values = np.zeros((n_states, n_actions))
    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))

    # 主循环
    for episode_num in range(1, num_episodes + 1):
        if episode_num % 2000 == 0:
            print(f"Episode {episode_num}/{num_episodes}")

        # 1. Exploring Start: 随机选择一个 (s, a) 对作为起点
        # 确保所有状态动作对都有可能被选为起点
        s_idx = np.random.randint(n_states)
        a_idx = np.random.randint(n_actions)
        
        initial_x = s_idx % env.env_size[0]
        initial_y = s_idx // env.env_size[0]
        initial_state_tuple = (initial_x, initial_y)

        # 2. 生成一个 episode
        env.reset()
        env.agent_state = initial_state_tuple
        
        episode_history = []
        
        # 第一步: 执行选定的 (s, a)
        first_action_to_take = env.action_space[a_idx]
        next_env_state, reward, done, _ = env.step(first_action_to_take)
        episode_history.append((s_idx, a_idx, reward))
        current_env_state = next_env_state

        # 后续步骤: 遵循当前策略 policy_matrix
        for _ in range(episode_length - 1):
            if done:
                break

            current_s_idx = current_env_state[0] + current_env_state[1] * env.env_size[0]

            # 直接从策略矩阵中采样动作
            action_from_policy_idx = np.random.choice(n_actions, p=policy_matrix[current_s_idx])
            
            action_from_policy = env.action_space[action_from_policy_idx]
            
            next_env_state, reward, done, _ = env.step(action_from_policy)
            episode_history.append((current_s_idx, action_from_policy_idx, reward))
            current_env_state = next_env_state

        # 3. 策略评估和改进 (Every-visit MC)
        G = 0
        
        # 从后往前遍历 episode
        for s_t, a_t, r in reversed(episode_history):
            G = r + gamma * G
            
            # 更新 Q 值 (策略评估)
            returns_sum[s_t, a_t] += G
            returns_count[s_t, a_t] += 1
            action_values[s_t, a_t] = returns_sum[s_t, a_t] / returns_count[s_t, a_t]
            
            # 更新策略 (ε-greedy 策略改进)
            best_a_idx = np.argmax(action_values[s_t])
            policy_matrix[s_t, :] = epsilon / n_actions  # 探索概率均匀分配
            policy_matrix[s_t, best_a_idx] += 1 - epsilon  # 利用概率集中在最优动作

    print("Finished training.")
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

    optimal_policy, optimal_V = policy_iteration_algorithm(env, gamma=0.9, epsilon=0.1)
    # 重置环境并可视化
    env.reset() 
    show_policy(env, optimal_V, optimal_policy)
    
    # 模拟路径
    simulate(env, optimal_policy, max_steps=50)

    print("Algorithm finished. Close the plot window to exit.")
    plt.ioff() 
    plt.show()
