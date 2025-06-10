import sys
sys.path.append("..")
from src.grid_world import GridWorld # 假设 GridWorld 类在 src 目录下
import numpy as np
import matplotlib.pyplot as plt

def sarsa_algorithm(env, gamma=0.9, num_episodes=100000, epsilon=0.2, alpha=0.1):
    """
    使用 SARSA 算法学习最优策略
    
    Args:
        env: GridWorld 环境
        gamma: 折扣因子
        num_episodes: 训练的总 episode 数量
        epsilon: ε-greedy 策略中的探索率
        alpha: 学习率
        
    Returns:
        policy: 最优策略矩阵 (n_states, n_actions)
        V: 最优状态价值函数 (n_states,)
    """
    n_states = env.num_states
    n_actions = len(env.action_space)
    
    # 初始化 Q 函数
    Q = np.zeros((n_states, n_actions))
    
    # 初始化策略矩阵 (ε-greedy)
    policy = np.ones((n_states, n_actions)) * epsilon / n_actions
    for s in range(n_states):
        policy[s, 0] += 1 - epsilon
    
    # SARSA 主循环
    for episode in range(1, num_episodes + 1):
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes}")
        
        # 初始化状态
        env.reset()
        state = env.agent_state
        s_idx = state[0] + state[1] * env.env_size[0]
        
        # 使用策略矩阵选择初始动作
        a_idx = np.random.choice(n_actions, p=policy[s_idx])
        
        done = False
        while not done:
            # 执行动作
            action = env.action_space[a_idx]
            next_state, reward, done, _ = env.step(action)
            next_s_idx = next_state[0] + next_state[1] * env.env_size[0]
            
            # 使用策略矩阵选择下一个动作
            next_a_idx = np.random.choice(n_actions, p=policy[next_s_idx])
            
            # SARSA 更新
            td_target = reward + gamma * Q[next_s_idx][next_a_idx]
            td_error = td_target - Q[s_idx][a_idx]
            Q[s_idx][a_idx] += alpha * td_error
            
            # 更新策略 (ε-greedy)
            best_action = np.argmax(Q[s_idx])
            policy[s_idx] = epsilon / n_actions
            policy[s_idx, best_action] += 1 - epsilon
            
            # 转移到下一个状态和动作
            s_idx = next_s_idx
            a_idx = next_a_idx
            state = next_state
    
    # 计算状态价值函数
    V = np.max(Q, axis=1)
    
    return policy, V

def show_policy(env, V, policy_matrix):
    """
    可视化价值函数和策略. (复用自 value_iteration)
    """
    env.render(animation_interval=0.01) 
    env.add_state_values(V)
    env.add_policy(policy_matrix)
    plt.draw() 
    plt.savefig("sarsa_visualization.png", dpi=300, bbox_inches='tight')



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
        
        # 根据策略选择动作 (ε-greedy)
        if np.random.rand() < policy_matrix[s_idx].max():  # 利用概率
            action_idx = np.argmax(policy_matrix[s_idx])
        else:
            action_idx = np.random.choice(len(env.action_space))
        
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

    # 使用 SARSA 算法
    optimal_policy, optimal_V = sarsa_algorithm(
        env, 
        gamma=0.9, 
        num_episodes=10000,
        epsilon=0.1,
        alpha=0.1  # 学习率
    )
    
    # 重置环境并可视化
    env.reset() 
    show_policy(env, optimal_V, optimal_policy)
    
    # 模拟路径
    simulate(env, optimal_policy, max_steps=50)

    print("Algorithm finished. Close the plot window to exit.")
    plt.ioff() 
    plt.show()
