import sys
sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np

def value_iteration(env, gamma=0.9, theta=1e-6):
    """
    值迭代算法

    Args:
        env: GridWorld 环境.
        gamma: 折扣因子.
        theta: 判断收敛的阈值.

    Returns:
        V: 最优值函数.
    """
    V = np.random.rand(env.num_states)  # 初始化值函数
    while True:
        delta = 0
        for s_idx in range(env.num_states):
            x = s_idx % env.env_size[0]
            y = s_idx // env.env_size[0]
            current_state = (x, y)

            v_old = V[s_idx]
            action_values = []
            for action_idx, action in enumerate(env.action_space):
                next_state, reward = env._get_next_state_and_reward(current_state, action)
                next_state_idx = next_state[0] + next_state[1] * env.env_size[0]
                action_values.append(reward + gamma * V[next_state_idx])
            
            if not action_values: # 如果没有可选动作（例如在目标或禁止区域，虽然已经跳过）
                 V[s_idx] = 0 # 或者根据具体情况处理
            else:
                 V[s_idx] = max(action_values)
            delta = max(delta, abs(v_old - V[s_idx]))
        
        if delta < theta:
            break
    return V

def extract_policy(env, V, gamma=0.9):
    """
    根据值函数提取最优策略.

    Args:
        env: GridWorld 环境.
        V: 最优值函数.
        gamma: 折扣因子.

    Returns:
        policy: 最优策略, 一个 (num_states, num_actions) 的数组.
                policy[s, a] 是在状态s时选择动作a的概率 (这里是确定性策略).
    """
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    for s_idx in range(env.num_states):
        x = s_idx % env.env_size[0]
        y = s_idx // env.env_size[0]
        current_state = (x, y)


        action_values = []
        for action in env.action_space:
            next_state, reward = env._get_next_state_and_reward(current_state, action)
            next_state_idx = next_state[0] + next_state[1] * env.env_size[0]
            action_values.append(reward + gamma * V[next_state_idx])
        

        best_action_idx = np.argmax(action_values)
        policy_matrix[s_idx, best_action_idx] = 1.0  # 确定性策略

    return policy_matrix

def show_policy(env, V, policy_matrix):
    """
    可视化价值函数和策略.
    """
    env.render(animation_interval=0.01) # 确保绘图已初始化
    env.add_state_values(V)
    env.add_policy(policy_matrix)
    import matplotlib.pyplot as plt # 确保plt在此函数作用域内可用
    plt.draw() # 更新显示
    plt.savefig("value_iteration_visualization.png", dpi=300, bbox_inches='tight')


def simulate(env, policy_matrix, max_steps=100):
    """
    从起点开始，根据策略模拟代理的路径.
    """
    print("\nStarting simulation...")
    state, _ = env.reset() # 重置环境到初始状态并清空轨迹
    env.render() # 显示初始状态

    for step in range(max_steps):
        x, y = state
        s_idx = x + y * env.env_size[0]
        
        # 确定性策略：选择概率最高的动作
        if np.sum(policy_matrix[s_idx]) > 0: # 确保该状态有定义的策略
            action_idx = np.argmax(policy_matrix[s_idx])
            action = env.action_space[action_idx]
        else:
            action = env.action_space[0]

        next_state, reward, done, _ = env.step(action)
        env.render() # 渲染每一步
        
        state = next_state
        
        if done:
            print(f"Target reached in {step + 1} steps!")
            break
    else: # for-else 循环，仅当循环未被break时执行
        print(f"Simulation ended after {max_steps} steps without reaching the target.")

if __name__ == '__main__':

    env = GridWorld() # 使用默认参数初始化


    optimal_V = value_iteration(env, gamma=0.9, theta=1e-6)
    optimal_policy = extract_policy(env, optimal_V, gamma=0.99)

    env.reset() # 在第一次渲染前初始化环境状态和轨迹
    # 可视化最终的价值和策略
    show_policy(env, optimal_V, optimal_policy)
    
    # 模拟从起点开始的路径
    simulate(env, optimal_policy, max_steps=100) # 为了快速演示，步数减少到100

    import matplotlib.pyplot as plt
    print("\nSimulation finished. Close the plot window to exit.")
    plt.ioff() # 关闭交互模式，这样窗口会保持打开直到手动关闭
    plt.show()
