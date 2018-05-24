"""
cart pole
action -> Q learningで学習
"""

import gym
import numpy as np
import random

def get_action(state):
    """
    actionを求める関数
    """
    epsilon = 0.01
    if epsilon < np.random.uniform(0, 1):
        next_action = np.argmax(q_table[state])
    else:
        next_action = np.random.randint(2)
    return next_action

def update_Qtable(q_table, state, action, reward, next_state):
    """
    Q-table(価値関数)の更新
    """
    gamma = 0.9
    alpha = 0.3

    q_table[state][action] = (1 - alpha) * q_table[state][action] + \
                             alpha * (reward + gamma * max(q_table[next_state]))
    return q_table

def convert_to_bin(type: str, value):
    """
    observationを離散値に変換
    """
    BINS = {
        'cart_position': np.linspace(-2.4, 2.4, 8),
        'cart_velocity': np.linspace(-2, 2, 8),
        'pole_angle': np.linspace(-0.4, 0.4, 8),
        'pole_velocity': np.linspace(-3.5, 3.5, 8)
    }
    return np.digitize(value, BINS[type])

def transform_state(observation):
    """
    観察したobservationを離散値のstateに変換し、結合する
    """
    cart_pos, cart_vel, pole_angle, pole_vel = observation
    state = '{cart_pos_bin}{cart_vel_bin}{pole_angle_bin}{pole_vel_bin}'\
        .format(cart_pos_bin=convert_to_bin('cart_position', cart_pos),
                cart_vel_bin=convert_to_bin('cart_velocity', cart_vel),
                pole_angle_bin=convert_to_bin('pole_angle', pole_angle),
                pole_vel_bin=convert_to_bin('pole_velocity', pole_vel))
    return int(state, 9)


env = gym.make('CartPole-v0')
# size 1つの状態を5分割にしたものが4通り -> 5の4乗通り * action数
# q_table = np.random.uniform(low=0, high=1, size=(5 ** 4, env.action_space.n))
q_table = [[random.uniform(0, 1), random.uniform(0, 1)] for i in range(9 ** 4) for j in range(2)]
max_number_step = 100  # 1試行のstep数
num_episodes = 10000  # 総試行回数
count_max = 0

for episode in range(num_episodes):
    # 環境の初期化
    observation = env.reset()
    count = 0

    # レンダリング
    env.render()

    # actionの決定
    state = transform_state(observation)
    action = np.argmax(q_table[state])

    for t in range(max_number_step):
        count += 1
        # 行動の試行とフィードバックの取得
        observation, reward, done, _ = env.step(action)

        if done:
            print('episode{0}-complete-reward{1}-count{2}-max{3}'.format(episode, action, count, count_max))

        # q table の更新
        next_state = transform_state(observation)
        q_table = update_Qtable(q_table, state, action, reward, next_state)

        # 次の行動を求め、次の状態に遷移する
        action = get_action(next_state)
        state = next_state

        if done:
            if count_max < count:
                count_max = count
            env.close()
            break
