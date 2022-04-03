import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
import multiprocessing
from tqdm.contrib.concurrent import process_map
matplotlib.use('Agg')

H = 20
K = 20000
B_MAX = 20
E_MAX = 20
P_MAX = B_MAX + E_MAX
P_CONS = 8

xi = 0.1
gamma = xi / 2
I = 1
eta = 2 * H / gamma

test_num = 1000

def get_best_action(Q_SA, current_B, current_E, h):
    best_action = 0
    for a in range(current_B + current_E + 1):
        if Q_SA[current_B, current_E, a, h] > Q_SA[current_B, current_E, best_action, h]:
            best_action = a
    return best_action


def get_battery(current_B, current_E, current_P):
    next_B = np.min([current_B + current_E - current_P, B_MAX])
    return next_B


def get_lr(t):
    return (H + 1) / (H + t)


def get_reward(current_P):
    r = np.log2(1 + current_P)
    R = r + eta * np.min(np.min([P_CONS - current_P, 0]) + xi, 0)
    return R


def cons_q_learning(E):
    Q_SA = np.zeros([B_MAX + 1, E_MAX + 1, P_MAX + 1, H + 1], dtype=float)
    V_S = np.zeros([B_MAX + 1, E_MAX + 1, H + 1], dtype=float)
    N_T = np.zeros([B_MAX + 1, E_MAX + 1, P_MAX + 1, H + 1], dtype=int)
    reward_cons_q = np.zeros(K)
    vio_cons_q = np.zeros(K)
    for h in range(H):
        Q_SA[:, :, :, h] = eta * (H - h)
        V_S[:, :, h] = eta * (H - h)
    #algorithm: Constrained Q_Learning
    for k in range(K):
        current_B = 0
        current_E = int(E[0])
        for h in range(H):
            current_P = get_best_action(Q_SA, current_B, current_E, h)
            next_E = int(E[h + 1])
            next_B = get_battery(current_B, current_E, current_P)
            N_T[current_B, current_E, current_P, h] = N_T[current_B, current_E, current_P, h] + 1
            t = N_T[current_B, current_E, current_P, h]
            alpha_t = get_lr(t)
            Q_SA[current_B, current_E, current_P, h] = (1 - alpha_t) * Q_SA[current_B, current_E, current_P, h]\
                                                       + alpha_t * (get_reward(current_P) + V_S[next_B, next_E, h + 1])
            V_S[current_B, current_E, h] = np.min([eta * H, Q_SA[current_B, current_E, get_best_action(Q_SA, current_B, current_E, h), h]])
            current_E = next_E
            current_B = next_B
            reward_cons_q[k] += np.log2(1 + current_P)
            vio_cons_q[k] += np.abs(np.min([P_CONS - current_P, 0]))
    #get the result for \bar{\pi}
    for i in range(1, K):
        reward_cons_q[i] = reward_cons_q[i] + reward_cons_q[i-1]
        vio_cons_q[i] = vio_cons_q[i] + vio_cons_q[i-1]

    reward_cons_q = reward_cons_q / np.arange(start=1, stop=K + 1)
    vio_cons_q = vio_cons_q / np.arange(start=1, stop=K + 1)
    return reward_cons_q, vio_cons_q


def result_plot(reward_cons_q, vio_cons_q):
    reward_mean = np.mean(reward_cons_q, axis=0)
    reward_std = np.std(reward_cons_q, axis=0)

    reward_min = reward_mean - reward_std
    reward_max = reward_mean + reward_std

    vio_mean = np.mean(vio_cons_q, axis=0)
    vio_std = np.std(vio_cons_q, axis=0)

    vio_min = vio_mean - vio_std
    vio_max = vio_mean + vio_std

    episode = np.arange(start=0, stop=K, step=10)
    episode2 = np.arange(start=0, stop=K, step=1)
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    lns1 = ax.plot(episode, reward_mean[episode], color='b', label='Total reward', linewidth=1)
    ax.fill_between(episode2, reward_min[episode2], reward_max[episode2], color='b', alpha=0.3)
    ax2 = ax.twinx()
    lns2 = ax2.plot(episode, vio_mean[episode], color='r', label='Violation', linewidth=1)
    ax2.fill_between(episode2, vio_min[episode2], vio_max[episode2], color='r', alpha=0.3)
    ax.grid()
    ax.set_xlabel("Episode k")
    ax.set_ylabel("Sum Transmission rate")
    ax2.set_ylabel("Violation in each episode k")
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='center right')
    tikzplotlib.save('cons_q0001.tex')
    plt.savefig('cons_q0001.png', dpi=600)
    #plt.show()


if __name__ == '__main__':
    energy = np.load('./data/energy.npy')
    energy_itr = []
    result_reward = np.zeros([test_num, K])
    result_vio = np.zeros([test_num, K])
    for i in range(test_num):
        energy_itr.append(energy[i, :])
    print('Start Parallelization')
    pool = multiprocessing.Pool()
    print('Parallel Learning')
    result = process_map(cons_q_learning, energy_itr)
    print('Unpack Result')
    for i in range(test_num):
        result_reward[i, :] = result[i][0]
        result_vio[i, :] = result[i][1]
    np.save('reward01.npy',result_reward)
    np.save('vio01.npy',result_vio)
    print('Plot Result')
    #result_plot(result_reward, result_vio)
    print('Finish')
