import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import cvxpy as cp
H = 20
K = 50000
B_MAX = 20
E_MAX = 20
P_MAX = B_MAX + E_MAX
P_CONS = 15
c = np.log2(1 + P_MAX)
xi = 0.01
gamma = xi / 2
reward_cons_q = np.zeros(K)
vio_cons_q = np.zeros([K, H])
I = 1
eta = 2 * H * I / gamma


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
    #eta = np.power(K * H, 0.1)
    r = np.log2(1 + current_P)
    #if P_CONS - current_P < 0:
    #    return r - H
    #else:
    #    return r
    R = r + eta * np.min(np.min([P_CONS - current_P, 0]) + xi, 0)
    return R


def greedy():
    reward_greedy = 0
    current_B = 0
    current_E = int(E[0])
    for h in range(H):
        current_P = np.min([P_CONS, current_B+current_E])
        reward_greedy = reward_greedy + np.log2(1 + current_P)
        current_B = get_battery(current_B, current_E, current_P)
        current_E = int(E[h + 1])
    print('reward_greedy:', reward_greedy)
    return reward_greedy


def balance():
    mean_power = np.mean(E)
    reward_balance = 0
    current_B = 0
    current_E = int(E[0])
    for h in range(H):
        current_P = np.min([mean_power, P_CONS, current_B + current_E])
        reward_balance = reward_balance + np.log2(1 + current_P)
        current_B = get_battery(current_B, current_E, current_P)
        current_E = int(E[h + 1])
    print('reward_balance:', reward_balance)
    return reward_balance



def cons_q_learning():
    reward_cons_q = np.zeros([K])
    Q_SA = np.zeros([B_MAX + 1, E_MAX + 1, P_MAX + 1, H + 1], dtype=float)
    V_S = np.zeros([B_MAX + 1, E_MAX + 1, H + 1], dtype=float)
    N_T = np.zeros([B_MAX + 1, E_MAX + 1, P_MAX + 1, H + 1], dtype=int)
    for h in range(H):
        Q_SA[:, :, :, h] = eta * (H - h)
        V_S[:, :, h] = eta * (H - h)
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
    print(np.mean(reward_cons_q[20000:K]))
    return reward_cons_q


def convex_solver():
    P = cp.Variable(H)
    D = cp.Variable(H)

    objective = cp.Maximize(cp.sum(cp.log(1 + P))) / np.log(2)
    constrains = [0 <= P, P <= P_CONS,
                  0 <= E[0] - P[0] - D[0], E[0] - P[0] - D[0] <= B_MAX,
                  0 <= np.sum(E[:2]) - cp.sum(P[:2]) - cp.sum(D[:2]),
                  np.sum(E[:2]) - cp.sum(P[:2]) - cp.sum(D[:2]) <= B_MAX,
                  0 <= np.sum(E[:3]) - cp.sum(P[:3]) - cp.sum(D[:3]),
                  np.sum(E[:3]) - cp.sum(P[:3]) - cp.sum(D[:3]) <= B_MAX,
                  0 <= np.sum(E[:4]) - cp.sum(P[:4]) - cp.sum(D[:4]),
                  np.sum(E[:4]) - cp.sum(P[:4]) - cp.sum(D[:4]) <= B_MAX,
                  0 <= np.sum(E[:5]) - cp.sum(P[:5]) - cp.sum(D[:5]),
                  np.sum(E[:5]) - cp.sum(P[:5]) - cp.sum(D[:5]) <= B_MAX,
                  0 <= np.sum(E[:6]) - cp.sum(P[:6]) - cp.sum(D[:6]),
                  np.sum(E[:6]) - cp.sum(P[:6]) - cp.sum(D[:6]) <= B_MAX,
                  0 <= np.sum(E[:7]) - cp.sum(P[:7]) - cp.sum(D[:7]),
                  np.sum(E[:7]) - cp.sum(P[:7]) - cp.sum(D[:7]) <= B_MAX,
                  0 <= np.sum(E[:8]) - cp.sum(P[:8]) - cp.sum(D[:8]),
                  np.sum(E[:8]) - cp.sum(P[:8]) - cp.sum(D[:8]) <= B_MAX,
                  0 <= np.sum(E[:9]) - cp.sum(P[:9]) - cp.sum(D[:9]),
                  np.sum(E[:9]) - cp.sum(P[:9]) - cp.sum(D[:9]) <= B_MAX,
                  0 <= np.sum(E[:10]) - cp.sum(P[:10]) - cp.sum(D[:10]),
                  np.sum(E[:10]) - cp.sum(P[:10]) - cp.sum(D[:10]) <= B_MAX,
                  0 <= np.sum(E[:11]) - cp.sum(P[:11]) - cp.sum(D[:11]),
                  np.sum(E[:11]) - cp.sum(P[:11]) - cp.sum(D[:11]) <= B_MAX,
                  0 <= np.sum(E[:12]) - cp.sum(P[:12]) - cp.sum(D[:12]),
                  np.sum(E[:12]) - cp.sum(P[:12]) - cp.sum(D[:12]) <= B_MAX,
                  0 <= np.sum(E[:13]) - cp.sum(P[:13]) - cp.sum(D[:13]),
                  np.sum(E[:13]) - cp.sum(P[:13]) - cp.sum(D[:13]) <= B_MAX,
                  0 <= np.sum(E[:14]) - cp.sum(P[:14]) - cp.sum(D[:14]),
                  np.sum(E[:14]) - cp.sum(P[:14]) - cp.sum(D[:14]) <= B_MAX,
                  0 <= np.sum(E[:15]) - cp.sum(P[:15]) - cp.sum(D[:15]),
                  np.sum(E[:15]) - cp.sum(P[:15]) - cp.sum(D[:15]) <= B_MAX,
                  0 <= np.sum(E[:16]) - cp.sum(P[:16]) - cp.sum(D[:16]),
                  np.sum(E[:16]) - cp.sum(P[:16]) - cp.sum(D[:16]) <= B_MAX,
                  0 <= np.sum(E[:17]) - cp.sum(P[:17]) - cp.sum(D[:17]),
                  np.sum(E[:17]) - cp.sum(P[:17]) - cp.sum(D[:17]) <= B_MAX,
                  0 <= np.sum(E[:18]) - cp.sum(P[:18]) - cp.sum(D[:18]),
                  np.sum(E[:18]) - cp.sum(P[:18]) - cp.sum(D[:18]) <= B_MAX,
                  0 <= np.sum(E[:19]) - cp.sum(P[:19]) - cp.sum(D[:19]),
                  np.sum(E[:19]) - cp.sum(P[:19]) - cp.sum(D[:19]) <= B_MAX,
                  0 <= np.sum(E[:20]) - cp.sum(P[:20]) - cp.sum(D[:20]),
                  np.sum(E[:20]) - cp.sum(P[:20]) - cp.sum(D[:20]) <= B_MAX,
                  D >= 0]

    prob = cp.Problem(objective, constrains)
    result_cvx = prob.solve(solver='SCS')
    print('reward_cvx:', result_cvx)
    return result_cvx


test_num = 100
for mean in range(10, 13):
    reward_greedy = np.zeros(test_num)
    reward_balance = np.zeros(test_num)
    reward_solver = np.zeros(test_num)
    reward_cons_q = np.zeros([test_num, K])
    vio_cons_q = np.zeros([test_num, K])
    Energy = np.load('./data/energy_'+str(mean)+'.npy')
    for i in range(test_num):
        print('test_num', i + 1)

        E = Energy[i, :]
        reward_greedy[i] = greedy()
        reward_balance[i] = balance()
        reward_solver[i] = convex_solver()
        reward_cons_q[i, :] = cons_q_learning()

    np.save('reward_greedy'+str(mean)+'.npy', reward_greedy)
    np.save('reward_balance'+str(mean)+'.npy', reward_balance)
    np.save('reward_solver'+str(mean)+'.npy', reward_solver)
    np.save('reward_cons_q'+str(mean)+'.npy', reward_cons_q)



