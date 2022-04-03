import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
import multiprocessing
from tqdm.contrib.concurrent import process_map
K = 20000
gamma = ['0.1', '0.01', '0.001']
style = ['solid', 'dotted', 'dashed']
def result_plot(reward_cons_q, vio_cons_q):
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    ax2 = ax.twinx()
    lns1 = []
    lns2 = []
    for i in range(3):
        reward_mean = np.mean(reward_cons_q[i], axis=0)
        reward_std = np.std(reward_cons_q[i], axis=0)

        reward_min = reward_mean - reward_std
        reward_max = reward_mean + reward_std

        vio_mean = np.mean(vio_cons_q[i], axis=0)
        vio_std = np.std(vio_cons_q[i], axis=0)

        vio_min = vio_mean - vio_std
        vio_max = vio_mean + vio_std

        episode = np.arange(start=0, stop=K, step=10)
        episode2 = np.arange(start=0, stop=K, step=1)

        lns1.append(ax.plot(episode, reward_mean[episode], color='b', label='Total reward:gamma='+gamma[i], linestyle=style[i], linewidth=1))
        ax.fill_between(episode2, reward_min[episode2], reward_max[episode2], color='b', alpha=0.3 - 0.1 * i)

        lns2.append(ax2.plot(episode, vio_mean[episode], color='r', label='Violation'+gamma[i], linestyle=style[i], linewidth=1))
        ax2.fill_between(episode2, vio_min[episode2], vio_max[episode2], color='r', alpha=0.3 - 0.1 * i)
    ax.grid()
    ax.set_xlabel("Episode k")
    ax.set_ylabel("Sum Transmission rate")
    ax2.set_ylabel("Violation in each episode k")
    lns = []
    for l in lns1:
        lns.append(l)
    for l in lns2:
        lns.append(l)
    labs = []
    handles = []
    for i in range(6):
        labs.append(lns[i][0].get_label())
        handles.append(lns[i][0])
    ax.legend(handles, labs, loc='center right')
    #tikzplotlib.save('cons_q0001.tex')
    plt.savefig('cons_q0001.png', dpi=600)
    plt.show()

reward_cons_q= []
reward_cons_q.append(np.load('reward01.npy'))
reward_cons_q.append(np.load('reward001.npy'))
reward_cons_q.append(np.load('reward0001.npy'))
vio_cons_q= []
vio_cons_q.append(np.load('vio01.npy'))
vio_cons_q.append(np.load('vio001.npy'))
vio_cons_q.append(np.load('vio0001.npy'))
result_plot(reward_cons_q, vio_cons_q)