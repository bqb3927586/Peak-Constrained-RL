<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

H = 20
K = 50000

test_num = 5
reward_greedy = np.zeros(test_num)
reward_balance = np.zeros(test_num)
reward_solver = np.zeros(test_num)
reward_cons_q = np.zeros(test_num)
for i in range(test_num):
    mean = 8 + i
    reward_greedy[i] = np.mean(np.load('reward_greedy'+str(mean)+'.npy'))
    reward_balance[i] = np.mean(np.load('reward_balance'+str(mean)+'.npy'))
    reward_solver[i] = np.mean(np.load('reward_solver'+str(mean)+'.npy'))
    temp = np.load('reward_cons_q'+str(mean)+'.npy')
    temp = np.mean(temp[:, int(0.5 * K):K], axis=1)
    reward_cons_q[i] = np.mean(temp)

mean = [8, 9, 10, 11, 12]
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.plot(mean, reward_greedy, color='r', label='greedy', linewidth=1, marker='>')
ax.plot(mean, reward_balance, color='g', label='balance', linewidth=1, marker='s')
ax.plot(mean, reward_solver, color='k', label='cvx', linewidth=1, marker='*')
ax.plot(mean, reward_cons_q, color='b', label='consQ', linewidth=1, marker='|')

ax.grid()
ax.legend(loc=0)
ax.set_xlabel("Mean of the energy distribution")
ax.set_ylabel("sum transmission rate")
plt.savefig('compare.png', dpi=600)
#tikzplotlib.save('compare.tex')
plt.show()
=======
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

H = 20
K = 50000

test_num = 5
reward_greedy = np.zeros(test_num)
reward_balance = np.zeros(test_num)
reward_solver = np.zeros(test_num)
reward_cons_q = np.zeros(test_num)
for i in range(test_num):
    mean = 8 + i
    reward_greedy[i] = np.mean(np.load('reward_greedy'+str(mean)+'.npy'))
    reward_balance[i] = np.mean(np.load('reward_balance'+str(mean)+'.npy'))
    reward_solver[i] = np.mean(np.load('reward_solver'+str(mean)+'.npy'))
    temp = np.load('reward_cons_q'+str(mean)+'.npy')
    temp = np.mean(temp[:, int(0.5 * K):K], axis=1)
    reward_cons_q[i] = np.mean(temp)

mean = [8, 9, 10, 11, 12]
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.plot(mean, reward_greedy, color='r', label='greedy', linewidth=1, marker='>')
ax.plot(mean, reward_balance, color='g', label='balance', linewidth=1, marker='s')
ax.plot(mean, reward_solver, color='k', label='cvx', linewidth=1, marker='*')
ax.plot(mean, reward_cons_q, color='b', label='consQ', linewidth=1, marker='|')

ax.grid()
ax.legend(loc=0)
ax.set_xlabel("Mean of the energy distribution")
ax.set_ylabel("sum transmission rate")
plt.savefig('compare.png', dpi=600)
#tikzplotlib.save('compare.tex')
plt.show()
>>>>>>> a333fb8c0b3eb631d90b5a577be528981d5baa6a
