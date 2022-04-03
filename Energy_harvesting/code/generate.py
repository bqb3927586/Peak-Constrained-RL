import numpy as np
from scipy.stats import truncnorm

H = 20
E_MAX = 20


def generate_random_mean(min_mean, max_min, std, test_num):
    mean = np.random.randint(min_mean, max_min + 1, H + 1)
    E = np.zeros([test_num, H + 1])

    for i in range(test_num):
        for j in range(H+1):
            lower_bound = (0 - mean[j]) / std
            upper_bound = (E_MAX - mean[j]) / std
            E[i, j] = truncnorm.rvs(lower_bound, upper_bound, loc=mean[j], scale=std, size=1)
            E[i, j] = np.round(E[i, j])

    #np.save('energy_random.npy', E)


def generate_fix_mean(min_mean, max_min, std, test_num):
    mean = np.arange(min_mean, max_min + 1)
    for avg in mean:
        E = np.zeros([test_num, H + 1])
        for i in range(test_num):
            lower_bound = (0 - avg) / std
            upper_bound = (E_MAX - avg) / std
            E[i, :] = truncnorm.rvs(lower_bound, upper_bound, loc=avg, scale=std, size= H + 1)
            E[i, :] = np.round(E[i, :])

        np.save('./data/energy_'+str(avg)+'.npy', E)


if __name__ == '__main__':
    generate_fix_mean(10, 11, 5, 1000)
    #generate_fix_mean(8, 13, 5, 1000)
