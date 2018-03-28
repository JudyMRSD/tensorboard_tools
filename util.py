import matplotlib.pyplot as plt
import csv
import numpy as np

def running_mean(y, N):
    y = np.array(y)
    # insert: Insert values 0 along the given axis before the given indices 0.
    cumsum = np.cumsum(np.insert(y, 0, 0))

    result = (cumsum[N:] - cumsum[:-N])/N
    # print("x",x.shape, "cumsum", result.shape)
    return result


def plot_running_mean(x, y, filename, runningMeanLength = 100):
    smoothed_rews = running_mean(y, runningMeanLength)

    skipNums = len(smoothed_rews)
    plt.plot(x[-skipNums:], smoothed_rews)
    # moving average of G_t
    #plt.plot(x, y, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    #plt.savefig('../result/' + filename +'.jpg')


x = []
y = []

with open('./tensorboard_data/reward.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots)
    for row in plots:
        x.append(float(row[1]))
        y.append(float(row[2]))

plot_running_mean(x, y, "loss.png")

'''
plt.plot(x,y, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
'''



