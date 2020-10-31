import torch
import numpy as np
import matplotlib.pyplot as plt 

dsNames = ['Mnist5', 'Cifar5'] #
offset = 0.02
for dsName in dsNames:
    data = torch.load('../Outputs/{}-EDLP-PrioriC'.format(dsName))
    x = data['x']
    realAcc = data['realAcc']
    maxEntropy = data['maxEntropy']
    maxRealAccIdx = np.argmax(realAcc)
    interIdx = 0
    for xidx in range(41):
        if abs(realAcc[xidx] - maxEntropy[xidx]) <= 0.02:
            interIdx = xidx
            break
    plt.xlabel(r"Prior $\mathcal{C}$")
    plt.ylabel('Value')
    plt.plot(x, realAcc, linestyle= '-', color = 'g', marker= '+',  label= 'real accuracy')
    plt.plot(x, maxEntropy, linestyle= ':', color = 'r', marker= 'x',  label= '% max entropy')
    plt.vlines(x[maxRealAccIdx], 0, 1, linestyle= '--', color = 'k', linewidth= 1.5)
    plt.vlines(x[interIdx], 0, 1, linestyle= '--', color = 'k', linewidth= 1.5)
    plt.vlines(20, 0, 1, linestyle= '--', color = 'b', linewidth= 2)
    if dsName == 'Mnist5':
        plt.axis([0 , 100, 0.0, 1.0])
    else:
        plt.axis([0 , 100, 0.0, 0.8])
    plt.grid(alpha= 0.8)
    plt.legend(loc= 4)
    plt.show()
