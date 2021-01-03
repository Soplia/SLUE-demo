import torch
import numpy as np
import matplotlib.pyplot as plt 

dsNames = ['Mnist5', 'Cifar5'] #
names= {'Mnist5': 'Mnist', 'Cifar5': 'Cifar'}
offset = 0.02
fsize = 25
lwidth = 5
markSize = 10
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
    plt.figure(figsize=(10,8))
    plt.subplots_adjust(top= .97119, bottom= 0.135, left = 0.135, hspace=0,wspace=0)
    plt.xlabel(r"Prior $\mathcal{C}$", fontsize= fsize)
    plt.ylabel('Value', fontsize= fsize)
    plt.plot(x, realAcc, linestyle= '-', color = 'g', marker= '+',  label= 'real accuracy', linewidth= lwidth, markersize=markSize)
    plt.plot(x, maxEntropy, linestyle= ':', color = 'r', marker= 'x',  label= '% max entropy', linewidth= lwidth, markersize=markSize)
    plt.vlines(x[maxRealAccIdx], 0, 1, linestyle= '--', color = 'k', linewidth= lwidth)
    plt.vlines(x[interIdx], 0, 1, linestyle= '--', color = 'k', linewidth= lwidth)
    plt.vlines(20, 0, 1, linestyle= '--', color = 'b', linewidth= lwidth + .5)
    if dsName == 'Mnist5':
        plt.axis([0 , 100, 0.0, 1.0])
    else:
        plt.axis([0 , 100, 0.0, 0.8])
    plt.yticks(fontsize= fsize)
    plt.xticks(fontsize= fsize)
    plt.grid(alpha= 0.8)
    plt.legend(loc= 4, fontsize= fsize)
    plt.savefig('./ResultPictures/{}PriorC'.format(names[dsName]))
    plt.show()
