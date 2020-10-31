import torch
import matplotlib.pyplot as plt 
import math
import numpy as np

inDomainDs = [ 'Cifar100','Cifar100'] #
outDomainForCifar10Ds = ['SVHN', 'LSUN', 'CIFAR100']
outDomainForCifar100Ds = ['SVHN', 'LSUN', 'CIFAR10']
outDomainDs = {'Cifar10': outDomainForCifar10Ds, 'Cifar100': outDomainForCifar100Ds}
models = ['CNN', 'EDL', 'EDLP']
print (math.log(10, 2))

for inDs in inDomainDs:
    for outDs in outDomainDs[inDs]:
        xcnn, ycnn = torch.load('../Outputs/{}-{}-CNN'.format(inDs, outDs))
        xedl, yedl = torch.load('../Outputs/{}-{}-EDL'.format(inDs, outDs)) 
        xedlp, yedlp = torch.load('../Outputs/{}-{}-EDLP'.format(inDs, outDs))
        
        offset = 0.00
        print('{}-{}'.format(inDs, outDs))
        plt.xlabel('Entropy')
        plt.ylabel('Probability')
        plt.plot(xcnn, ycnn, linestyle= ':', color = 'b',  marker= '^', label= 'CNN')
        plt.plot(xedl, yedl, linestyle= '-', color = 'g', marker= '+', label= 'EDL')
        plt.plot(xedlp, yedlp, linestyle= '--', color = 'r', marker= '*', label= 'EDLP')
        plt.axis([0.0 , 3.4, 0.0, 1.0+offset])
        plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.4])
        plt.grid(alpha= 0.8)
        plt.legend()
        plt.show()
