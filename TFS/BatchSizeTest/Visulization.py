import torch
import matplotlib.pyplot as plt 
import math
import numpy as np

inDomainDs = ['Cifar10', 'Cifar100'] # 
outDomainForCifar10Ds = ['SVHN',  'LSUN'] #'TEXTURE','PLACES365',, 'CIFAR100'
outDomainForCifar100Ds = ['SVHN', 'LSUN'] #'TEXTURE', 'PLACES365', 'CIFAR10'
outDomainDs = {'Cifar10': outDomainForCifar10Ds, 'Cifar100': outDomainForCifar100Ds}
linestyles = ['-', '--', '-.', ':', '-']
colors = ['orangered', 'orange', 'lawngreen', 'darkturquoise', 'dodgerblue']
markers = ['3', '+', '*', '>', 'o']
batchSizes = [20, 50, 100, 200, 500] #, 1000
fsize = 25
for inDs in inDomainDs:
    for outDs in outDomainDs[inDs]:
        print('{}-{}'.format(inDs, outDs))
        plt.figure(figsize=(10,8))
        plt.subplots_adjust(top= .97119, bottom= 0.135, left = 0.135, hspace=0,wspace=0)

        for i, batchSize in enumerate(batchSizes):
            xedlp, yedlp = torch.load('../Outputs/bsTest/{}-{}-EDLP-{}'.format(inDs, outDs, batchSize))
            offset = 0.00            
            plt.plot(xedlp, yedlp, linestyle= linestyles[i], linewidth= 5, color = colors[i], label= str(batchSize))
        #plt.title('{}-{}'.format(inDs, outDs))
        plt.xlabel('Entropy', fontsize=fsize)
        plt.ylabel('Probability', fontsize=fsize)
        plt.axis([0.0 , 3.4, 0.0, 1.0+offset])
        plt.yticks(fontsize= fsize)
        plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.4], fontsize= fsize)
        plt.grid(alpha= 0.8)
        plt.legend(fontsize=fsize, loc= 4)
        plt.savefig('./ResultPictures/BS-{}-{}-EDLP'.format(inDs, outDs))
        plt.show()
        