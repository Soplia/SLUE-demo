import torch
import matplotlib.pyplot as plt 
import math
import numpy as np

inDomainDs = ['Cifar10', 'Cifar100'] # 'Cifar100',
outDomainForCifar10Ds = ['TEXTURE', 'PLACES365', 'LSUN', 'CIFAR100'] #'SVHN', 
outDomainForCifar100Ds = ['TEXTURE', 'PLACES365', 'LSUN', 'CIFAR10'] #'SVHN', 
outDomainDs = {'Cifar10': outDomainForCifar10Ds, 'Cifar100': outDomainForCifar100Ds}
models = ['CNN', 'EDL', 'EDLP']
fsize = 25
lwidth = 5
markSize = 20

for inDs in inDomainDs:
    for outDs in outDomainDs[inDs]:
        xcnn, ycnn = torch.load('../Outputs/{}-{}-CNN'.format(inDs, outDs))
        xedl, yedl = torch.load('../Outputs/{}-{}-EDL'.format(inDs, outDs)) 
        xedlp, yedlp = torch.load('../Outputs/{}-{}-EDLP'.format(inDs, outDs))
        
        offset = 0.00
        print('{}-{}'.format(inDs, outDs))
        plt.figure(figsize=(10,8))
        plt.subplots_adjust(top= .97119, bottom= 0.135, left = 0.135, hspace=0,wspace=0)

        plt.xlabel('Entropy', fontsize= fsize)
        plt.ylabel('Probability', fontsize= fsize)
        plt.plot(xcnn, ycnn, linestyle= ':', color = 'b',  marker= '1', label= 'CNN', linewidth= lwidth, markersize=markSize)
        plt.plot(xedl, yedl, linestyle= '-', color = 'g', marker= '+', label= 'EDL', linewidth= lwidth, markersize=markSize)
        plt.plot(xedlp, yedlp, linestyle= '--', color = 'r', marker= '*', label= 'SLUE', linewidth= lwidth, markersize=markSize)
        plt.axis([0.0 , 3.4, 0.0, 1.0+offset])
        plt.yticks(fontsize= fsize)
        plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.4], fontsize= fsize)
        plt.grid(alpha= 0.8)
        plt.legend(fontsize= fsize)
        plt.savefig('./ResultPictures/{}-{}'.format(inDs.lower(), outDs.lower()))
        plt.show()
