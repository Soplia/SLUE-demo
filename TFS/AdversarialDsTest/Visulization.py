import torch
import matplotlib.pyplot as plt 

epsilonMnist = torch.load('../Outputs/EpsilonMnist')
cnnAccMnist = torch.load('../Outputs/MnistAdvRealAcc-CNN')
edlAccMnist = torch.load('../Outputs/MnistAdvRealAcc-EDL')
edlpAccMnist = torch.load('../Outputs/MnistAdvRealAcc-EDLP')

cnnEntropyMnist = torch.load('../Outputs/MnistAdvMaxEntropy-CNN')
edlEntropyMnist = torch.load('../Outputs/MnistAdvMaxEntropy-EDL')
edlpEntropyMnist = torch.load('../Outputs/MnistAdvMaxEntropy-EDLP')

offset = 0.01
fsize = 25
lwidth = 5
markSize = 20
linestyles = ['-', '--', '-.', ':', '-']
colors = ['orangered', 'orange', 'lawngreen', 'darkturquoise', 'dodgerblue']
markers = ['3', '+', '*', '>', 'o']

plt.figure(figsize=(10,8))
plt.subplots_adjust(top= .97119, bottom= 0.135, left = 0.135, hspace=0,wspace=0)
plt.xlabel('Epsilon', fontsize= fsize)
plt.ylabel('Real Accyracy', fontsize= fsize)
plt.plot(epsilonMnist, cnnAccMnist, linestyle= '-', color = 'b', marker= 'x', linewidth= lwidth, markersize=markSize, label= 'CNN')
plt.plot(epsilonMnist, edlAccMnist, linestyle= '--', color = 'g', marker= '+',  linewidth= lwidth, markersize=markSize, label= 'EDL')
plt.plot(epsilonMnist, edlpAccMnist, linestyle= '-.', color = 'r', marker= 's',  linewidth= lwidth, markersize=markSize, label= 'SLUE')
plt.yticks(fontsize= fsize)
plt.xticks(fontsize= fsize)
plt.axis([0.0 , 1.0+offset, 0.0, 1.0])
plt.grid(alpha= 0.8)
plt.legend(fontsize= fsize)
plt.savefig('./ResultPictures/MnistEplisonRealAcc')
plt.show()

plt.figure(figsize=(10,8))
plt.subplots_adjust(top= .97119, bottom= 0.135, left = 0.135, hspace=0,wspace=0)
plt.xlabel('Epsilon', fontsize= fsize)
plt.ylabel('% Max Entropy', fontsize= fsize)
plt.plot(epsilonMnist, cnnEntropyMnist, linestyle= '-', color = 'b', marker= 'x',  label= 'CNN',  linewidth= lwidth, markersize=markSize)
plt.plot(epsilonMnist, edlpEntropyMnist, linestyle= '--', color = 'g', marker= '+',  label= 'EDL',  linewidth= lwidth, markersize=markSize)
plt.plot(epsilonMnist, edlEntropyMnist, linestyle= '-.', color = 'r', marker= 's',  label= 'SLUE',  linewidth= lwidth, markersize=markSize)
plt.axis([0.0 , 1.0 +offset, 0.0, max(torch.max(cnnEntropyMnist), torch.max(edlEntropyMnist), torch.max(edlpEntropyMnist)).item()+offset])
plt.yticks(fontsize= fsize)
plt.xticks(fontsize= fsize)
plt.grid(alpha= 0.8)
plt.legend(fontsize= fsize)
plt.savefig('./ResultPictures/MnistEplisonMaxEntropy')
plt.show()

epsilonCifar5 = torch.load('../Outputs/EpsilonCifar5')
cnnAccCifar5 = torch.load('../Outputs/Cifar5AdvRealAcc-CNN')
edlAccCifar5 = torch.load('../Outputs/Cifar5AdvRealAcc-EDL')
edlpAccCifar5 = torch.load('../Outputs/Cifar5AdvRealAcc-EDLP')
cnnEntropyCifar5 = torch.load('../Outputs/Cifar5AdvMaxEntropy-CNN')
edlEntropyCifar5 = torch.load('../Outputs/Cifar5AdvMaxEntropy-EDL')
edlpEntropyCifar5 = torch.load('../Outputs/Cifar5AdvMaxEntropy-EDLP')

plt.figure(figsize=(10,8))
plt.subplots_adjust(top= .97119, bottom= 0.135, left = 0.135, hspace=0,wspace=0)
plt.xlabel('Epsilon', fontsize= fsize)
plt.ylabel('Real Accyracy', fontsize= fsize)
plt.plot(epsilonCifar5, cnnAccCifar5, linestyle= '-', color = 'b', marker= 'x',  label= 'CNN',  linewidth= lwidth, markersize=markSize)
plt.plot(epsilonCifar5, edlAccCifar5, linestyle= '--', color = 'g', marker= '+',  label= 'EDL',  linewidth= lwidth, markersize=markSize)
plt.plot(epsilonCifar5, edlpAccCifar5, linestyle= '-.', color = 'r', marker= 's',  label= 'SLUE',  linewidth= lwidth, markersize=markSize)
plt.axis([0.0 , 0.4+offset, 0.0, max(torch.max(cnnAccCifar5), torch.max(edlAccCifar5), torch.max(edlpAccCifar5)).item()+offset])
plt.xticks(fontsize= fsize)
plt.yticks([0.1, 0.2, 0.3,0.4, .5, .6, .7, .8, .9], fontsize= fsize)
plt.grid(alpha= 0.8)
plt.legend(fontsize= fsize)
plt.savefig('./ResultPictures/Cifar5EplisonRealAcc')
plt.show()

plt.figure(figsize=(10,8))
plt.subplots_adjust(top= .97119, bottom= 0.135, left = 0.135, hspace=0,wspace=0)
plt.xlabel('Epsilon', fontsize= fsize)
plt.ylabel('% Max Entropy', fontsize= fsize)
plt.plot(epsilonCifar5, cnnEntropyCifar5, linestyle= '-', color = 'b', marker= 'x',  label= 'CNN',  linewidth= lwidth, markersize=markSize)
plt.plot(epsilonCifar5, edlEntropyCifar5, linestyle= '--', color = 'g', marker= '+',  label= 'EDL',  linewidth= lwidth, markersize=markSize)
plt.plot(epsilonCifar5, edlpEntropyCifar5, linestyle= '-.', color = 'r', marker= 's',  label= 'SLUE',  linewidth= lwidth, markersize=markSize)
plt.axis([0.0 , 0.4+offset, 0.0, max(torch.max(cnnEntropyCifar5), torch.max(edlEntropyCifar5), torch.max(edlpEntropyCifar5)).item()+offset])
plt.yticks([0.1, 0.2, 0.3,0.4, .5], fontsize= fsize)
plt.xticks(fontsize= fsize)
plt.grid(alpha= 0.8)
plt.legend(fontsize= fsize)
plt.savefig('./ResultPictures/Cifar5EplisonMaxEntropy')
plt.show()

