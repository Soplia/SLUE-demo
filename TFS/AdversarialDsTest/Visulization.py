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
plt.xlabel('Epsilon')
plt.ylabel('Real Accyracy')
plt.plot(epsilonMnist, cnnAccMnist, linestyle= '-', color = 'b', marker= 'x',  label= 'CNN')
plt.plot(epsilonMnist, edlAccMnist, linestyle= '--', color = 'g', marker= '+',  label= 'EDL')
plt.plot(epsilonMnist, edlpAccMnist, linestyle= '-.', color = 'r', marker= 's',  label= 'EDLP')
plt.axis([0.0 , 1.0+offset, 0.0, 1.0])
plt.grid(alpha= 0.8)
plt.legend()
plt.show()


plt.xlabel('Epsilon')
plt.ylabel('% Max Entropy')
plt.plot(epsilonMnist, cnnEntropyMnist, linestyle= '-', color = 'b', marker= 'x',  label= 'CNN')
plt.plot(epsilonMnist, edlEntropyMnist, linestyle= '--', color = 'g', marker= '+',  label= 'EDL')
plt.plot(epsilonMnist, edlpEntropyMnist, linestyle= '-.', color = 'r', marker= 's',  label= 'EDLP')
plt.axis([0.0 , 1.0 +offset, 0.0, max(torch.max(cnnEntropyMnist), torch.max(edlEntropyMnist), torch.max(edlpEntropyMnist)).item()+offset])
plt.grid(alpha= 0.8)
plt.legend()
plt.show()

epsilonCifar5 = torch.load('../Outputs/EpsilonCifar5')
cnnAccCifar5 = torch.load('../Outputs/Cifar5AdvRealAcc-CNN')
edlAccCifar5 = torch.load('../Outputs/Cifar5AdvRealAcc-EDL')
edlpAccCifar5 = torch.load('../Outputs/Cifar5AdvRealAcc-EDLP')
cnnEntropyCifar5 = torch.load('../Outputs/Cifar5AdvMaxEntropy-CNN')
edlEntropyCifar5 = torch.load('../Outputs/Cifar5AdvMaxEntropy-EDL')
edlpEntropyCifar5 = torch.load('../Outputs/Cifar5AdvMaxEntropy-EDLP')

plt.xlabel('Epsilon')
plt.ylabel('Real Accyracy')
plt.plot(epsilonCifar5, cnnAccCifar5, linestyle= '-', color = 'b', marker= 'x',  label= 'CNN')
plt.plot(epsilonCifar5, edlAccCifar5, linestyle= '--', color = 'g', marker= '+',  label= 'EDL')
plt.plot(epsilonCifar5, edlpAccCifar5, linestyle= '-.', color = 'r', marker= 's',  label= 'EDLP')
plt.axis([0.0 , 0.4+offset, 0.0, max(torch.max(cnnAccCifar5), torch.max(edlAccCifar5), torch.max(edlpAccCifar5)).item()+offset])
plt.yticks([0.1, 0.2, 0.3,0.4, .5, .6, .7, .8, .9])
plt.grid(alpha= 0.8)
plt.legend()
plt.show()

plt.xlabel('Epsilon')
plt.ylabel('% Max Entropy')
plt.plot(epsilonCifar5, cnnEntropyCifar5, linestyle= '-', color = 'b', marker= 'x',  label= 'CNN')
plt.plot(epsilonCifar5, edlEntropyCifar5, linestyle= '--', color = 'g', marker= '+',  label= 'EDL')
plt.plot(epsilonCifar5, edlpEntropyCifar5, linestyle= '-.', color = 'r', marker= 's',  label= 'EDLP')
plt.axis([0.0 , 0.4+offset, 0.0, max(torch.max(cnnEntropyCifar5), torch.max(edlEntropyCifar5), torch.max(edlpEntropyCifar5)).item()+offset])
plt.yticks([0.1, 0.2, 0.3,0.4, .5])
plt.grid(alpha= 0.8)
plt.legend()
plt.show()

