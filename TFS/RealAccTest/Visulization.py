import torch
import matplotlib.pyplot as plt 

uncertaintyTh = torch.load('../Outputs/UncertaintyThresholds')
mnistAcc = torch.squeeze(torch.load('../Outputs/MnistAccAcordingToUncertaintyTh'))
cifarAcc = torch.squeeze(torch.load('../Outputs/Cifar5AccAcordingToUncertaintyTh'))

offset = 0.02
plt.xlabel('Uncertainty Threshold')
plt.ylabel('Real Accuracy')

plt.plot(uncertaintyTh, mnistAcc, linestyle= '-', color = 'k', marker= 's',  label= 'MNIST dataset')
plt.plot(uncertaintyTh, cifarAcc, linestyle= '-', color = 'k', marker= '^',  label= 'CIFAR5 dataset')
plt.axis([uncertaintyTh[0] , 1.0+offset, min(torch.min(mnistAcc), torch.min(cifarAcc)) - offset , max(torch.max(mnistAcc), torch.max(cifarAcc))])
plt.yticks([0.83, 0.85, 0.97,0.99, 1.0])
plt.grid(alpha= 0.8)
plt.legend()
plt.show()
