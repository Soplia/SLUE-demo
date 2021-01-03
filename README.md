### The EDLP result visualization demo

#### Coding environment configurations

1. OS: Windows10; Complier: Visual Studio2017; Environment Manager: Conda
4. conda env: EdlpEnvironment.yml

####  Experiments

##### Initial stage choice

<img src="https://hexo-eu-1259148800.cos.eu-frankfurt.myqcloud.com/ijcnn/CifarPriorC.png" alt="CifarPriorC" style="zoom:33%;" /><img src="https://hexo-eu-1259148800.cos.eu-frankfurt.myqcloud.com/ijcnn/MnistPriorC.png" style="zoom:33%;" />

To balance these two criteria and take the integer multiples value of the number of training classes, a value **four times the number of training classes** was used in the experiments.

#####  Synthetic dataset

<img src="https://hexo-eu-1259148800.cos.eu-frankfurt.myqcloud.com/ijcnn/cifar10-lsun.png" alt="cifar10-lsun" style="zoom: 33%;" /><img src="https://hexo-eu-1259148800.cos.eu-frankfurt.myqcloud.com/ijcnn/cifar100-lsun.png" alt="cifar100-lsun" style="zoom:33%;" />

Empirical CDF for the entropy of the predictive distributions on the out-of-domain dataset (LSUN) based on a model trained with CIFAR10 (left column) and CIFAR100 (right column) datasets

##### Adversarial dataset

<img src="https://hexo-eu-1259148800.cos.eu-frankfurt.myqcloud.com/ijcnn/MnistEplisonRealAcc.png" alt="MnistEplisonMaxEntropy" style="zoom:33%;" /><img src="https://hexo-eu-1259148800.cos.eu-frankfurt.myqcloud.com/ijcnn/MnistEplisonMaxEntropy.png" alt="MnistEplisonMaxEntropy" style="zoom:33%;" />

Real accuracy and entropy as a function of the adversarial perturbation based on MNIST

#####  Batch size estimation

<img src="https://hexo-eu-1259148800.cos.eu-frankfurt.myqcloud.com/ijcnn/BS-Cifar10-LSUN-EDLP.png" alt="BS-Cifar10-LSUN-EDLP" style="zoom:33%;" /><img src="https://hexo-eu-1259148800.cos.eu-frankfurt.myqcloud.com/ijcnn/BS-Cifar10-SVHN-EDLP.png" alt="BS-Cifar10-LSUN-EDLP" style="zoom:33%;" />

Empirical CDF for the entropy of the predictive distributions on the out-of-domain datasets (LSUN and SVHN) based on a model trained with CIFAR10

#### Conclusion

The aim of the present research was to extend the existing EDL method by taking base rates into account. It verified the SLUE method uncertainty performance through out-of-domain datasets and adversarial datasets. It introduced how to select the initial parameters and use real accuracy as one of the criteria. The most obvious finding to emerge from this study is that guided by base rates, the new SLUE method works better not only in terms of real accuracy but also on the uncertainty evaluation performance. As it can reject out-of-domain samples, this approach will prove useful in improving model robustness. Although extensive research has been carried out, one issue is that the uncertainty delineation is still not complete. In this study, the uncertainty is calculated for the whole state space. Consequently, in the future, we will provide uncertainty for all subsets of state space to provide more information for making predictions.

