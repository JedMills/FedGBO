## About
Federated Global Biased Optimiser (FedGBO) is an optimisation algorithm for 
Federated Learning that used fixed local statistics during the client update 
loop, whilst having lower computational and communication costs compared to
other adaptive-FL algorithms. This repo contains code for reproducing the 
experiments in "Accelerating Federated Learning with a Global Biased Optimiser",
published in _IEEE Transactions on Computers_. If citing, please cite the IEEE 
paper. This repo contains PyTorch implementations of the following algorithms. 
- **FedGBO**: (proposed). "Accelerating Federated Learning with a Global Biased
Optimiser", Mills et al., IEEE TC, 2022.
- **FedAvg**: "Communication-Efficient Learning of Deep Networks from Decentralized
Data", McMahan et al., AISTATS 2017.
- **Mimelite**: "Mime: Mimicking Centralized Stochastic Algorithms in Federated 
Learning", Karimireddy et al., arxiv 2020. 
- **MFL**: "Accelerating Federated Learning via Momentum Gradient Descent”, Liu et 
al., IEEE TPDS 2020. 
- **AdaptiveFedOpt**: "Adaptive Federated Optimization", Reddi et al., ICLR 2021.
- **FedProx**: "Federated Optimization in Heterogeneous Networks", Li et al., MLSys 2020.
- **FedMAX**: "FedMAX: Mitigating Activation Divergence for Accurate and 
Communication-Efficient Federated Learning”, Chen et al., ECML PKDD 2020. 

The repo also contains a modifcation of the Mimelite algorithm (MimeXlite) as 
presented in Section 5.2 of the TC paper.


## Requirements
| Module       | Version    |
| ------------ | ---------- |
| Python       | 3.8.10     |
| Numpy        | 1.22.0     |
| Pytorch      | 1.8.2      |
| Scipy        | 1.7.3      |
| Torchvision  | 0.9.2      |
| Matplotlib   | 3.5.1      |
| H5Py         | 3.6.0      |
| Progressbar2 | 4.0.0      |


## Setup
The experiments in the paper use 4 benchmark FL datasets. Instructions for
downloading and preprocessing each for is given 
below. After downloading and preprocessing, the data files should be copied to
the appropriate folder in `./data/`.

**CIFAR100**: provided as part of [Tensorflow Federated API](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/cifar100). As of 05/10/2022 the compressed 
dataset can be directly downloaded [here](https://storage.googleapis.com/tff-datasets-public/cifar100.sqlite.lzma). 
After uncompressing copy the training and testing `.h5py` files to 
`./data/cifar100/`. In experiments, all 500 available workers were used.

**FEMNIST**: from the [LEAF](https://leaf.cmu.edu/) 
benchmark suite, with the relevant downloading and preprocessing instructions 
[here](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist). The 
command-line arguments for the LEAF preprocessing utility used were to generate
the full-sized non-iid dataset, with minimum 15 samples/user, sample-based 
80-20 train-test split were: 
`./preprocess.sh -s niid --sf 1.0 -k 15 -t sample --tf 0.8`. The resulting training 
`.json` files files should then be copied to `./data/femnist/train/` and the 
testing files to `./data/femnist/test/`. In the experiments, 3000 of the maximum 
3500 workers were used.

**Sent140**: also from [LEAF](https://leaf.cmu.edu/), with the instructions 
[here](https://github.com/TalwalkarLab/leaf/tree/master/data/sent140). The 
full-sized non-iid dataset, with minimum 10 samples/user, sample-based 80-20 
train-test split was created with the following command: `./preprocess.sh -s 
niid --sf 1.0 -k 10 -t sample --tf 0.8`. After that, use the  `gen_bag_of_words.py` 
script to create sparse bag-of-words vectors of size 5000 
as `.pkl` files. Copy these files to `./data/sent140`. All 21876 workers were 
used in the experiments.

**Shakespeare**: also from [LEAF](https://leaf.cmu.edu/) with instructions [here](https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare). The full-sized non-iid dataset, with a minimum of 0 samples/user and 80-20 train-test splits was created with the following command: `./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8`. The generated training and testing `.json` files can 
then be copied to `./data/shakes/`.

## Running
After downloading and preprocessing the FL datasets, the `main.py` can be run. 
The main function has example experiments for each algorithm to recreate the 
CIFAR - SGDm plot in Fig. 1 of the paper (plus the MimeXlite algorithm on the 
Shakespeare dataset). After running for 5000 rounds, the results can be plotted 
to reproduce the figure. The `experiment-settings.csv` file contains the tuned 
hyperparameters used in every simulation presented in the paper. `main.py` can 
be easily edited to run other experiments with these settings.
