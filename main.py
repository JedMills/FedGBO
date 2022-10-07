import os
# required for pytorch deterministic GPU behaviour
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import pickle
from data_utils import *
from models import *
from optimizers import *
from torch.optim import SGD as ClientSGD
from fl_algs import *  
from progressbar import progressbar
import random



def setup_lr_task(dset_name, device):
    """
    Return data feeders, model class, per-round clients and batch size for 
    given experimental task as used in the FedGBO paper experiments.
    
    Args:
    - dset_name: {str} one of ['femnist', 'cifar', 'sent140', 'shakes']
    - device:    {torch.device} to place train and test data tensors on
    
    Returns:
    - data_feeders: {list} of PyTorchDataFeeder, length==num_workers
    - model:        {FLModel} class 
    - M:            {int} number of workers sampled per round
    - B:            {int} batch size
    - test_data:    {tuple} (x,y) of torch.tensors on device
    """
    if dset_name == 'femnist':
        train, test     = load_femnist( './datasets/femnist/train', 
                                        './datasets/femnist/test', 
                                        3000)
        data_feeders    = [PyTorchDataFeeder(   x, torch.float32, 
                                                y, 'long', 
                                                device=torch.device('cpu'),
                                                cast_device=device)
                            for (x, y) in zip(train[0], train[1])]
        model           = FEMNISTModel
        M               = 30
        B               = 32
        test_data       = ( to_tensor(test[0], device, torch.float32), 
                            to_tensor(test[1], device, 'long'))
    
    elif dset_name == 'sent140':
        train, test     = load_sent140( 
                                './datasets/sent140/train_data_sparse.pkl',
                                './datasets/sent140/test_data_sparse.pkl', 
                                21876)
        data_feeders    = [ PyTorchDataFeeder(  x, torch.float32, 
                                                y, 'long', 
                                                device=torch.device('cpu'),
                                                cast_device=device)
                            for (x, y) in zip(train[0], train[1])]
        model           = Sent140Model
        M               = 22
        B               = 8
        test_data       = ( to_tensor(test[0], device, torch.float32), 
                            to_tensor(test[1], device, 'long'))

    elif dset_name == 'shakes':
        train, test     = load_shakes(  
                                './datasets/shakespeare/shakes_niid_train.json',
                                './datasets/shakespeare/shakes_niid_test.json', 
                                660)
        data_feeders    = [ PyTorchDataFeeder(  x, 'long', 
                                                y, 'long', 
                                                device=torch.device('cpu'),
                                                cast_device=device)
                            for (x, y) in zip(train[0], train[1])]
        model           = ShakesModel
        M               = 7
        B               = 32
        test_data       = ( to_tensor(test[0], device, 'long'), 
                            to_tensor(test[1], device, 'long'))

    elif dset_name == 'cifar':
        train, test     = load_cifar100(
                                './datasets/cifar100/fed_cifar100_train.h5',
                                './datasets/cifar100/fed_cifar100_test.h5', 
                                500)
        crop            = torchvision.transforms.RandomCrop(32, padding=4)
        flip            = torchvision.transforms.RandomHorizontalFlip(p=0.5)
        transform       = lambda x: crop(flip(x))
        data_feeders    = [ PyTorchDataFeeder(  x, torch.float32, 
                                                y, 'long',
                                                device=torch.device('cuda:0'),
                                                transform=transform)
                            for (x, y) in zip(train[0], train[1])]
        model           = CIFAR100Model
        M               = 5
        B               = 32
        test_data       = ( to_tensor(test[0], device, torch.float32), 
                            to_tensor(test[1], device, 'long'))

    else:
        raise RuntimeError('Invalid dataset: {}'.format(dset_name))
        
        
    return data_feeders, model, M, B, test_data


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility. 
    
    Args:
    - seed: {int}
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    
def main():
    """
    Demonstrations of simulation runs for all algorithms used in the paper.
    """

    # Settings used in all paper experiments are in './experiment-settings.csv'
    T         = 5000        # total communication rounds
    TEST_FREQ = 50          # how frequently to compute test stats
    seed      = 0           # random seed
    device    = torch.device('cuda:0')
    torch.use_deterministic_algorithms(True)
    
    # FedGBO+SGDm on CIFAR100, K=10. Experiments are set up as follows: call 
    # setup_lr_task to get the task settings used in the paper, create the 
    # PyTorch model that optimisation will be performed with, create the 
    # appropriate optimiser and set that as the model's optimiser, then call the
    # relevant algorithm function from fl_algs.py.
    # CIFAR100,FedGBO,SGDm,10,"lr=0.1, beta=0.9"
    set_random_seeds(seed)
    data_feeders, model_class, M, B, test_data = setup_lr_task('cifar', device)
    model       = model_class().to(device)
    optimizer   = FixedStatsSGDm(   model.parameters(),
                                    lr=0.1,
                                    beta=0.9,
                                    device=device)
    model.set_optim(optimizer)
    data        = run_fedgbo(   data_feeders, 
                                test_data,
                                model, 
                                optimizer, 
                                T=T, 
                                M=M, 
                                K=10, 
                                B=B,
                                test_freq=TEST_FREQ)
    fname = 'cifar_fedgbo-sgdm_K={}_lr={}_b={}.pkl'
    save_data(data, fname.format(10, 0.1, 0.9), seed)
    
    
    # Mimelite+SGDm on CIFAR100, K=10. FedGBO, Mimelite and MimeXlite all use 
    # FixedStatsSGDm/FixedStatsRMSProp/FixedStatsAdam optimisers. 
    # CIFAR100,Mimelite,SGDm,10,"lr=0.01, beta=0.9"
    set_random_seeds(seed)
    data_feeders, model_class, M, B, test_data = setup_lr_task('cifar', device)
    model       = model_class().to(device)
    optimizer   = FixedStatsSGDm(   model.parameters(),
                                    lr=0.01,
                                    beta=0.9,
                                    device=device)
    model.set_optim(optimizer)
    data        = run_mimelite( data_feeders, 
                                test_data,
                                model, 
                                optimizer, 
                                T=T, 
                                M=M, 
                                K=10, 
                                B=B,
                                test_freq=TEST_FREQ)
    fname = 'cifar_mimelite-sgdm_K={}_lr={}_b={}.pkl'
    save_data(data, fname.format(10, 0.01, 0.9), seed)
    
    
    # MFL+SGDm on CIFAR100, K=10. MFL uses MFLSGDm/MFLRMSProp/MFLAdam optimisers
    # CIFAR100,MFL,SGDm,10,"lr=0.01, beta=0.99"
    set_random_seeds(seed)
    data_feeders, model_class, M, B, test_data = setup_lr_task('cifar', device)
    model       = model_class().to(device)
    optimizer   = MFLSGDm(  model.parameters(),
                            lr=0.01,
                            beta=0.99,
                            device=device)
    model.set_optim(optimizer)
    data        = run_mfl(  data_feeders, 
                            test_data,
                            model, 
                            optimizer, 
                            T=T, 
                            M=M, 
                            K=10, 
                            B=B,
                            test_freq=TEST_FREQ)
    fname = 'cifar_mfl-sgdm_K={}_lr={}_b={}.pkl'
    save_data(data, fname.format(10, 0.01, 0.99), seed)
    
    
    # AdaptiveFedOpt+SGDm on CIFAR100, K=10. For AdaptiveFedOpt, clients use 
    # ClientSGD optimiser and ServerSGD/ServerRMSPRop/ServerAdam optimiser. The 
    # called FL function is the generalised FedAvg that allows server optimisers
    # CIFAR100,AdaptiveFedOpt,SGDm,10,"slr=1.0, sbeta=0.99, clr=0.1"
    set_random_seeds(seed)
    data_feeders, model_class, M, B, test_data = setup_lr_task('cifar', device)
    model        = model_class().to(device)
    optimizer    = ClientSGD(model.parameters(), 0.1)
    server_optim = ServerSGDm(  model.get_params_numpy(),
                                lr=1.0,
                                beta=0.99)
    model.set_optim(optimizer)
    data         = run_adaptive_fed_opt(data_feeders, 
                                        test_data, 
                                        model, 
                                        server_optim, 
                                        T=T, 
                                        M=M, 
                                        K=10, 
                                        B=B, 
                                        test_freq=TEST_FREQ)
    fname = 'shakes_adaptivefedopt-sgdm_K={}_slr={}_b={}_clr={}.pkl'
    save_data(data, fname.format(10, 1.0, 0.99, 0.1), seed)
    
    
    # FedAvg on CIFAR100, K=10. AdaptiveFedOpt with ServerSgd(lr=1.0) is
    # equivalent to the basic FedAvg.
    # CIFAR100,FedAvg,SGD,10,lr=0.01
    set_random_seeds(seed)
    data_feeders, model_class, M, B, test_data = setup_lr_task('cifar', device)
    model        = model_class().to(device)
    optimizer    = ClientSGD(model.parameters(), 0.01)
    server_optim = ServerSGD(lr=1.0)
    model.set_optim(optimizer)
    data         = run_adaptive_fed_opt(data_feeders, 
                                        test_data, 
                                        model, 
                                        server_optim, 
                                        T=T, 
                                        M=M, 
                                        K=10, 
                                        B=B, 
                                        test_freq=TEST_FREQ)
    fname = 'cifar_fedavg_K={}_lr={}.pkl'
    save_data(data, fname.format(10, 0.01), seed)
    
    
    # FedMAX on CIFAR100 with K=10. Settings are the same as FedAvg but with 
    # fedmax_beta>0 set, as FedMAX simply has a different error function on 
    # the clients.
    # CIFAR100,FedMAX,SGD,10,"lr=0.01, beta=1.0"
    set_random_seeds(seed)
    data_feeders, model_class, M, B, test_data = setup_lr_task('cifar', device)
    model        = model_class(fedmax_beta=1.0).to(device)
    optimizer    = ClientSGD(model.parameters(), 0.01)
    server_optim = ServerSGD(lr=1.0)
    model.set_optim(optimizer)
    data         = run_adaptive_fed_opt(data_feeders, 
                                        test_data, 
                                        model, 
                                        server_optim, 
                                        T=T, 
                                        M=M, 
                                        K=10, 
                                        B=B, 
                                        test_freq=TEST_FREQ)
    fname = 'cifar_fedmax_K={}_lr={}_b={}.pkl'
    save_data(data, fname.format(10, 0.01, 1.0), seed)
    
    
    # FedProx on CIFAR100, K=10, uses custom optimiser
    # CIFAR100,FedProx,SGD,10,"lr=0.01, mu=0.001"
    set_random_seeds(seed)
    data_feeders, model_class, M, B, test_data = setup_lr_task('cifar', device)
    model     = model_class().to(device)
    optimizer = FedProxOptim(   model.parameters(), 
                                lr=0.01, 
                                mu=0.001, 
                                device=device)
    model.set_optim(optimizer)
    data =  run_fedprox(data_feeders, 
                        test_data, 
                        model, 
                        optimizer, 
                        T=T, 
                        M=M, 
                        K=10, 
                        B=B, 
                        test_freq=TEST_FREQ)
    fname = 'cifar_fedprox_K={}_lr={}_mu={}.pkl'
    save_data(data, fname.format(10, 0.01, 0.001), seed)
    
    
    # The paper only includes MimeXlite results for the Shakespeare dataset,
    # so here is an example using MimeXlite+Adam on Shakespeare, K=10.
    # MimeXlite uses run_mimelite with full_grads=False
    # Shakespeare,MimeXlite,Adam,10,"lr=0.001, beta1=0.6, beta2=0.99, eps=1e-3"
    set_random_seeds(seed)
    data_feeders, model_class, M, B, test_data = setup_lr_task('shakes', device)
    model       = model_class().to(device)
    optimizer   = FixedStatsAdam(   model.parameters(),
                                    lr=0.001,
                                    beta1=0.6,
                                    beta2=0.99,
                                    epsilon=1e-3,
                                    device=device)
    model.set_optim(optimizer)
    data        = run_mimelite( data_feeders, 
                                test_data,
                                model, 
                                optimizer, 
                                T=T, 
                                M=M, 
                                K=10, 
                                B=B,
                                test_freq=TEST_FREQ,
                                full_grads=False)    # mimeXlite minibatch grads
    fname = 'shakes_mimeXlite-adam_K={}_lr={}_b1={}_b2={}_eps={}.pkl'
    save_data(data, fname.format(10, 0.001, 0.6, 0.99, 1e-3), seed)


if __name__ == '__main__':
    main()
