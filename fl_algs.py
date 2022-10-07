"""
Implementations of the following algorithms:
- FedGBO
- FedAvg
- Mimelite
- MimeXlite
- MFL
- AdaptiveFedOpt
- FedProx
- FedMAX
"""
import numpy as np
import torch
import torchvision
from progressbar import progressbar
from numpy.ma import array as masked_array



def init_stat_arrays(T):
    """
    Returns 4 0-arrays of length T.
    """
    return tuple(np.zeros((T), dtype=np.float32) for i in range(4))



def run_adaptive_fed_opt(   data_feeders, test_data, model, server_optim, 
                            T, M, K, B, test_freq=1, test_B=128):
    """
    Run generalized FedAvg with given server optimiser from the paper "Adaptive 
    Federated Optimization", Reddi et al, ICLR 2021. FedAvg with ServerSGD 
    (lr = 1.0) equates to standard FedAvg.
    
    Args:
    - data_feeders: {list} of NumpyDataFeeders, one for each worker
    - test_data:    {tuple} of torch.tensors, containing (x,y) test data
    - model:        {FLModel} that will perform the learning
    - server_optim: {ServerOptimizer} to update model on the server
    - T:            {int} number of rounds of FL
    - M:            {int} number of clients selected per round
    - K:            {int} number of local steps clients perform per round
    - B:            {int} client batch size
    - test_freq:    {int} how many rounds between testing the global model
    - test_B:       {int} test-set batch size
    
    Returns: train_errs, train_accs, test_errs, test_accs
    {np.ndarrays} of length T containing statistics. If test_freq>1, then 
    the test arrays will have 0s in the non-tested rounds.
    """
    W = len(data_feeders)
    train_es, train_as, test_es, test_as = init_stat_arrays(T)
    
    global_model = model.get_params_numpy()    # current global model
    round_agg    = model.get_params_numpy()    # client aggregate model
    
    for t in progressbar(range(T), redirect_stdout=True):
        round_agg       = round_agg.zeros_like()
        # id's of workers participating in this round
        user_idxs       = np.random.choice(W, M, replace=False)
        round_samples   = 0
        
        for user_idx in user_idxs:
            model.set_params(global_model)   # 'download' global model
            feeder = data_feeders[user_idx]
            
            # perform local training
            feeder.activate()
            for k in range(K):
                x, y         = feeder.next_batch(B)
                loss, acc    = model.train_step(x, y)
                train_es[t] += loss
                train_as[t] += acc
            feeder.deactivate()

            # 'upload' model deltas
            client_model = model.get_params_numpy()
            round_agg    = round_agg + (global_model - client_model)
            
        # 'pseudogradient' is the average of client model deltas
        grads       = round_agg / M
        # apply the server optimizer to produce new global model
        global_model = server_optim.apply_gradients(global_model, grads)
        
        if t % test_freq == 0:
            model.set_params(global_model)
            test_es[t], test_as[t] = model.test(test_data[0], 
                                                test_data[1], 
                                                test_B)
    
    train_es /= M * K
    train_as /= M * K
    
    return train_es, train_as, test_es, test_as



def run_mimelite(   feeders, test_data, model, optimizer, 
                    T, M, K, B, test_freq=1, test_B=128, full_grads=True):
    """
    Run mimelite from the paper "Mime: Mimicking Centralized Stochastic
    Algorithms in Federated Learning", Karimireddy et al., 8 Aug 2020, arXiv. 
    Also includes option to use stochastic gradients for the global optimisier 
    (rather than full-batch), for MimeXlite as presented in Section 5.2 of the 
    FedGBO paper.
    
    Args:
    - feeders:      {list} of NumpyDataFeeders, one for each worker
    - test_data:    {tuple} of torch.tensors, containing (x,y) test data
    - model:        {FLModel} that will perform the learning
    - optimizer:    {MimeliteOptimizer} to update client models
    - T:            {int} number of rounds of FL
    - M:            {int} number of clients selected per round
    - K:            {int} number of local steps clients perform per round
    - B:            {int} client batch size
    - test_freq:    {int} how many rounds between testing the global model
    - test_B:       {int} test-set batch size
    - full_grads:   {bool} use full-batch gradients for optimiser, or stochastic
    
    Returns: train_errs, train_accs, test_errs, test_accs
    {np.ndarrays} of length T containing statistics. If test_freq>1, then 
    the test arrays will have 0s in the non-tested rounds.
    """
    W = len(feeders)
    train_es, train_as, test_es, test_as = init_stat_arrays(T)

    global_model = model.get_params_numpy()  # current global model
    round_agg    = model.get_params_numpy()  # aggregate of client models 
    round_df     = model.get_params_numpy()  # aggregate of full-batch gradients
    
    for t in progressbar(range(T), redirect_stdout=True):
        round_agg = round_agg.zeros_like()
        round_df  = round_df.zeros_like()
        # id's of workers participating in this round
        user_idxs = np.random.choice(W, M, replace=False)
        
        for user_idx in user_idxs:
            model.set_params(global_model)   # 'download' global model
            data = feeders[user_idx]
            
            data.activate()
            # calculate and 'upload' full-batch gradients.
            if full_grads:
                round_df = round_df + model.calc_full_grads_numpy(data, test_B)
            else:
                round_df = round_df + model.calc_batch_grads_numpy(*data.next_batch(B))
            
            # perform local training
            for k in range(K):
                x, y          = data.next_batch(B)
                err, acc      = model.train_step(x, y)
                train_es[t]   += err
                train_as[t]   += acc
            data.deactivate()
                
            # 'upload' client model
            round_agg = round_agg + model.get_params_numpy()
            
        # update the global model and optimizer
        global_model = round_agg / M
        optimizer.update_moments(round_df / M)
        
        if t % test_freq == 0:
            model.set_params(global_model)
            test_es[t], test_as[t] = model.test(test_data[0], 
                                                test_data[1], 
                                                test_B)
    
    train_es /= (M * K)
    train_as /= (M * K)
    
    return train_es, train_as, test_es, test_as



def run_fedgbo( feeders, test_data, model, optimizer, 
                T, M, K, B, test_freq=1, test_B=128):
    """
    Run FedGBO - our proposed algorithm!
    
    Args:
    - feeders:      {list} of NumpyDataFeeders, one for each worker
    - test_data:    {tuple} of torch.tensors, containing (x,y) test data
    - model:        {FLModel} that will perform the learning
    - optimizer:    {FixedStatsOptimizer} to update client models
    - T:            {int} number of rounds of FL
    - M:            {int} number of clients selected per round
    - K:            {int} number of local steps clients perform per round
    - B:            {int} client batch size
    - test_freq:    {int} how many rounds between testing the global model
    - test_B:       {int} test-set batch size
    
    Returns: train_errs, train_accs, test_errs, test_accs
    {np.ndarrays} of length T containing statistics. If test_freq>1, then 
    the test arrays will have 0s in the non-tested rounds.
    """
    W = len(feeders)
    train_es, train_as, test_es, test_as = init_stat_arrays(T)
    global_model    = model.get_params_numpy()
    round_agg       = model.get_params_numpy()
    
    for t in progressbar(range(T), redirect_stdout=True):
        round_agg       = round_agg.zeros_like()
        # id's of workers participating in this round
        user_idxs       = np.random.choice(W, M, replace=False)
        
        for user_idx in user_idxs:
            model.set_params(global_model)
            data = feeders[user_idx]
            
            data.activate()
            # perform local training
            for k in range(K):
                x, y         = data.next_batch(B)
                err, acc     = model.train_step(x, y)
                train_es[t] += err
                train_as[t] += acc
            data.deactivate()
                
            # 'upload' client model
            round_agg    = round_agg + model.get_params_numpy()
            
        round_agg = round_agg / M

        grads        = optimizer.inv_grads(global_model, round_agg, K)
        global_model = round_agg.copy()
        optimizer.update_moments(grads)
        
        if t % test_freq == 0:
            model.set_params(global_model)
            test_es[t], test_as[t] = model.test(test_data[0], 
                                                test_data[1], 
                                                test_B)
    
    train_es /= M * K
    train_as /= M * K
    
    return train_es, train_as, test_es, test_as



def run_mfl(data_feeders, test_data, model, optimizer, 
            T, M, K, B, test_freq=1, test_B=128):
    """
    Run generalised Momentum Federated learning, that averages client optimisers
    alongside client models each round, from "Accelerating Federated Learning 
    via Momentum Gradient Descent", Liu et al, IEEE TPDS 2020. 
    
    Args:
    - feeders:      {list} of NumpyDataFeeders, one for each worker
    - test_data:    {tuple} of torch.tensors, containing (x,y) test data
    - model:        {FLModel} that will perform the learning
    - optimizer:    {MimeliteOptimizer} to update client models
    - T:            {int} number of rounds of FL
    - M:            {int} number of clients selected per round
    - K:            {int} number of local steps clients perform per round
    - B:            {int} client batch size
    - test_freq:    {int} how many rounds between testing the global model
    - test_B:       {int} test-set batch size
    
    Returns: train_errs, train_accs, test_errs, test_accs
    {np.ndarrays} of length T containing statistics. If test_freq>1, then 
    the test arrays will have 0s in the non-tested rounds.
    """
    W = len(data_feeders)
    train_es, train_as, test_es, test_as = init_stat_arrays(T)
    
    global_model    = model.get_params_numpy()     # current global model
    global_mmntm    = optimizer.get_momentum()     # current global momentum
    round_agg       = global_model.zeros_like()    # aggregate model
    round_m_agg     = global_mmntm.zeros_like()    # aggregate momentum
    
    for t in progressbar(range(T), redirect_stdout=True):
        round_agg       = round_agg.zeros_like()
        round_m_agg     = round_m_agg.zeros_like()
        # id's of workers participating in this round
        user_idxs       = np.random.choice(W, M, replace=False)
        round_samples   = 0
        
        for user_idx in user_idxs:
            model.set_params(global_model)          # 'download' global model
            optimizer.set_momentum(global_mmntm)    # 'download' global momentum
            feeder = data_feeders[user_idx]
            
            # perform local training
            feeder.activate()
            for k in range(K):
                x, y        = feeder.next_batch(B)
                loss, acc   = model.train_step(x, y)
                train_es[t] += loss
                train_as[t] += acc
            feeder.deactivate()

            # 'upload' model deltas
            round_agg   = round_agg + model.get_params_numpy()
            round_m_agg = round_m_agg + optimizer.get_momentum()
        
        global_model = round_agg / M
        global_mmntm = round_m_agg / M
        
        if t % test_freq == 0:
            model.set_params(global_model)
            test_es[t], test_as[t] = model.test(test_data[0], 
                                                test_data[1], 
                                                test_B)
    
    train_es /= M * K
    train_as /= M * K
    
    return train_es, train_as, test_es, test_as



def run_fedprox(data_feeders, test_data, model, prox_optim, T, M, K, 
                B, test_freq=1, test_B=128):
    """
    Run FedProx FedAvg from the paper "Federated Optimization in Heterogeneous
    Networks", Li et al, MLSys 2020. 
    
    Args:
    - data_feeders: {list} of NumpyDataFeeders, one for each worker
    - test_data:    {tuple} of torch.tensors, containing (x,y) test data
    - model:        {FLModel} that will perform the learning
    - prox_optim:   {FedProxOptim} optimizer used on clients
    - T:            {int} number of rounds of FL
    - M:            {int} number of clients selected per round
    - K:            {int} number of local steps clients perform per round
    - B:            {int} client batch size
    - test_freq:    {int} how many rounds between testing the global model
    - test_B:       {int} test-set batch size
    
    Returns: train_es, train_as, test_es, test_as
    {np.ndarrays} of length T containing statistics. If test_freq>1, then 
    the test arrays will have 0s in the non-tested rounds.
    """
    W = len(data_feeders)
    train_es, train_as, test_es, test_as = init_stat_arrays(T)
    
    round_model = model.get_params_numpy()    # current global model
    round_agg   = model.get_params_numpy()    # client aggregate model
        
    for t in progressbar(range(T)):
        round_agg = round_agg.zeros_like()
        user_idxs = np.random.choice(W, M, replace=False) # round clients
        
        # setting $\omega^t$ in Algorithm 2 of FedProx paper
        prox_optim.set_prox_model(round_model)
        
        round_samples = 0
        
        for user_idx in user_idxs:
            model.set_params(round_model)   # 'download' global model
            feeder          = data_feeders[user_idx]
            n_samples       = feeder.n_samples
            round_samples   += n_samples
          
            feeder.activate()
            # perform local training
            for k in range(K):
                x, y         = feeder.next_batch(B)
                loss, acc    = model.train_step(x, y)
                train_es[t] += loss
                train_as[t] += acc
            
            feeder.deactivate()
            
            # 'upload' client model
            round_agg    = round_agg + model.get_params_numpy()

        # round_model = round_agg / round_samples
        round_model = round_agg / M
        
        # calculate test statistics
        if t % test_freq == 0:
            model.set_params(round_model)
            test_es[t], test_as[t] = model.test(test_data[0], 
                                                test_data[1], 
                                                test_B)
    
    train_es /= M * K
    train_as /= M * K
    
    return train_es, train_as, test_es, test_as
