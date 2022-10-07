"""
Client- and server-based optimiser classes to be used with the FL algorithms 
defined in fl_algs.py. Client optimisers act on pytorch models, server 
optimisers (for AdaptiveFedOpt) work with NumpyModels exclusively on the CPU.
"""
import torch
from torch.optim.optimizer import Optimizer
import numpy as np 
from models import NumpyModel



class FixedStatsSGDm(Optimizer):
    """ SGDm optimiser with fixed statistics. """

    def __init__(self, params, lr, beta, device):
        """
        Returns a new FixedStatsSGDm optimiser.
        
        Args:
        - params:   {list} of params from a pytorch model
        - lr:       {float} learning rate 
        - beta:     {float} momentum value, 0 <= beta < 1.0
        - device:   {torch.device} to place optimizer values on 
        """
        defaults    = dict(lr=lr, beta=beta)
        super(FixedStatsSGDm, self).__init__(params, defaults)
        self.lr     = lr
        self.beta   = beta
        self.device = device
        self._init_m()
        
    def _init_m(self):
        """ Initialise the momentum terms of the optimizer with zeros. """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['m'] = torch.zeros_like(  p, 
                                                device=self.device, 
                                                dtype=torch.float32)

    def step(self, closure=None):
        """
        Perform one step of momentum-SGD, without updating statistics. U step in
        Table 1.
        
        Args:
        - closure: {callable} see torch.optim documentation
        
        Returns: {None, float} see torch.optim documentation
        """
        loss = None 
        if closure is not None:
            loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data         # parameter gradient
                m = self.state[p]['m']  # current momentum value
                
                p.data.sub_(self.lr * (self.beta * m + (1 - self.beta) * g))
        
        return loss
        
    def update_moments(self, grads):
        """
        Update momentum terms, T step in Table 1.
        
        Args:
        - grads: {NumpyModel} containing gradients to update with.
        """
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                m = self.state[p]['m']
                # need to convert np.ndarray to torch.tensor
                g = torch.tensor(   grads[i], 
                                    dtype=torch.float32, 
                                    device=self.device)
                
                self.state[p]['m'] = (self.beta * m) + ((1 - self.beta) * g)
                i += 1
    
    def set_params(self, moments):
        """
        Set the momentum parameters of the optimiser.
        
        Args:
        - moments: {NumpyModel} containing new moments.
        """
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # need to convert np.ndarray to torch.tensor
                m = torch.tensor(   moments[i], 
                                    dtype=torch.float32, 
                                    device=self.device)
                
                self.state[p]['m'] = m
                i += 1

    def get_params_numpy(self):
        """ Return momentum values of optimizer as a NumpyModel. """
        params = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:  # ignore gradient-less variables
                    continue
                
                params.append(np.copy(self.state[p]['m'].cpu().data.numpy()))

        return NumpyModel(params)

    def inv_grads(self, x_t, x_t1, K):
        """
        Return the average gradient computed on clients during a round of 
        FedGBO. I step in Table 1.
        
        Args:
        - x_t:  {NumpyModel} model at start of round
        - x_t1: {NumpyModel} average model uploaded by workers at end of round
        - K:    {int} number of SGDm steps that workers performed in round
        
        Returns: {NumpyModel} of average gradients, as per Table 1.
        """
        ms = self.get_params_numpy()
        a = (x_t - x_t1) / (self.lr * K)
        b = self.beta * ms
        
        return (a - b) / (1 - self.beta)



class FixedStatsRMSProp(Optimizer):
    """ RMSProp optimiser with fixed statistics. """

    def __init__(self, params, lr, beta, epsilon, device):
        """
        Returns a new FixedStatsRMSProp optimizer.
        
        Args:
        - params:   {list} of params from a pytorch model
        - lr:       {float} learning rate 
        - beta:     {float} momentum value, 0 <= beta < 1.0
        - epsilon:  {float} stability valu to avoid division by 0
        - device:   {torch.device} to place optimizer values on 
        """
        defaults        = dict(lr=lr, beta=beta, epsilon=epsilon)
        super(FixedStatsRMSProp, self).__init__(params, defaults)
        self.device     = device
        self.lr         = lr
        self.beta       = beta
        self.epsilon    = epsilon
        self._init_v()
        
    def _init_v(self):
        """ Initialise the momentum terms of the optimizer with zeros. """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['v'] = torch.ones_like(   p, 
                                                device=self.device, 
                                                dtype=torch.float32)

    def step(self, closure=None):
        """
        Perform one step of RMSProp, without updating the moment parameters. U 
        step in Table 1.
        
        Args:
        - closure: {callable} see torch.optim documentation
                
        Returns: {None, float} see torch.optim documentation
        """
        loss = None 
        if closure is not None:
            loss = closure()
                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad.data
                v = self.state[p]['v']
                
                p.data.sub_((self.lr * g) / (torch.sqrt(v) + self.epsilon))
        
        return loss

    def update_moments(self, grads):
        """
        Update momentum terms. T step in Table 1.
        
        Args:
        - grads: {NumpyModel} containing gradients to update with.
        """
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                v = self.state[p]['v']
                
                g = torch.tensor(   grads[i], 
                                    dtype=torch.float32, 
                                    device=self.device)
                b = self.beta
                
                self.state[p]['v'] = (b * v) + ((1 - b) * torch.square(g))
                i += 1
    
    def get_params_numpy(self):
        """ Return momentum values of optimizer as a NumpyModel. """
        params = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                params.append(np.copy(self.state[p]['v'].cpu().data.numpy()))

        return NumpyModel(params)
    
    def set_params(self, grads):
        """
        Set the momentum parameters of the optimiser.
        
        Args:
        - moments: {NumpyModel} containing new moments.
        """
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                v = torch.tensor(   grads[i], 
                                    dtype=torch.float32, 
                                    device=self.device)
                
                self.state[p]['v'] = torch.square(v)
                i += 1
 
    def inv_grads(self, x_t, x_t1, K):
        """
        Return the average gradient computed on clients during a round of 
        FedGBO. I step in Table 1.
        
        Args:
        - x_t:  {NumpyModel} model at start of round
        - x_t1: {NumpyModel} average model uploaded by workers at end of round
        - K:    {int} number of RMSProp steps that workers performed in round
        
        Returns: {NumpyModel} of average gradients, as per Table 1.
        """
        vs = self.get_params_numpy()
        
        return (x_t - x_t1) * (vs**0.5 + self.epsilon) / (self.lr * K)



class FixedStatsAdam(Optimizer):
    """
    Adam optimiser that does not update statistics during the training loop.
    """

    def __init__(self, params, lr, beta1, beta2, epsilon, device):
        """
        Returns a new FixedStatsAdam optimizer.
        
        Args:
        - params:   {list} of params from a pytorch model
        - lr:       {float} learning rate 
        - beta1:    {float} momentum value, 0 <= beta1 < 1.0
        - beta2:    {float} momentum value, 0 <= beta2 < 1.0
        - epsilon:  {float} stability parameter to avoid division by 0
        - device:   {torch.device} to place optimizer values on 
        """
        defaults        = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(FixedStatsAdam, self).__init__(params, defaults)
        self.device     = device
        self.lr         = lr
        self.beta1      = beta1
        self.beta2      = beta2
        self.epsilon    = epsilon
        self._init_m_v()
    
    def get_params_numpy(self, together=True):
        """
        Return momentum values of optimizer as a NumpyModel. If togther=True, 
        then will return a single NumpyModel with parameters in the order 
        [m0,v0,m1,v1,...]. Otherwise will reutrn two NumpyModels, the first with
        all the m's, the second with all the v's.
        """
        if together:
            params = []
        else:
            ms = []
            vs = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                m = np.copy(self.state[p]['m'].cpu().data.numpy())
                v = np.copy(self.state[p]['v'].cpu().data.numpy())

                if together:
                    params.append(m)
                    params.append(v)
                else:
                    ms.append(m)
                    vs.append(v)

        if together:
            return NumpyModel(params)
        else:
            return NumpyModel(ms), NumpyModel(vs)
        
    def _init_m_v(self):
        """
        Initialise the momentum terms of the optimizer with zeros.
        """
        for group in self.param_groups:
            for p in group['params']:
                state       = self.state[p]
                state['m']  = torch.zeros_like( p, 
                                                device=self.device, 
                                                dtype=torch.float32)
                state['v']  = torch.ones_like(  p, 
                                                device=self.device, 
                                                dtype=torch.float32)
        
    def step(self, closure=None):
        """
        Perform one step of Adam without updating moments. U step in Table 1.
        
        Args:
        - closure: {callable} see torch.optim documentation
        
        Returns: {None, float} see torch.optim documentation
        """
        loss = None 
        if closure is not None:
            loss = closure()
        
        lr  = self.lr
        b1  = self.beta1
        eps = self.epsilon
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad.data 
                m = self.state[p]['m']
                v = self.state[p]['v']
                
                # fixed-momentum update rule as per paper
                p.data.sub_(lr * (b1*m + (1 - b1)*g) / (torch.sqrt(v) + eps))
                    
        return loss
    
    def update_moments(self, grads):
        """
        Update momentum terms. T step in Table 1.
        
        Args:
        - grads: {NumpyModel} containing gradients to update with.
        """
        i   = 0
        b1  = self.beta1
        b2  = self.beta2
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                m = self.state[p]['m']
                v = self.state[p]['v']
                g = torch.tensor(   grads[i], 
                                    dtype=torch.float32, 
                                    device=self.device)
                
                # Adam m and v update rule
                self.state[p]['m'] = (b1*m) + (1-b1) * g
                self.state[p]['v'] = (b2*v) + (1-b2) * g * g
                i += 1
                
                
    def inv_grads(self, x_t, x_t1, K):
        """
        Return the average gradient computed on clients during a round of 
        FedGBO. I step in Table 1.
        
        Args:
        - x_t:  {NumpyModel} model at start of round
        - x_t1: {NumpyModel} average model uploaded by workers at end of round
        - K:    {int} number of Adam steps that workers performed in round
        
        Returns: {NumpyModel} of average gradients as per Table 1.
        """
        ms, vs  = self.get_params_numpy(together=False)
        a       = (x_t - x_t1) * (vs**0.5 + self.epsilon) / (self.lr * K)
        b       = (self.beta1 * ms)
        
        return (a - b) / (1 - self.beta1)



class ServerOptimizer():
    """
    Server optimiser class for use with AdaptiveFedOpt algorithm.
    """
    
    def apply_gradients(self, model, grads):
        """
        Update the global model using gradients and update internal moments.
        
        Args:
        - model: {NumpyModel} to update
        - grads: {NumpyModel} round psuedogradient
        """
        raise NotImplementedError()



class ServerSGD(ServerOptimizer):
    """
    SGD optimizer (FedSGD) to be used on the FL server, as per the "Adaptive
    Federated Optimization", Reddi et al, arXiv 2020. FedSGD with lr = 1.0 is 
    the same as the original FedAvg.
    """
    
    def __init__(self, lr):
        """
        Returns a new optimizer for use in Federated experiments.
        
        Args:
        - lr:       {float} learning rate 
        """
        self.lr     = lr
        
    def apply_gradients(self, model, grads):
        """
        Return a new NumpyModel that is the result of one step of SGD 
        optimization using the given current model and gradients.
        
        Args:
        - model: {NumpyModel} current model
        - grads: {NumpyModel} gradients to apply
        
        Returns: {NumpyModel}
        """
        return model - self.lr * grads



class ServerSGDm(ServerOptimizer):
    """
    Momentum-SGD optimizer (FedSGDm) to be used on the FL server, as per the 
    "Adaptive Federated Optimization", Reddi et al, arXiv 2020. Weighted 
    heavy-ball momentum.
    """
    
    def __init__(self, params, lr, beta):
        """
        Returns a new optimizer for use in Federated experiments.
        
        Args:
        - params:   {NumpyModel} of params from a pytorch model
        - lr:       {float} learning rate 
        - beta:     {float} momentum value, 0 <= beta < 1.0
        """
        self.m      = params.zeros_like()
        self.lr     = lr
        self.beta   = beta
        
    def apply_gradients(self, model, grads):
        """
        Return a new NumpyModel that is the result of one step of SGD 
        optimization using the given current model and gradients.
        
        Args:
        - model: {NumpyModel} current model
        - grads: {NumpyModel} gradients to apply
        
        Returns: {NumpyModel}
        """
        self.m  = (self.beta * self.m) + (1 - self.beta) * grads
        
        return model - self.lr * self.m



class ServerRMSProp(ServerOptimizer):
    """
    RMSProp optimizer (FedRMSProp) to be used on the FL server, as per the 
    "Adaptive Federated Optimization", Reddi et al, arXiv 2020.
    """
    
    def __init__(self, params, lr, beta, epsilon):
        """
        Returns a new optimizer for use in Federated experiments.
        
        Args:
        - params:   {NumpyModel} of params from a pytorch model
        - lr:       {float} learning rate 
        - beta:     {float} momentum value, 0 <= beta < 1.0
        - epsilon:  {float} stability parameter to avoid division by 0
        """
        self.v      = params.ones_like()
        self.lr     = lr
        self.b      = beta
        self.eps    = epsilon
        
    def apply_gradients(self, model, grads):
        """
        Return a new NumpyModel that is the result of one step of RMSProp 
        optimization using the given current model and gradients. Will update 
        the internal v parameters of the optimizer.
        
        Args:
        - model: {NumpyModel} current model
        - grads: {NumpyModel} gradients to apply
        
        Returns: {NumpyModel}
        """
        self.v = (self.b * self.v) + (1 - self.b) * (grads ** 2)
        
        return model - (self.lr * grads) / (self.v**0.5 + self.eps)



class ServerAdam(ServerOptimizer):
    """
    Adam optimizer (FedAdam) to be used on the FL server, as per the "Adaptive
    Federated Optimization", Reddi et al, arXiv 2020.
    """
    
    def __init__(self, params, lr, beta1, beta2, epsilon):
        """
        Returns a new optimizer for use in Federated experiments.
        
        Args:
        - params:   {NumpyModel} of params from a pytorch model
        - lr:       {float} learning rate 
        - beta1:    {float} momentum value, 0 <= beta1 < 1.0
        - beta2:    {float} momentum value, 0 <= beta2 < 1.0
        - epsilon:  {float} stability parameter to avoid division by 0
        """
        self.m      = params.zeros_like()
        self.v      = params.ones_like()
        self.t      = 0
        self.lr     = lr
        self.b1     = beta1
        self.b2     = beta2
        self.eps    = epsilon
        
    def apply_gradients(self, model, grads):
        """
        Return a new NumpyModel that is the result of one step of Adam 
        optimization using the given current model and gradients. Will update 
        the internal m/v parameters of the optimizer. This implementation uses a
        fixed learning rate, rather than the scaled learning rate presented in 
        the original Adam paper.
        
        Args:
        - model: {NumpyModel} current model
        - grads: {NumpyModel} gradients to apply
        
        Returns: {NumpyModel}
        """
        self.t  += 1
        self.m  = (self.b1 * self.m) + (1 - self.b1) * grads
        self.v  = (self.b2 * self.v) + (1 - self.b2) * (grads ** 2)
        
        # the Google paper does not use the scaled learning rate from the
        # original Adam paper:
        # lr      = self.lr * np.sqrt(1 - self.b2**self.t)/(1 - self.b1**self.t)
        # but rather a fixed learning rate:
        
        return model - self.lr * self.m / (self.v**0.5 + self.eps)



class MFLSGDm(Optimizer):
    """
    SGDm optimiser that *does* update moments during the model-update step, to
    be used with the MFL algorithm.
    """

    def __init__(self, params, lr, beta, device):
        """
        Return a new MFLSGDm optimiser.
        
        Args:
        - params: {list} list of params returned from a pytorch model
        - lr:     {float} learning rate
        - beta:   {float} decay parameter
        - device: {torch.device} where to place optimiser
        """
        defaults    = dict(lr=lr, beta=beta)
        super(MFLSGDm, self).__init__(params, defaults)
        self.device = device
        self.lr     = lr
        self.beta   = beta
        self._init_m()

    def _init_m(self):
        """ Initialise momentum tensors with zeros. """
        for group in self.param_groups:
            for p in group['params']:
                state        = self.state[p]
                state['m']   = torch.zeros_like(p, 
                                                device=self.device, 
                                                dtype=p.dtype)

    def set_momentum(self, m):
        """
        Set the momentum parameters of the optimiser.
        
        Args:
        - m: {NumpyModel} moments to set
        """
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['m'] = torch.tensor(  m[i], 
                                                    dtype=torch.float32, 
                                                    device=self.device)
                i += 1
    
    def get_momentum(self):
        """
        Returns all momentum parameters as a NumpyModel.
        """
        params = []
        for group in self.param_groups:
            for p in group['params']:
                tensor = self.state[p]['m'].to(torch.device('cpu'))
                params.append(tensor.numpy())
        
        return NumpyModel(params)
    
    def step(self, closure=None):
        """
        Perform one step of SGDm, as normal (i.e. update moments, then perform 
        step of SGDm on model parameters).
        
        Args:
        - closure: {callable} see torch.optim documentation
        
        Returns: {None, float} see torch.optim documentation
        """
        loss = None 
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                m = (self.beta * self.state[p]['m']) + ((1 - self.beta) * g)
                p.data.sub_(lr * m)
                self.state[p]['m'] = m
        
        return loss



class MFLRMSProp(Optimizer):
    """
    RMSProp optimiser that *does* update moments during the model-update step, 
    to be used with the MFL algorithm.
    """

    def __init__(self, params, lr, beta, epsilon, device):
        """
        Return a new MFLRMSProp optimiser.
        
        Args:
        - params: {list} list of params returned from a pytorch model
        - lr:     {float} learning rate
        - beta:   {float} decay parameter
        - epsilon {float} stability parameter
        - device: {torch.device} where to place optimiser
        """
        defaults        = dict(lr=lr, beta=beta, epsilon=epsilon)
        super(MFLRMSProp, self).__init__(params, defaults)
        self.device = device
        self.lr         = lr
        self.beta       = beta
        self.epsilon    = epsilon
        self._init_v()

    def _init_v(self):
        """ Initialise adaptivity tensors with ones. """
        for group in self.param_groups:
            for p in group['params']:
                state        = self.state[p]
                state['v']   = torch.ones_like( p, 
                                                device=self.device, 
                                                dtype=p.dtype)

    def set_momentum(self, v):
        """
        Set the momentum parameters of the optimiser.
        
        Args:
        - v: {NumpyModel} moments to set
        """
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['v'] = torch.tensor(  v[i], 
                                                    dtype=torch.float32, 
                                                    device=self.device)
                i += 1
    
    def get_momentum(self):
        """ Returns all adaptivity parameters as a NumpyModel. """
        params = []
        for group in self.param_groups:
            for p in group['params']:
                tensor = self.state[p]['v'].to(torch.device('cpu'))
                params.append(tensor.numpy())
        
        return NumpyModel(params)
    
    def step(self, closure=None):
        """
        Perform one step of RMSProp, as normal (i.e. update moments, then 
        perform step of RMSProp on model parameters).
        
        Args:
        - closure: {callable} see torch.optim documentation
        
        Returns: {None, float} see torch.optim documentation
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                g   = p.grad.data
                v   = self.state[p]['v']
                v1  = (self.beta * v) + ((1 - self.beta) * torch.square(g))
                
                p.data.sub_(lr * g / (torch.sqrt(v1) + self.epsilon))
                self.state[p]['v'] = v1
        
        return loss



class MFLAdam(Optimizer):
    """
    Adam optimiser that *does* update moments during the model-update step, to
    be used with the MFL algorithm.
    """

    def __init__(self, params, lr, beta1, beta2, epsilon, device):
        """
        Return a new MFLAdam optimiser.
        
        Args:
        - params: {list} list of params returned from a pytorch model
        - lr:     {float} learning rate
        - beta1:  {float} m decay parameter
        - beta2:  {float} v decay parameter
        - epsilon {float} stability parameter
        - device: {torch.device} where to place optimiser
        """
        defaults        = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(MFLAdam, self).__init__(params, defaults)
        self.device = device
        self.lr         = lr
        self.beta1      = beta1
        self.beta2      = beta2
        self.epsilon    = epsilon
        self._init_mv()

    def _init_mv(self):
        """ Initialise momentum with 0's and adaptivity with 1's."""
        for group in self.param_groups:
            for p in group['params']:
                state        = self.state[p]
                state['m']   = torch.zeros_like(p, 
                                                device=self.device, 
                                                dtype=p.dtype)
                state['v']   = torch.ones_like( p, 
                                                device=self.device, 
                                                dtype=p.dtype)

    def set_momentum(self, mv):
        """
        Set the momentum parameters of the optimiser. Passed mv param should 
        have twice the number of paramters as the model, and values should be 
        in the order [m0, v0, m1, v1, ...,].
        
        Args:
        - mv: {NumpyModel} moments to set
        """
        i = 0
        for m_type in ['m', 'v']:
            for group in self.param_groups:
                for p in group['params']:
                    self.state[p][m_type] = torch.tensor(mv[i], 
                                                         dtype=torch.float32, 
                                                         device=self.device)
                    i += 1
    
    def get_momentum(self):
        """
        Returns all momentum parameters as a NumpyModel, in the order 
        [m0, v0, m1, v1, ...,].
        """
        params = []
        for m_type in ['m', 'v']:
            for group in self.param_groups:
                for p in group['params']:
                    tensor = self.state[p][m_type].to(torch.device('cpu'))
                    params.append(tensor.numpy())
        
        return NumpyModel(params)
    
    def step(self, closure=None):
        """
        Perform one step of Adam, as normal (i.e. update moments, then perform 
        step of Adam on model parameters). Uses fixed learing rate, not the 
        scaled lr from the original Adam paper.
        
        Args:
        - closure: {callable} see torch.optim documentation
        
        Returns: {None, float} see torch.optim documentation
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']    # fixed lr
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # calculate the adam step
                g   = p.grad.data
                v   = self.state[p]['v']
                m   = self.state[p]['m']
                m1  = (self.beta1 * m) + ((1 - self.beta1) * g)
                v1  = (self.beta2 * v) + ((1 - self.beta2) * torch.square(g))
                
                p.data.sub_(lr * m1 / (torch.sqrt(v1) + self.epsilon))
                
                self.state[p]['m'] = m1
                self.state[p]['v'] = v1
        
        return loss



class FedProxOptim(Optimizer):
    """
    Implementation of the FedProx optimizer from the paper: "Federated 
    Optimization in Heterogeneous Networks", Li et al, MLSys 2020. 
    """
    
    def __init__(self, params, lr, mu, device):
        """
        Returns a new optimizer for use in FedProx Federated experiments.
        
        Args:
        - params:   {list} of params from a pytorch model
        - lr:       {float} learning rate 
        - mu:       {float} proximal term, 0 <= beta1 < 1.0
        - device:   {torch.device} to place optimizer values on 
        """
        defaults        = dict(lr=lr, mu=mu)
        super(FedProxOptim, self).__init__(params, defaults)
        self.device     = device
        self.init_lr    = lr
        self.lr         = lr
        self.mu         = mu
        self._init_prox_model()
        
    def _init_prox_model(self):
        """
        Initialise the variable that will contain the global model, $\omega^t$ 
        from Algorithm 2 of the Li paper.
        """
        for group in self.param_groups:
            for p in group['params']:
                state           = self.state[p]
                state['prox']   = torch.zeros_like( p, 
                                                    device=self.device, 
                                                    dtype=p.dtype)
        
    def set_prox_model(self, prox):
        """
        Set the proximal (global) model $\omega^t$ that is part of the client 
        loss function in Eqn 2 of Li paper.
        
        Args:
        - prox: {NumpyModel} contains the proximal terms
        """
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['prox'] = torch.tensor(   prox[i], 
                                                        dtype=torch.float32, 
                                                        device=self.device)
                i += 1

    def step(self, closure=None):
        """
        Perform one step of FedProxSGD as per Li et al 2020: 
        $ \omega \gets \omega - \eta (\delta F -  \mu (\omega - \omega^t))$
        """
        loss = None 
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                prox = self.state[p]['prox']
                # update rule
                p.data.sub_(lr * (p.grad.data + self.mu * (p.data - prox)))
                    
        return loss
