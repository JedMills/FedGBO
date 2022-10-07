"""
Pytorch models for use with the FEMNIST, CIFAR100, Shakespeare, StackOverflow FL
simulations. Also contains the NumpyModel class for conveniently containing and 
performing operations on an entire model/set of values at once.
"""
import torch
import numpy as np 
import numbers
import operator
from torch.nn.functional import softmax, log_softmax, kl_div, relu


class FLModel(torch.nn.Module):
    """
    Extension of the pytorch Module class that provides methods for easily 
    extracting/loading model params, training, calculating gradients etc. when
    writing FL loops.
    """
    
    def __init__(self, fedmax_beta=None):
        """ 
        Return a new FL model. The fedmax_beta parameter controls the entropy-
        weighting term as proposed in: "FedMAX: Mitigating Activation Divergence
        for Accurate and Communication-Efficient Federated Learning", Chen et 
        al., ECML PKDD 2021. Experiments displayed as FEdMAX in the FedGBO paper
        use FedAvg with models that have fedmax_beta>0. All other algorithms 
        use fedmax_beta=0. It is possible in principle to try adaptive 
        optimisation algorithms with fedmax_beta>0 but this is left to future 
        work.
        
        Args:
            fedmax_beta: {float} 
        
        """
        super(FLModel, self).__init__()
        self.optim       = None
        self.loss_fn     = None
        self.fedmax_beta = fedmax_beta
        self.device      = torch.device('cpu')
        
    def to(self, device):
        self.device = device
        return super().to(device)

    def set_optim(self, optim):
        """
        Allocates an optimizer for this model to use during training.
        
        Args:
        - optim:    {torch.optim.optimizer}
        """
        self.optim = optim

    def get_params(self):
        """ Returns copies of model parameters as a list of Numpy ndarrays. """
        return [np.copy(p.data.cpu().numpy()) for p in list(self.parameters())]
        
    def get_params_numpy(self):
        """ Returns copy of model parameters as a NumpyModel. """
        return NumpyModel(self.get_params())
        
    def set_params(self, new_params):
        """
        Set all the parameters of this model (values are copied).
        
        Args:
        - new_params: {list, NumpyModel} all ndarrays must be same shape as 
                      model params
        """
        with torch.no_grad():
            for (p, new_p) in zip(self.parameters(), new_params):
                p.copy_(torch.tensor(new_p))
   
    def forward(self, x, return_act_vec=False):
        """
        Return the result of a forward pass of this model. Will return the 
        activation layer is return_act_vec==True (used for FedMAX algorithm).
        
        Args:
        - x:              {torch.tensor} with shape [batch_size, sample_shape]
        - return_act_vec: {bool}
        
        Returns:
            Logits: {torch.tensor} with shape: [batch_size, output_shape] if 
            return_act_vec==False. If return_act_vec==True, will return 
            (logits, activations), activations shape [batch_size, activ_shape].
        """
        raise NotImplementedError()
        
    def calc_acc(self, x, y):
        """
        Return the performance metric (not necessarily accuracy) of the model 
        with inputs x and target y.
        
        Args:
        - x: {torch.tensor} with shape [batch_size, input_shape]
        - y: {torch.tensor} with shape [batch_size, output_shape]
        
        Returns:
            {float} mean performance metric across batch
        """
        raise NotImplementedError()
    
    def train_step(self, x, y):
        """
        Perform a single step of training using samples x and targets y. The 
        set_optim method must have been called with a torch.optim.optimizer 
        before using this method. If self.fedmax_beta>0, then 
        self.train_step_fedmax(..) will be used.
        
        Args:
        - x: {torch.tensor} with shape [batch_size, input_shape]
        - y: {torch.tensor} with shape [batch_size, output_shape]
        
        Returns:
            (float, float) loss and performance metric for given x, y
        """
        if not self.fedmax_beta is None:
            return self.train_step_fedmax(x, y)
        
        logits = self.forward(x)            # forward pass            
        loss   = self.loss_fn(logits, y)
        acc    = self.calc_acc(logits, y)
        self.optim.zero_grad()              # reset model gradient tensors
        loss.backward()
        self.optim.step()
        
        return loss.item(), acc
        
    def train_step_fedmax(self, x, y):
        """
        Training step using the FedMAX objective with maximum-entropy activation
        layer: "FedMAX: Mitigating Activation Divergence for Accurate and
        Communication-Efficient Federated Learning", Chen et al, ECML PKDD 2021.
        Note that a separate function is required (rather than just a general 
        train_step that functions with fedmax_beta=0) because FedMAX is defined 
        only for deep models with an activation layer (and the FedGOB paper uses
        one linear model). 
        
        Args:
        - x: {torch.tensor} with shape [batch_size, input_shape]
        - y: {torch.tensor} with shape [batch_size, output_shape]
        
        Returns:
            (float, float) loss and performance metric for given x, y
        """
        logits, act = self.forward(x, return_act_vec=True)
        zeros       = torch.zeros(act.size()).to(self.device)
        
        # normal loss
        loss1       = self.loss_fn(logits, y)
        
        # maximum-entropy loss from activation layer
        loss2       = kl_div(   log_softmax(act, dim=1), 
                                softmax(zeros, dim=1), 
                                reduction='batchmean')
        loss        = loss1 + (self.fedmax_beta * loss2)
        acc         = self.calc_acc(logits, y)
        
        # perform the SGD step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss1.item(), acc
        
    def calc_batch_grads_numpy(self, x, y):
        """
        Returns the mean gradients of the model given x and y as a NumpyModel.
        
        Args:
        - x: {torch.tensor} with shape [batch_size, input_shape]
        - y: {torch.tensor} with shape [batch_size, output_shape]
        
        Returns:
            {NumpyModel} of mean gradients.
        """
        self.optim.zero_grad()
        err = self.loss_fn(self.forward(x), y)
        err.backward()
        
        return NumpyModel([p.grad.cpu().numpy() for p in self.parameters()])

    def calc_full_grads_numpy(self, feeder, B):
        """
        Return the average gradients over all samples contained in feeder as a
        NumpyModel. Used to calculate unbiased full-batch gradients in 
        Mimelite algorithm.
        
        Args:
        - feeder: {PyTorchDataFeeder} containing samples and labels
        - B:      {int} batch size to use while calculating grads
        
        Returns:
            {NumpyModel} containing average gradients
        """
        n_batches = int(np.ceil(feeder.n_samples / B))
        grads     = None
        
        for b in range(n_batches):
            x, y        = feeder.next_batch(B)
            batch_grads = self.calc_batch_grads_numpy(x, y)
            grads       = batch_grads if grads is None else grads + batch_grads
        
        return grads / n_batches
        
    def test(self, x, y, B):
        """
        Return the average error and performance metric over all samples.
        
        Args:
        - x: {torch.tensor} of shape [num_samples, input_shape]
        - y: {torch.tensor} of shape [num_samples, output_shape]
        - B: {int} batch size to use whilst testing
        
        Returns:
        
        """
        n_batches = int(np.ceil(x.shape[0] / B))
        err       = 0.0       # cumulative error
        acc       = 0.0       # cumulative performance metric
        
        with torch.no_grad():
            for b in range(n_batches):
                logits  = self.forward(x[b*B:(b+1)*B])
                err    += self.loss_fn(logits, y[b*B:(b+1)*B]).item()
                acc    += self.calc_acc(logits, y[b*B:(b+1)*B])
            
        return err/n_batches, acc/n_batches



class Sent140Model(FLModel):
    """
    Convex binary classification model with 5000 inputs and 2 softmax outputs.
    """
    
    def __init__(self):
        super(Sent140Model, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.out     = torch.nn.Linear(5000, 2)
        
    def forward(self, x, return_act_vec=False):
        if return_act_vec:
            raise RuntimeError('Activation vector requested for Sent140 model.')
        
        return self.out(x)
        
    def calc_acc(self, logits, y):
        return (torch.argmax(logits, dim=1) == y).float().mean()



class FEMNISTModel(FLModel):
    """
    A Convolutional (conv) model for use with the FEMNIST dataset, using 
    standard cross entropy loss. Model layers consist of:
    - 3x3 conv, stride 1, 32 filters, ReLU
    - 2x2 max pooling, stride 2
    - 3x3 conv, stride 1, 64 filters, ReLU
    - 2x2 max pooling, stride 2
    - 512 neuron fully connected, ReLU
    - 62 neuron softmax output
    """
    
    def __init__(self, fedmax_beta=None):
        """ Return a new FEMNISTModel. """
        super(FEMNISTModel, self).__init__(fedmax_beta)
        self.loss_fn    = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.flat  = torch.nn.Flatten()
        self.fc1   = torch.nn.Linear(1600, 512)
        self.out   = torch.nn.Linear(512, 62)
        
    def forward(self, x, return_act_vec=False):
        a   = self.pool1(relu(self.conv1(x)))
        b   = self.pool2(relu(self.conv2(a)))
        c   = relu(self.fc1(self.flat(b)))
        out = self.out(c)
        
        if return_act_vec:
            return out, c
        
        return out
        
    def calc_acc(self, logits, y):
        return (torch.argmax(logits, dim=1) == y).float().mean()



class CIFAR100Model(FLModel):
    """
    A Convolutional (conv) model for use with the CIFAR100 dataset, using 
    standard cross entropy loss. Model layers consist of:
    - 3x3 conv, stride 1, 32 filters, ReLU
    - 2x2 max pooling, stride 2
    - 3x3 conv, stride 1, 64 filters, ReLU
    - 2x2 max pooling, stride 2
    - 512 neuron fully connected, ReLU
    - 100 neuron softmax output
    """
    
    def __init__(self, fedmax_beta=None):
        """ Return a new CIFAR100Model. """
        super(CIFAR100Model, self).__init__(fedmax_beta)
        self.loss_fn  = torch.nn.CrossEntropyLoss(reduction='mean')
        
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)   
        self.flat  = torch.nn.Flatten()
        self.fc1   = torch.nn.Linear(2304, 512)
        self.out   = torch.nn.Linear(512, 100)

    def forward(self, x, return_act_vec=False):
        a   = self.pool1(relu(self.conv1(x)))
        b   = self.pool2(relu(self.conv2(a)))
        c   = relu(self.fc1(self.flat(b)))
        out = self.out(c)
        
        if return_act_vec:
            return out, c
        
        return out
        
    def calc_acc(self, logits, y):
        return (torch.argmax(logits, dim=1) == y).float().mean()



class ShakesModel(FLModel):
    """
    A Gated Recurrent Unit (GRU) model to be used with the Shakespeare dataset,
    using stabdard cross entropy loss. Model layers consist of: 
    - (79 to 8) embedding
    - 128 neuron GRU
    - 128 neuron GRU
    - 79 neuron softmax output
    """
    
    def __init__(self, fedmax_beta=None):
        """ Return a new ShakesModel. """
        super(ShakesModel, self).__init__(fedmax_beta)
        self.loss_fn    = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # vocab size is 79
        self.embed      = torch.nn.Embedding(79, 8)
        self.gru        = torch.nn.GRU( input_size=8,
                                        hidden_size=128,
                                        num_layers=2,
                                        batch_first=True)
        self.out        = torch.nn.Linear(128, 79)
        
    def forward(self, x, return_act_vec=False):
        batch_size  = x.size(0)
        a           = self.embed(x)
        b, _        = self.gru(a)
        out = self.out(b[:,-1,:])
        
        if return_act_vec:
            return out, b[:,-1,:]
        
        return out
        
    def calc_acc(self, logits, y):
        return (torch.argmax(logits, dim=1) == y).float().mean()        



class NumpyModel():
    """
    A convenient class for containing an entire model/set of optimiser values. 
    Operations (+, -, *, /, **) can then be done on a whole model/set of values 
    conveniently.
    """
    
    def __init__(self, params):
        """
        Returns a new NumpyModel.
        
        Args:
        - params:  {list} of Numpy ndarrays/pytorch tensors 
        """
        self.params = params
        
    def _op(self, other, f):
        """
        Check type of other and perform function f on values contained in this
        NumpyModel.
        
        Args:
        - other:    {int, float, NumpyArray}
        - f:        number-returning function to apply
        
        Returns:
        The NumpyModel produced as a result of applying f to self and other.
        """
        if isinstance(other, numbers.Number):
            new_params = [f(p, other) for p in self.params]
            
        elif isinstance(other, NumpyModel):
            new_params = [f(p, o) for (p, o) in zip(self.params, other.params)]
            
        else:
            raise ValueError('Incompatible type for op: {}'.format(other))
        
        return NumpyModel(new_params)
        
    def __array_ufunc__(self, *args, **kwargs):
        """
        If an operation between a Numpy scalar/array and a NumpyModel has the 
        numpy value first (e.g. np.float32 * NumpyModel), Numpy will attempt to 
        broadcast the value to the NumpyModel, which acts as an iterable. This 
        results in a NumpyModel *not* being returned from the operation. The 
        explicit exception prevents this from happening silently. To fix, put 
        the NumpyModel first in the operation, e.g. (NumpyModel * np.float32) 
        instead of (np.float32 * NumpyModel), which will call the NumpModel's 
        __mul__, instead of np.float32's.
        """
        raise NotImplementedError(  "Numpy attempted to broadcast to a "
                                  + "NumpyModel. See docstring of "
                                  + "NumpyModel's __array_ufunc__")
        
    def param_sums(self):
        """ List of the sums of all parameters in NumpModel. """
        return [np.sum(p) for p in self.params]
        
    def to_vec(self):
        """ Returns all values in all parameters as a single vector. """
        return np.concatenate([p.flatten() for p in self.params])
        
    def copy(self):
        """ Return a new NumpyModel with copied values. """
        return NumpyModel([np.copy(p) for p in self.params])
        
    def abs(self):
        """ Return a new NumpyModel with all absolute values. """
        return NumpyModel([np.abs(p) for p in self.params])
        
    def zeros_like(self):
        """ Return new NumpyModel with same shape, but filled with 0's. """
        return NumpyModel([np.zeros_like(p) for p in self.params])
        
    def ones_like(self):
        """ Return a new NumpModel with same shapes but filled with 1's. """
        return NumpyModel([np.ones_like(p) for p in self.params])
        
    def __add__(self, other):
        """ Return new NumpyModel resulting from self+other. """
        return self._op(other, operator.add)
        
    def __radd__(self, other):
        """ Return new NumpyModel resulting from other+self.. """
        return self._op(other, operator.add)

    def __sub__(self, other):
        """ Return new NumpyModel resulting from self-other. """
        return self._op(other, operator.sub)
        
    def __mul__(self, other):
        """ Return new NumpyModel resulting from self*other. """
        return self._op(other, operator.mul)
        
    def __rmul__(self, other):
        """ Return new NumpyModel resulting from other*self. """
        return self._op(other, operator.mul)
        
    def __truediv__(self, other):
        """ Return new NumpyModel resulting from self/other. """
        return self._op(other, operator.truediv)
        
    def __pow__(self, other):
        """ Return new NumpyModel resulting from self**other. """
        return self._op(other, operator.pow)
        
    def __getitem__(self, key):
        """
        Get param at index key.
        
        Args:
        - key:  int, index of parameter to retrieve
        
        Returns:
        Numpy ndarray param at index key
        """
        return self.params[key]
        
    def __len__(self):
        """ Returns number of params (Numpy ndarrays) contained in self. """
        return len(self.params)
        
    def __iter__(self):
        """ Returns an iterator over this NumpyModel's parameters. """
        for p in self.params:
            yield p
