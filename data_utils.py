"""
Functions for loading FEMNIST, CIFAR100, Shakespeare, StackOverflow datasets.
PyTorchDataFeeder class for conveniently containing a single worker's dataset,
and retrieving a stream of data batches from it. Also has some data utility 
functions.
"""
import torch
import pickle
import numpy as np
import os
import json
import h5py
import scipy.sparse
from timeit import default_timer as time


    
class PyTorchDataFeeder():
    """
    Used to easily contain the samples of a FL worker. Can hold the samples on 
    the GPU, and produce an endless stream of randomly drawn samples with a 
    given transformation applied.
    """

    def __init__(   self, x, x_dtype, y, y_dtype, device, 
                    cast_device=None, transform=None):
        """
        Return a new PyTorchDataFeeder with copies of x and y as torch.tensors.
        Data will be stored on device. If x_dtype or y_dtype are the string 
        'long' then these tensors will be cast to the torch long dtype (used
        typically when pytorch models are expecting integer values). If 
        cast_device is passed, the data returned by next_batch will be cast to 
        this device. Doing so allows data held on the CPU to be easily fed to a 
        model sitting on the GPU, if for example all data won't fit in GPU but 
        the model is to be optimised on the GPU. If transform is passed, the 
        samples are transformed by this function before being returned by 
        next_batch(...).
       
        Args:
        - x:            {numpy.ndarray, scipy.sparse.coo_matrix} of samples
        - x_dtype:      {torch.dtype, 'long'} that x will be
        - y:            {numpy.ndarray, scipy.sparse.coo_matrix} of targets
        - y_dtype:      {torch.dtype, 'long'} that y will be 
        - device:       {torch.device} that x and y will sit on
        - cast_device:  {torch.device} next_batch returned data will be on here
        - transform:    {callable} applied by next_batch
        """
        self.x, self.x_sparse = self._matrix_type_to_tensor(x, x_dtype, device)
        self.y, self.y_sparse = self._matrix_type_to_tensor(y, y_dtype, device)
        self.idx              = 0
        self.n_samples        = x.shape[0]
        self.cast_device      = device if cast_device is None else cast_device
        self.transform        = transform
        self.active           = False
        self.activate()
        self._shuffle_data()
        self.deactivate()
        
    def _matrix_type_to_tensor(self, matrix, dtype, device):
        """
        Converts a scipy.sparse.coo_matrix or a numpy.ndarray into a 
        torch.sparse_coo_tensor or torch.tensor. 
        
        Args:
        - matrix:   {scipy.sparse.coo_matrix or np.ndarray} to convert
        - dtype:    {torch.dtype} of the tensor to make 
        - device:   {torch.device} where the tensor should be placed
        
        Returns: (tensor, is_sparse)
        - tensor:       {torch.sparse_coo_tensor or torch.tensor}
        - is_sparse:    {bool} True if returning a torch.sparse_coo_tensor
        """
        tensor_type = torch.int32 if dtype == 'long' else dtype 

        # sparse tensors, used for Sent140 dataset
        if type(matrix) == scipy.sparse.coo_matrix:
            is_sparse = True
            idxs = np.vstack((matrix.row, matrix.col))
            tensor = torch.sparse_coo_tensor(   idxs, 
                                                matrix.data, 
                                                matrix.shape, 
                                                device=device, 
                                                dtype=tensor_type)
        
        elif type(matrix) == np.ndarray:
            is_sparse = False
            tensor = torch.tensor(  matrix, 
                                    device=device, 
                                    dtype=tensor_type)
        else:
            raise TypeError('Only np.ndarray/scipy.sparse.coo_matrix accepted.')

        tensor = tensor.long() if dtype == 'long' else tensor
        
        return tensor, is_sparse
    
    def activate(self):
        """
        Activate this PyTorchDataFeeder to allow .next_batch(...) to be called. 
        Will turn torch.sparse_coo_tensors into dense representations ready for 
        training.
        """
        self.active = True
        
        if self.x_sparse:
            self.all_x_data = self.x.to_dense().to(self.cast_device)
        else:
            self.all_x_data = self.x.to(self.cast_device)
            
        if self.y_sparse:
            self.all_y_data = self.y.to_dense().to(self.cast_device)
        else:
            self.all_y_data = self.y.to(self.cast_device)
    
    def deactivate(self):
        """
        Deactivate this PyTorchDataFeeder to disallow .next_batch(...). Will 
        deallocate the dense matrices created by .activate() to save memory.
        """
        self.active = False
        self.all_x_data = None
        self.all_y_data = None       
        
    def _shuffle_data(self):
        """ Co-shuffle the x and y data. """
        """
        if not self.active:
            raise RuntimeError('._shuffle_data() called when not active.')
        
        ord             = torch.randperm(self.n_samples)
        self.all_x_data = self.all_x_data[ord]
        self.all_y_data = self.all_y_data[ord]
        
        if self.x_sparse:
            self.x = self.all_x_data.to_sparse()
        else:
            self.x = self.all_x_data
            
        if self.y_sparse:
            self.y = self.all_y_data.to_sparse()
        else:
            self.y = self.all_y_data
        """
        self.data_idxs = torch.randperm(self.n_samples)
        
    def next_batch(self, B):
        """
        Return a batch of randomly ordered data from this dataset. If B=-1, 
        return all the data as one big batch. If self.cast_device != None, 
        returned data will be located on self.cast_device(), else self.deivce().
        If self.transform != None, self.transform will be applied to the data 
        before being returned.
        
        Args:
        - B: {int} size of batch to return.
        """
        if not self.active:
            raise RuntimeError('next_batch(...) called when feeder not active.')
        
        if B == -1:                 # return all data as big batch
            x = self.all_x_data
            y = self.all_y_data
            self._shuffle_data()
        
        elif self.idx + B > self.n_samples: # need to wraparound dataset 
            extra = (self.idx + B) - self.n_samples
            # x     = torch.cat(( self.all_x_data[self.idx:], 
            #                     self.all_x_data[:extra]))
            x     = torch.cat(( self.all_x_data[self.data_idxs[self.idx:]], 
                                self.all_x_data[self.data_idxs[:extra]]))
            # y     = torch.cat(( self.all_y_data[self.idx:], 
            #                     self.all_y_data[:extra]))
            y     = torch.cat(( self.all_y_data[self.data_idxs[self.idx:]], 
                                self.all_y_data[self.data_idxs[:extra]]))
            self._shuffle_data()
            self.idx    = 0
            
        else:
            # x           = self.all_x_data[self.idx:self.idx+B]
            x = self.all_x_data[self.data_idxs[self.idx:self.idx+B]]
            y = self.all_y_data[self.data_idxs[self.idx:self.idx+B]]
            # y           = self.all_y_data[self.idx:self.idx+B]
            self.idx    += B
    
        if not self.transform is None:
            x = self.transform(x)

        return x, y



def load_sent140(train_f, test_f, W):
    """
    Load preprocesses sentiment 140 data as bag-of-word vectors. Data should be 
    processed as described in "Faster Federated Learning with Decaying Number of 
    Local SGD Steps", and stored as lists of scipy.sparse.coo_matrix's. 
    
    Args:
    - train_f:  {str} path to training data file
    - test_f:   {str} path to test data file 
    - W:        {int} number of workers' worth of data to load

    Returns: (train_x, train_y), (test_x, test_y)
    - train_x: {list} of scipy.sparse.coo_matrix
    - train_y: {list} of np.ndarrays
    - test_x:  {np.ndarray}
    - test_y:  {np.ndarray}
    """
    if W > 21876:
        raise ValueError('Sent140 dataset has max 21876 users.')
    
    with open(train_f, 'rb') as f:
        train_x, train_y = pickle.load(f)
    with open(test_f, 'rb') as f:
        test_x, test_y = pickle.load(f)
        
    test_x = scipy.sparse.vstack(test_x[:W]).toarray()
    test_y = np.concatenate(test_y[:W])
    
    return (train_x[:W], train_y[:W]), (test_x, test_y)



def load_femnist(train_dir, test_dir, W):
    """
    Load the FEMNIST data contained in train_dir and test_dir. These dirs should
    contain only .json files that have been produced by the LEAF 
    (https://leaf.cmu.edu/) preprocessing tool. Will load W workers' worth of 
    data from these files.
    
    Args:
    - train_dir:    {str} path to training data folder
    - test_dir:     {str} path to test data folder
    - W:            {int} number of workers' worth of data to load
    
    Returns: (x_train, y_train), (x_test, y_test)
    - x_train: {list} of np.ndarrays
    - y_train: {list} of np.ndarrays
    - x_test:  {np.ndarray}
    - y_test:  {np.ndarray}
    """
    train_fnames    = sorted([train_dir+'/'+f for f in os.listdir(train_dir)])
    test_fnames     = sorted([test_dir+'/'+f for f in os.listdir(test_dir)])
    # each .json file contains data for 100 workers
    n_files         = int(np.ceil(W / 100))
    
    x_train = []
    y_train = []
    x_test  = []
    y_test  = []

    tot_w = 0
    for n in range(n_files):
        with open(train_fnames[n], 'r') as f:
            train = json.load(f)
        with open(test_fnames[n], 'r') as f:
            test = json.load(f)
        
        keys = sorted(train['user_data'].keys())
    
        for key in keys:
            # (1.0 - data) so images are white on black like classic MNIST
            x = 1.0 - np.array(train['user_data'][key]['x'], dtype=np.float32)
            x = x.reshape((x.shape[0], 28, 28, 1))
            # transpose (rather than reshape) required to get actual order of 
            # data in ndarray to change. If reshape is used, when data is 
            # passed to a torchvision.transform, then the resulting images come
            # out incorrectly.
            x = np.transpose(x, (0, 3, 1, 2))
            y = np.array(train['user_data'][key]['y'])
            
            x_train.append(x)
            y_train.append(y)
            
            x = 1.0 - np.array(test['user_data'][key]['x'], dtype=np.float32)
            x = x.reshape((x.shape[0], 28, 28, 1))
            x = np.transpose(x, (0, 3, 1, 2))
            y = np.array(test['user_data'][key]['y'])
            
            x_test.append(x)
            y_test.append(y)
            
            tot_w += 1
            
            if tot_w == W:
                break
                
    assert tot_w == W, 'Could not load enough workers from files.'
    
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    return (x_train, y_train), (x_test, y_test)



def load_shakes(train_fname, test_fname, W):
    """
    Load the Shakespeare data contained in train_fname and test_fname. These 
    files should be .json files that have been produced by the LEAF 
    (https://leaf.cmu.edu/) preprocessing tool. Will load W workers' worth of 
    data from these files.
    
    Args:
    - train_fname: {str} path to training data file
    - test_fname:  {str} path to test data file
    - W:           {int} number of workers' worth of data to load
    
    Returns: (train_x, train_y), (test_x, test_y)
    - train_x: {list} of np.ndarrays
    - train_y: {list} of np.ndarrays
    - test_x:  {np.ndarray}
    - test_y:  {np.ndarray}
    """
    if W > 660:
        raise ValueError('Shakespeare dataset has max 660 users.')
    
    # all the symbols in the shakespeare text
    vocab       = ' !"&\'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}'
    chr_to_int  = {c:i for (i, c) in enumerate(vocab)}
    
    with open(train_fname, 'r') as f:
        train = json.load(f)
    with open(test_fname, 'r') as f:
        test = json.load(f)
    
    train_x, train_y, test_x, test_y = [], [], [], []
    
    users = sorted(train['users'])
    
    # load W worth of character text, and convert to ints
    for w in range(W):
        for (source, dest) in zip([train, test], [train_x, test_x]):
            x = source['user_data'][users[w]]['x']
            x = [[chr_to_int[c] for c in sentence] for sentence in x]
            dest.append(np.array(x, dtype=np.int32))
        
        for (source, dest) in zip([train, test], [train_y, test_y]):
            y = source['user_data'][users[w]]['y']
            y = [chr_to_int[c] for c in y]
            dest.append(np.array(y, dtype=np.int32))
        
    test_x = np.concatenate(test_x)
    test_y = np.concatenate(test_y)
    
    return (train_x, train_y), (test_x, test_y)



def normalise(x, means, stds):
    """
    Centers images in x using means, and divides by stds. 
    
    Args:
    - x:        {np.ndarray} of shape [n_imgs, 3, img_h, img_w]
    - means:    {list} of floats containing per-channel means
    - stds:     {list} of floats containing per-channel standard deviations
    
    Returns:
    {np.ndarray} of x, normalised per channel. 
    """
    x1 = np.copy(x)
    for i in range(3):
        x1[:,i,:,:] = (x1[:,i,:,:] - means[i]) / stds[i]

    return x1



def load_cifar100(train_fname, test_fname, W):
    """
    Load the CIFAR100 data contained in train_fname and test_fname. These 
    files should be .h5py files downloaded by tensorflow federated when using
    the CIFAR100 dataset function.
    
    Args:
    - train_fname: {str} path to training data file
    - test_fname:  {str} path to test data file
    - W:           {int} number of workers' worth of data to load
    
    Returns: (train_x, train_y), (test_x, test_y)
    - train_imgs:   {list} of np.ndarrays
    - train_labels: {list} of np.ndarrays
    - test_imgs:    {np.ndarray}
    - test_labels:  {np.ndarray}
    """
    with h5py.File(train_fname, 'r') as f:
        keys            = sorted(list(f['examples'].keys()))[:W]
        train_imgs      = [f['examples'][k]['image'][()]/255.0 for k in keys]
        train_labels    = [f['examples'][k]['label'][()] for k in keys]
    
    with h5py.File(test_fname, 'r') as f:
        keys            = sorted(list(f['examples'].keys()))
        test_imgs       = [f['examples'][k]['image'][()]/255.0 for k in keys]
        test_labels     = [f['examples'][k]['label'][()] for k in keys]
    
    # transpose (rather than reshape) required to get actual order of 
    # data in ndarray to change. If reshape is used, when data is 
    # passed to a torchvision.transform, then the resulting images come
    # out incorrectly.
    train_imgs  = [np.transpose(imgs, (0, 3, 1, 2)) for imgs in train_imgs]
    test_imgs   = [np.transpose(imgs, (0, 3, 1, 2)) for imgs in test_imgs]
    
    test_imgs   = np.concatenate(test_imgs)
    test_labels = np.concatenate(test_labels)
    
    # means and stds computed per channel for entire dataset
    means   = [0.4914, 0.4822, 0.4465]
    stds    = [0.2023, 0.1994, 0.2010]
    
    for w in range(W):
        train_imgs[w] = normalise(train_imgs[w], means, stds)
    
    test_imgs = normalise(test_imgs, means, stds)
    
    return (train_imgs, train_labels), (test_imgs, test_labels)



def to_tensor(x, device, dtype):
    """
    Returns x as a torch.tensor.
    
    Args:
    - x:      {np.ndarray} data to convert
    - device: {torch.device} where to store the tensor
    - dtype:  {torch.dtype or 'long'} type of data
    
    Returns: {torch.tensor}
    """
    if dtype == 'long':
        return torch.tensor(x, device=device, 
                            requires_grad=False, dtype=torch.int32).long()
    else:
        return torch.tensor(x, device=device, requires_grad=False, dtype=dtype)



def step_values(x, m):
    """
    Return a stepwise copy of x, where the values of x that are equal to m are 
    taken from the last non-m value of x.
    
    Args:
    - x: {np.ndarray} values to make step-wise
    - m: {number} (same type as x) value to step over/ignore
    """
    stepped = np.zeros_like(x)
    curr = x[0]
    
    for i in range(1, x.size):
        if x[i] != m:
            curr = x[i]
        stepped[i] = curr
    
    return stepped



def save_data(data, fname, seed):
    """
    Save data in a dictionary entry in fname using given seed as the key. Will 
    append to file at fname if it already exists
    
    Args:
    - data:  {object} to save
    - fname: {str} path to file
    - seed:  {int} seed of random trial
    """
    all_data = {}
    if os.path.isfile(fname):
        with open(fname, 'rb') as f:
            all_data = pickle.load(f)

    all_data[seed] = data
    with open(fname, 'wb') as f:
        pickle.dump(all_data, f)

