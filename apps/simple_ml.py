"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("/WdHeDisk/users/tang22/CMU_DLSys/cmu_dlsys_hw4/python")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # 我没做hw0，copy from https://github.com/kcxain/dlsys/blob/master/hw0/src/simple_ml.py
    f = gzip.open(image_filesname)
    data = f.read()
    f.close()
    h = struct.unpack_from('>IIII', data, 0)
    offset = struct.calcsize('>IIII')
    imgNum = h[1]
    rows = h[2]
    columns = h[3]
    pixelString = '>' + str(imgNum * rows * columns) + 'B'
    pixels = struct.unpack_from(pixelString, data, offset)
    X = np.reshape(pixels, [imgNum, rows * columns]).astype('float32')
    X_max = np.max(X)
    X_min = np.min(X)
    # X_max = np.max(X, axis=1, keepdims=True)
    # X_min = np.min(X, axis=1, keepdims=True)

    X_normalized = ((X - X_min) / (X_max - X_min))

    f = gzip.open(label_filename)
    data = f.read()
    f.close()
    h = struct.unpack_from('>II', data, 0)
    offset = struct.calcsize('>II')
    num = h[1]
    labelString = '>' + str(num) + 'B'
    labels = struct.unpack_from(labelString, data, offset)
    y = np.reshape(labels, [num]).astype('uint8')

    return (X_normalized, y)
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # combine soft-max and cross-entropy
    batch_num = Z.shape[0]
    loss_per_batch = needle.ops.log(needle.ops.exp(Z).sum(axes=(1,))) - (Z * y_one_hot).sum(axes=(1,))
    return loss_per_batch.sum() / batch_num
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    for i in range(0, num_examples, batch):
        X_batch = X[i: i + batch]
        y_batch = y[i: i + batch]
        X_batch = ndl.Tensor(X_batch)
        Z1 = ndl.ops.relu(X_batch @ W1)
        Z = Z1 @ W2
        # get one-hot
        y_one_hot = np.zeros(Z.shape, dtype="float32")
        y_one_hot[np.arange(Z.shape[0]), y_batch] = 1
        loss = softmax_loss(Z, ndl.Tensor(y_one_hot))
        loss.backward()

        # detach: create a new tensor that shares the data but detaches from the graph(no op and inputs, only has cached data)
        # create new Tensors for W1 and W2 with these numpy values
        # 因为更新W的计算: W - lr * grad不应该成为计算图中的内容
        # 新的W1和W2在新一轮的计算图中还是叶子结点
        W1 = (W1 - lr * W1.grad).detach()
        W2 = (W2 - lr * W2.grad).detach()
    return W1, W2
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
                      clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    losses = []
    corrects = []
    dataset_size = 0
    train = opt is not None
    if train:
        model.train()
    else:
        model.eval()

    nbatch, batch_size = data.shape

    hidden = None
    for i in range(0, nbatch - 1, seq_len):
        x, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)

        batch_size = y.shape[0]
        dataset_size += batch_size
        y_pred, hidden = model(x, hidden)

        # 把前一个batch的final hidden state作为下一个batch的initial hidden state
        # lstm
        if isinstance(hidden, tuple):
            h, c = hidden
            hidden = (h.detach(), c.detach())
        # rnn
        else:
            hidden = hidden.detach()
        # soft_max = nn.SoftmaxLoss()
        # y_pred shape: (seq_len * batch_size, output_size)
        # y shape: (seq_len * batch_size)
        loss = loss_fn(y_pred, y)
        # print(loss)
        if train:
            opt.reset_grad()
            loss.backward()
            opt.step()

        losses.append(loss.numpy() * batch_size)
        correct = np.sum(y_pred.numpy().argmax(axis=1) == y.numpy())
        corrects.append(correct)

    avg_acc = np.sum(np.array(corrects)) / dataset_size
    avg_loss = np.sum(np.array(losses)) / dataset_size
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
              lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss(), clip=None,
              device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=opt, device=device,
                                              dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss(),
                 device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=None, device=device,
                                          dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
