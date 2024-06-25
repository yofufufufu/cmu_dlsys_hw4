"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, ReLU, Tanh


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # 不用1 / (1 + ops.exp(-x))是因为scalar / tensor（可能）会出问题
        return (1 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.bias = bias
        bound = np.sqrt(1 / hidden_size)
        self.W_ih = Parameter(
            init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True)
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True)
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(
                init.rand(hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None
        if nonlinearity == "tanh":
            self.nonlinearity = Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = ReLU()
        else:
            raise ValueError("unsupported nonlinearity function. Only support ReLU and Tanh.")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        if self.bias:
            return self.nonlinearity(X @ self.W_ih + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size)) +
                                     h @ self.W_hh + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size)))
        else:
            return self.nonlinearity(X @ self.W_ih + h @ self.W_hh)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.rnn_cells = []
        for i in range(num_layers):
            if i == 0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[1]
        seq_len = X.shape[0]
        h_list = []
        h_n = []
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        h0_tuple = ops.split(h0, axis=0)
        X_tuple = ops.split(X, axis=0)
        for i in range(self.num_layers):
            h = h0_tuple[i]
            for j in range(seq_len):
                if i == 0:
                    h = self.rnn_cells[i](X_tuple[j], h)
                    h_list.append(h)
                else:
                    h = self.rnn_cells[i](h_list[j], h)
                    h_list[j] = h
            h_n.append(h)
        return ops.stack(h_list, axis=0), ops.stack(h_n, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.bias = bias
        bound = np.sqrt(1 / hidden_size)
        if bias:
            self.bias_ih = Parameter(
                init.rand(4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(
                init.rand(4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.W_ih = Parameter(
            init.rand(input_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True)
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, 4*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True)
        )
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        if self.bias:
            gate_res = self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to((bs, 4*self.hidden_size)) + X @ self.W_ih + \
                       self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to((bs, 4*self.hidden_size)) + h0 @ self.W_hh
        else:
            gate_res = X @ self.W_ih + h0 @ self.W_hh
        # 我们实现的split只能num_splits=1，所以切分完再stack
        splitted = tuple(ops.split(gate_res, axis=1))
        i = ops.stack(splitted[:self.hidden_size], axis=1)
        f = ops.stack(splitted[self.hidden_size:2*self.hidden_size], axis=1)
        g = ops.stack(splitted[2*self.hidden_size:3*self.hidden_size], axis=1)
        o = ops.stack(splitted[3*self.hidden_size:], axis=1)
        i, f, g, o = Sigmoid()(i), Sigmoid()(f), Tanh()(g), Sigmoid()(o)
        c_out = f * c0 + i * g
        h_out = o * Tanh()(c_out)
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_layers = num_layers
        self.lstm_cells = []
        for i in range(num_layers):
            if i == 0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[1]
        seq_len = X.shape[0]
        h_list = []
        h_n = []
        c_n = []
        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        h0_tuple = ops.split(h0, axis=0)
        c0_tuple = ops.split(c0, axis=0)
        X_tuple = ops.split(X, axis=0)
        for i in range(self.num_layers):
            h = h0_tuple[i]
            c = c0_tuple[i]
            for j in range(seq_len):
                if i == 0:
                    h, c = self.lstm_cells[i](X_tuple[j], (h, c))
                    h_list.append(h)
                else:
                    h, c = self.lstm_cells[i](h_list[j], (h, c))
                    h_list[j] = h
            h_n.append(h)
            c_n.append(c)
        return ops.stack(h_list, axis=0), (ops.stack(h_n, axis=0), ops.stack(c_n, axis=0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0, device=device, dtype=dtype,
                                           requires_grad=True))
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        num_embeddings, embedding_dim = self.weight.shape
        # one_hot的结果是(seq_len, bs, num_embeddings)，为了进行embedding矩阵乘法，需要reshape
        x_one_hot = init.one_hot(num_embeddings, x, device=self.device, dtype=self.dtype).reshape((seq_len * bs, num_embeddings))
        return (x_one_hot @ self.weight).reshape((seq_len, bs, embedding_dim))
        ### END YOUR SOLUTION