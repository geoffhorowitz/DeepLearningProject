"""
LSTM model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns:
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # reference: https://pytorch.org/tutorials/beginner/nn_tutorial.html

        self.init_states = None

        wi_shape = (input_size, hidden_size)
        bi_shape = (hidden_size)
        wh_shape = (hidden_size, hidden_size)
        bh_shape = (hidden_size)

        # i_t: input gate
        self.wii = nn.Parameter(torch.zeros(wi_shape))
        self.bii = nn.Parameter(torch.zeros(bi_shape))
        self.whi = nn.Parameter(torch.zeros(wh_shape))
        self.bhi = nn.Parameter(torch.zeros(bh_shape))

        # f_t: the forget gate
        self.wif = nn.Parameter(torch.zeros(wi_shape))
        self.bif = nn.Parameter(torch.zeros(bi_shape))
        self.whf = nn.Parameter(torch.zeros(wh_shape))
        self.bhf = nn.Parameter(torch.zeros(bh_shape))

        # g_t: the cell gate
        self.wig = nn.Parameter(torch.zeros(wi_shape))
        self.big = nn.Parameter(torch.zeros(bi_shape))
        self.whg = nn.Parameter(torch.zeros(wh_shape))
        self.bhg = nn.Parameter(torch.zeros(bh_shape))

        # o_t: the output gate
        self.wio = nn.Parameter(torch.zeros(wi_shape))
        self.bio = nn.Parameter(torch.zeros(bi_shape))
        self.who = nn.Parameter(torch.zeros(wh_shape))
        self.bho = nn.Parameter(torch.zeros(bh_shape))

        # activation functions
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        if init_states:
            h_prior, c_prior = init_states
        elif self.init_states:
            h_prior, c_prior = self.init_state
        else:
            h_prior = torch.zeros((x.shape[0], self.hidden_size))
            c_prior = torch.zeros((x.shape[0], self.hidden_size))

        for i in range(x.shape[1]):
            x_i = x[:, i, :]
            # @ == torch.matmul
            i_t = self.sigmoid(x_i @ self.wii + self.bii + h_prior @ self.whi + self.bhi)
            f_t = self.sigmoid(x_i @ self.wif + self.bif + h_prior @ self.whf + self.bhf)
            g_t = self.tanh(x_i @ self.wig + self.big + h_prior @ self.whg + self.bhg)
            o_t = self.sigmoid(x_i @ self.wio + self.bio + h_prior @ self.who + self.bho)
            c_t = f_t * c_prior + i_t * g_t
            h_t = o_t * self.tanh(c_t)
            h_prior = h_t
            c_prior = c_t

        self.init_state = (h_prior, c_prior)

        return (h_t, c_t)
