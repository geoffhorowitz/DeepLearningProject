"""
S2S Encoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        # initialize embedding layer
        self.embedding = nn.Embedding(input_size, emb_size)

        # initialize recurrent layer
        self.model = nn.LSTM(emb_size, encoder_hidden_size, batch_first=True)

        # initialize Linear layers + ReLU activation
        self.linear1 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(encoder_hidden_size, decoder_hidden_size)

        # initialize dropout layer
        self.dropout = nn.Dropout(dropout)


    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder;
                hidden (tensor): the weights coming out of the last hidden unit
        """

        emb = self.embedding(input)
        emb_w_drop = self.dropout(emb)
        model_output = self.model(emb_w_drop)
        output, (hidden_post_model, cell_state) = model_output
        hidden_pre_tanh = self.linear2(self.relu(self.linear1(hidden_post_model)))
        hidden = torch.tanh(hidden_pre_tanh)
        hidden = (hidden, cell_state)

        return output, hidden
