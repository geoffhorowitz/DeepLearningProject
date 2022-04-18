"""
S2S Decoder model.  (c) 2021 Georgia Tech

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


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type


        # initialize embedding layer
        self.embedding = nn.Embedding(output_size, emb_size) # same as output size b/c prior output is fed as current input

        # initialize recurrent layer
        self.model = nn.LSTM(emb_size, decoder_hidden_size, batch_first=True)

        # initialize Linear layers
        self.linear = nn.Linear(decoder_hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # initialize dropout layer
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, 1); HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder
            Returns:
                output (tensor): the output of the decoder
                hidden (tensor): the weights coming out of the hidden unit
        """

        emb = self.embedding(input)
        emb_w_drop = self.dropout(emb)
        output_post_model, hidden = self.model(emb_w_drop, hidden) # hidden output is (hidden, cell_state) if model_type is "LSTM"
        output = self.logsoftmax(self.linear(output_post_model[:, 0, :]))

        return output, hidden
