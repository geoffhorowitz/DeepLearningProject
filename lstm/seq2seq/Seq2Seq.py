import random

"""
Seq2Seq model.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn
import torch.optim as optim


# import custom models


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)


    def forward(self, source, out_seq_len=None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        seq_len = source.shape[1]
        if out_seq_len is None:
            out_seq_len = seq_len

        #for i in range(seq_len):
        #    encoder_out, encoder_hidden = self.encoder.forward(source[:, i])
        encoder_out, encoder_hidden = self.encoder.forward(source)
        #print('source: ', source.shape)
        #print('out seq: ', out_seq_len)
        #print('encoder out: ', encoder_out.shape)
        #print('encoder hidden: ', encoder_hidden, encoder_hidden.shape)

        sos_token = source[:, 0:1]
        outputs = torch.zeros(batch_size, out_seq_len, self.decoder.output_size, device=self.device)
        #print('sos: ', sos_token.shape)
        decoder_next_in = sos_token
        decoder_hidden = encoder_hidden
        #print('encoder_hidden: ', encoder_hidden[0].shape if isinstance(encoder_hidden, tuple) else encoder_hidden.shape)
        for i in range(out_seq_len):
            #print(i)
            decoder_out, decoder_hidden = self.decoder.forward(decoder_next_in, decoder_hidden)
            outputs[:, i] = decoder_out
            decoder_next_in = decoder_out.argmax(1, keepdim=True)
            #if i == 0:
            #    print('decoder out: ', decoder_out.shape)
            #    print('decoder hidden: ', decoder_hidden[0].shape if isinstance(decoder_hidden, tuple) else decoder_hidden.shape)
            #    print('decoder next in: ', decoder_next_in.shape)

        return outputs
