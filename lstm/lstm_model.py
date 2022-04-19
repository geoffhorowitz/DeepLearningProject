
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
#import torchwordemb
import word2vec

class IngredModel(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)
        #_, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V) #torchwordemb doesn't pip install
        vec = self.word2vec_emb(opts)
        self.embs = nn.Embedding(vec.size(0), opts.ingrW2VDim, padding_idx=0) # not sure about the padding idx
        self.embs.weight.data.copy_(vec)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def word2vec_emb(self, opts):
        # more info for generating embedding: https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
        # word2vec documentation https://radimrehurek.com/gensim/models/word2vec.html

        model = word2vec.load(opts.ingrW2V) #dict_keys(['vocab', 'vectors', 'clusters', 'vocab_hash'])
        return torch.Tensor(model.vectors)

    def forward(self, data): # from img2recipe forward
    #def forward(self, x, sq_lengths):
        input_var = list()
        for j in range(len(data)):
            input_var.append(data[j].to(self.device))

        x = input_var[3]
        sq_lengths = input_var[4]

        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x)

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx\
                .view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the rnn
        out, hidden = self.irnn(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM
        # bi-directional
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
        output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
        output = output.view(output.size(0),output.size(1)*output.size(2))

        return output, None
