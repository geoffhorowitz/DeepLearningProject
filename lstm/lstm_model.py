
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

from lstm.custom_LSTM import LSTM

class IngredModel_ref(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.irnn = nn.LSTM(input_size=args.ingredient_embedding_dim, hidden_size=args.ingredient_lstm_dim,
                            bidirectional=True, batch_first=True)
        #_, vec = torchwordemb.load_word2vec_bin(args.ingrW2V) #torchwordemb doesn't pip install
        vec = self.word2vec_emb(args)
        self.embs = nn.Embedding(vec.size(0), args.ingredient_embedding_dim, padding_idx=0) # not sure about the padding idx
        self.embs.weight.data.copy_(vec)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def word2vec_emb(self, args):
        # more info for generating embedding: https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
        # word2vec documentation https://radimrehurek.com/gensim/models/word2vec.html

        model = word2vec.load(args.ingredient_w2v_path) #dict_keys(['vocab', 'vectors', 'clusters', 'vocab_hash'])
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
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
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

        return output


class IngredModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        # instantiate the LSTM model
        self.lstm = nn.LSTM(input_size=args.ingredient_embedding_dim, hidden_size=args.ingredient_lstm_dim, bidirectional=True, batch_first=True)
        #self.lstm = LSTM(input_size=args.ingredient_embedding_dim, hidden_size=args.ingredient_lstm_dim)

        # taking the embedding weights from the pretrained bi-directional LSTM logistic regression on vocab
        vec = self.word2vec_emb(args)
        self.emb = nn.Embedding(vec.size(0), args.ingredient_embedding_dim)
        self.emb.weight.data.copy_(vec)

        # saving cuda device for
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def word2vec_emb(self, args):
        # more info for generating embedding: https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
        # word2vec documentation https://radimrehurek.com/gensim/models/word2vec.html

        model = word2vec.load(args.ingredient_w2v_path) #dict_keys(['vocab', 'vectors', 'clusters', 'vocab_hash'])
        return torch.Tensor(model.vectors)

    def forward(self, data):
        x = data[3].to(self.device)
        seq_length = data[4].to(self.device)

        embedded = self.emb(x)
        out, hidden = self.lstm(embedded)
        # pack padded and pad packed lets the model ignore padded elements (and resequences them on the backend)
        # ref: https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
        # per docs, to avoid sorting the inputs, can pass enforce_sorted=False as long as we don't need ONNX exportability
        #packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_length, batch_first=True, enforce_sorted=False)
        #output, hidden = self.lstm(packed_embedded)
        #output_unpacked, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, enforce_sorted=False)
        #out = output_unpacked[:, -1, :]

        return out[:, -1, :]


class RecipeModel_ref(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lstm = nn.LSTM(input_size=args.recipe_embedding_dim, hidden_size=args.recipe_lstm_dim,
                            bidirectional=False, batch_first=True)

        # saving cuda device for
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, x, data):
        input_var = list()
        for j in range(len(data)):
            input_var.append(data[j].to(self.device))

        x = input_var[1]
        sq_lengths = input_var[2]

        # here we use a previous LSTM to get the representation of each instruction
        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())

        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the lstm
        out, hidden = self.lstm(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unsorted_idx = original_idx.view(-1,1,1).expand_as(unpacked)
        # we get the last index of each sequence in the batch
        idx = (sq_lengths-1).view(-1,1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        # we sort and get the last element of each sequence
        output = unpacked.gather(0, unsorted_idx.long()).gather(1,idx.long())
        output = output.view(output.size(0),output.size(1)*output.size(2))

        return output


class RecipeModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lstm = nn.LSTM(input_size=args.recipe_embedding_dim, hidden_size=args.recipe_lstm_dim, bidirectional=False, batch_first=True)
        #self.lstm = LSTM(input_size=args.recipe_embedding_dim, hidden_size=args.recipe_lstm_dim)

        # saving cuda device for
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def forward(self, data):
        x = data[1].to(self.device)
        seq_length = data[2].to(self.device)

        embedded = x
        out, hidden = self.lstm(x)

        # see comments in IngredRecipe above
        #packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_length, batch_first=True, enforce_sorted=False)
        #output, hidden = self.lstm(packed_embedded)
        #output_unpacked, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, enforce_sorted=False)
        #out = output_unpacked[:, -1, :]

        return out[:, -1, :]
