
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

#from lstm.custom_LSTM import LSTM
# from lstm.Transformer import TransformerTranslator

'''
reference ("_ref") models source:
@article{marin2019learning,
  title = {Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images},
  author = {Marin, Javier and Biswas, Aritro and Ofli, Ferda and Hynes, Nicholas and
  Salvador, Amaia and Aytar, Yusuf and Weber, Ingmar and Torralba, Antonio},
  journal = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  year = {2019}
}

@inproceedings{salvador2017learning,
  title={Learning Cross-modal Embeddings for Cooking Recipes and Food Images},
  author={Salvador, Amaia and Hynes, Nicholas and Aytar, Yusuf and Marin, Javier and
          Ofli, Ferda and Weber, Ingmar and Torralba, Antonio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}
'''

class IngredModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # saving cuda device for
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # instantiate the LSTM model
        if args.recipe_model == 'transformer':
            #self.model = nn.Transformer(args.ingredient_embedding_dim, args.num_heads, dim_feedforward=args.dim_feedforward, batch_first=True, device=self.device)
            self.model = TransformerTranslator(input_size=args.ingredient_embedding_dim, output_size=args.ingredient_embedding_dim, device=self.device, hidden_dim=args.hidden_dim, num_heads=args.num_heads, dim_feedforward=args.dim_feedforward, dim_k=96, dim_v=96, dim_q=96, max_length=50, add_position_embedding=False)
        else:
            self.model = nn.LSTM(input_size=args.ingredient_embedding_dim, hidden_size=args.ingredient_lstm_dim, bidirectional=True, batch_first=True)
            #self.model = LSTM(input_size=args.ingredient_embedding_dim, hidden_size=args.ingredient_lstm_dim)


        # taking the embedding weights from the pretrained bi-directional LSTM logistic regression on vocab
        vec = self.word2vec_emb(args)
        self.emb = nn.Embedding(vec.size(0), args.ingredient_embedding_dim)
        self.emb.weight.data.copy_(vec)

    def word2vec_emb(self, args):
        # more info for generating embedding: https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
        # word2vec documentation https://radimrehurek.com/gensim/models/word2vec.html

        model = word2vec.load(args.ingredient_w2v_path) #dict_keys(['vocab', 'vectors', 'clusters', 'vocab_hash'])
        return torch.Tensor(model.vectors)

    def forward(self, data):
        x = data[3].to(self.device)
        seq_length = data[4].to(self.device)

        embedded = self.emb(x)

        if self.args.recipe_model == 'transformer':
            output = self.model(embedded)
            print('recipe_out', output.shape)
            output = output[:, -1, :]
            return output
        elif self.args.ingred_model_variant == 'base':
            # after 10 epochs, .1 train/.05 val: .71 train loss, 1.46 val loss, Mean median 17.25 Recall {1: 0.054000000000000006, 5: 0.22000000000000003, 10: 0.35500000000000004}
            output, hidden = self.model(embedded)
            #print('ingred_out', output.shape)
            output = output[:, -1, :]
            return output

        elif self.args.ingred_model_variant == 'custom_basic':
            # pack padded and pad packed lets the model ignore padded elements (and resequences them on the backend) --> helps train faster
            # ref: https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
            # ref: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
            # ref: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html

            # per docs, to avoid sorting the inputs, can pass enforce_sorted=False as long as we don't need ONNX exportability
            packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_length.cpu().data.numpy(), batch_first=True, enforce_sorted=False)
            output, hidden = self.model(packed_embedded)
            output_unpacked, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            #print('ingred_out', output_unpacked.shape)
            out = output_unpacked[:, -1, :]
            return out
        
        elif self.args.ingred_model_variant == 'custom_fusion':
        # after 10 epochs, .1 train/.05 val: .71 train loss, 1.46 val loss, Mean median 17.25 Recall {1: 0.035, 5: 0.20199999999999996, 10: 0.35600000000000004}

            # pack padded and pad packed lets the model ignore padded elements (and resequences them on the backend) --> helps train faster
            # ref: https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
            # ref: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
            # ref: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html

            # per docs, to avoid sorting the inputs, can pass enforce_sorted=False as long as we don't need ONNX exportability
            packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_length.cpu().data.numpy(), batch_first=True, enforce_sorted=False)
            output, hidden = self.model(packed_embedded)
            #output_unpacked, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(hidden[0], batch_first=True)
            original_ndx = packed_embedded.sorted_indices
            unsort_ndx = original_ndx.view(1,-1,1).expand_as(hidden[0])
            #unsort_ndx = packed_embedded.unsorted_indices.expand_as(hidden[0])
            output_unpacked = hidden[0].gather(1,unsort_ndx).transpose(0,1).contiguous()
            #print('ingred', output_unpacked.shape)
            out = output_unpacked.view(output_unpacked.size(0),output_unpacked.size(1)*output_unpacked.size(2))
            #print('ingred_out', out.shape)

            return out

        elif self.args.ingred_model_variant == 'paper':

            # after 10 epochs, .1 train/.05 val: .7 train loss, 1.42 val loss, Mean median 16.75 Recall {1: 0.037, 5: 0.142, 10: 0.246}
            sorted_len, sorted_idx = seq_length.sort(0, descending=True)
            index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(embedded)
            sorted_inputs = embedded.gather(0, index_sorted_idx.long())
            packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
            out, hidden = self.model(packed_embedded)
            _, original_idx = sorted_idx.sort(0, descending=False)
            unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
            output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
            output = output.view(output.size(0),output.size(1)*output.size(2))

            return output

        else:
            print('ingredient model variant not understood.')


class RecipeModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # saving cuda device for
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if args.recipe_model == 'transformer':
            #self.model = nn.Transformer(args.recipe_embedding_dim, args.num_heads, dim_feedforward=args.dim_feedforward, batch_first=True, device=self.device)
            self.model = TransformerTranslator(input_size=args.recipe_embedding_dim, output_size=args.recipe_embedding_dim, device=self.device, hidden_dim=args.hidden_dim, num_heads=args.num_heads, dim_feedforward=args.dim_feedforward, dim_k=96, dim_v=96, dim_q=96, max_length=50)
        else:
            self.model = nn.LSTM(input_size=args.recipe_embedding_dim, hidden_size=args.recipe_lstm_dim, bidirectional=False, batch_first=True)
            #self.model = LSTM(input_size=args.recipe_embedding_dim, hidden_size=args.recipe_lstm_dim)

    def forward(self, data):

        x = data[1].to(self.device)
        seq_length = data[2].to(self.device)
        embedded = x
        #print(x.shape)

        if self.args.recipe_model == 'transformer':
            output = self.model(x)
            print('recipe_out', output.shape)
            output = output[:, -1, :]
            return output
        elif self.args.recipe_model_variant == 'base':
            output, hidden = self.model(x)
            #print('recipe_out', output.shape)
            output = output[:, -1, :]
            return output

        elif self.args.recipe_model_variant == 'custom_basic':
            # after 10 epochs, .1 train/.05 val: .78 train loss, 1.54 val loss, Mean median 29.4 Recall {1: 0.019999999999999997, 5: 0.11099999999999999, 10: 0.215}
            # see comments in IngredRecipe above
            packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_length.cpu().data.numpy(), batch_first=True, enforce_sorted=False)
            output, hidden = self.model(packed_embedded)
            #output = self.trans(x)
            output_unpacked, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            #print('recipe_out', output_unpacked.shape)
            out = output_unpacked[:, -1, :]
            return out

        elif self.args.recipe_model_variant == 'custom_fusion':
            # after 10 epochs, .1 train/.05 val: .74 train loss, 1.45 val loss, Mean median 26.4 Recall {1: 0.06100000000000001, 5: 0.22400000000000003, 10: 0.36699999999999994}
            packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_length.cpu().data.numpy(), batch_first=True, enforce_sorted=False)
            output, hidden = self.model(packed_embedded)
            output_unpacked, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            original_ndx = packed_embedded.sorted_indices
            unsort_ndx = original_ndx.view(-1,1,1).expand_as(output_unpacked)
            #unsort_ndx = packed_embedded.unsorted_indices.expand_as(output_unpacked)
            ref_idx = (seq_length-1).view(-1,1).expand(output_unpacked.size(0), output_unpacked.size(2)).unsqueeze(1)
            output = output_unpacked.gather(0, unsort_ndx.long()).gather(1, ref_idx.long())
            #print('recipe_out', output.shape)
            output = output.view(output.size(0),output.size(1)*output.size(2))
            out = output
            return out

        elif self.args.recipe_model_variant == 'paper':
            # after 10 epochs, .1 train/.05 val: .7 train loss, 1.42 val loss, Mean median 16.75 Recall {1: 0.037, 5: 0.142, 10: 0.246}
            sorted_len, sorted_idx = seq_length.sort(0, descending=True)
            index_sorted_idx = sorted_idx.view(-1,1,1).expand_as(embedded)
            sorted_inputs = embedded.gather(0, index_sorted_idx.long())
            # pack sequence
            packed_seq = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
            # pass it to the lstm
            out, hidden = self.model(packed_seq)

            # unsort the output
            _, original_idx = sorted_idx.sort(0, descending=False)

            unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            unsorted_idx = original_idx.view(-1,1,1).expand_as(unpacked)
            idx = (seq_length-1).view(-1,1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
            output = unpacked.gather(0, unsorted_idx.long()).gather(1,idx.long())
            output = output.view(output.size(0),output.size(1)*output.size(2))

            return output

        else:
            print('recipe model variant not understood.')
