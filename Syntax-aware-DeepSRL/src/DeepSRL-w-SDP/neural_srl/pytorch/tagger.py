from optimizer import *
from layer import *
from HighWayLSTM import *

from collections import OrderedDict
import itertools
import os
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from ..SDPLSTM.Encoder import EncoderSDP


def drop_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    # tensor.new: build a tensor with the same data type
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    tag_masks = tag_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    # unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.
    word_masks = word_masks.unsqueeze(dim=2)  # ?
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings


def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)


def _model_var(model, x):
    if model.cuda_id:
        return torch.autograd.Variable(x).cuda()
    return torch.autograd.Variable(x)


class BiLSTMTaggerModel(nn.Module):
    """ Constructs the network and builds the following Theano functions:
        - pred_function: Takes input and mask, returns prediction.
        - loss_function: Takes input, mask and labels, returns the final cross entropy loss (scalar).
    """
    def __init__(self, data, config, gpu_id=""):
        super(BiLSTMTaggerModel, self).__init__()
        self.embedding_shapes = data.embedding_shapes
        self.config = config
        self.lstm_type = config.lstm_cell
        self.lstm_hidden_size = int(config.lstm_hidden_size)  # SRL: 300
        self.num_lstm_layers = int(config.num_lstm_layers)  # SRL:
        self.max_grad_norm = float(config.max_grad_norm)
        self.vocab_size = data.word_dict.size()
        self.label_space_size = data.label_dict.size()
        self.syntactic_label_num = data.syntactic_dict.size()
        self.syntactic_label_dim = config.syn_label_size
        self.unk_id = data.unk_id
        self.cuda_id = gpu_id

        # Initialize layers and parameters
        word_embedding_shape = data.embedding_shapes[0]
        assert word_embedding_shape[0] == self.vocab_size
        self.word_embedding_dim = word_embedding_shape[1]  # get the embedding dim
        self.embedding = nn.Embedding(word_embedding_shape[0], self.word_embedding_dim, padding_idx=0)
        self.predicate_embedding = nn.Embedding(3, self.word_embedding_dim, padding_idx=0)  # {pad, 0, 1}
        self.syntactic_label_embedding = nn.Embedding(self.syntactic_label_num, self.syntactic_label_dim)

        self.embedding.weight.data.copy_(torch.from_numpy(np.asarray(data.embeddings[0])))
        torch.nn.init.normal(self.predicate_embedding.weight, 0.0, 0.01)
        torch.nn.init.normal(self.syntactic_label_embedding.weight, 0.0, 0.01)

        self.sdp_lstm = EncoderSDP(self.word_embedding_dim, self.word_embedding_dim)
        self.sdp_lstm_projection = nn.Linear(2 * self.word_embedding_dim, self.word_embedding_dim, bias=False)

        # Initialize BiLSTM
        self.bilstm = HighwayBiLSTM(
            input_size= 3 * self.word_embedding_dim,
            hidden_size=self.lstm_hidden_size,  # // 2 for MyLSTM
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.input_dropout_prob,
            dropout_out=config.recurrent_dropout_prob
        )
        self.softmax = nn.Linear(self.lstm_hidden_size, self.label_space_size)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.normal(self.linear_projection.weight, 0.0, 0.01)  # linear projection
        nn.init.normal(self.softmax.weight, 0.0, 0.01)  # softmax layer
        nn.init.constant(self.softmax.bias, 0.0)

    def __init_hidden(self, mini_batch_size):
        if self.cuda_id:
            return (Variable(torch.zeros(self.num_lstm_layers * 2, mini_batch_size, self.lstm_hidden_size // 2)).cuda(),
                    Variable(torch.zeros(self.num_lstm_layers * 2, mini_batch_size, self.lstm_hidden_size // 2)).cuda())
        else:
            return (Variable(torch.zeros(self.num_lstm_layers, mini_batch_size, self.lstm_hidden_size // 2)),
                    Variable(torch.zeros(self.num_lstm_layers, mini_batch_size, self.lstm_hidden_size // 2)))

    def init_masks(self, batch_size, lengths):
        max_length = max(lengths)
        masks = Variable(torch.Tensor(batch_size, max_length).zero_(), requires_grad=False)
        for i, length in enumerate(lengths):
            masks.data[i][:length] += 1.0
        if self.cuda_id:
            masks = masks.cuda()
        return masks

    def forward(self, x, x_predicate, x_syn_labels, x_pes, x_lengths):
        embeddings, precidate_embeddings, syn_label_embeddings = \
            self.embedding(x), self.predicate_embedding(x_predicate), self.syntactic_label_embedding(x_syn_labels)

        _, predicates_id = x_predicate.max(dim=1)
        syn_label_embeddings = self.sdp_lstm(syn_label_embeddings, predicates_id, x_pes)
        syn_label_embeddings = self.sdp_lstm_projection(syn_label_embeddings)

        embeddings = torch.cat((embeddings, precidate_embeddings, syn_label_embeddings), 2)

        masks = self.init_masks(len(x_lengths), x_lengths)
        lstm_out, _ = self.bilstm(embeddings, masks)
        self.label_scores = self.softmax(lstm_out)
        self.label_scores = F.log_softmax(self.label_scores, dim=2)
        return self.label_scores

    def compute_loss(self, output, answer, weights):
        # output: [B, T, L], answer: [B, T], mask: [B, T, L]
        # print answer
        output = output.view(output.size()[0] * output.size()[1], output.size()[2])
        target = answer.view(-1, 1)
        # print target
        negative_log_likelihood_flat = -torch.gather(output, dim=1, index=target)
        # print negative_log_likelihood_flat
        negative_log_likelihood = negative_log_likelihood_flat.view(*answer.size())  # [B, T]
        negative_log_likelihood = negative_log_likelihood * weights.float()
        loss = negative_log_likelihood.sum(1).mean()
        return loss

    def save(self, filepath):
        """ Save model parameters to file.
        """
        torch.save(self.state_dict(), filepath)
        print('Saved model to: {}'.format(filepath))

    def load(self, filepath):
        """ Load model parameters from file.
        """
        model_params = torch.load(filepath)
        for model_param, pretrained_model_param in zip(self.parameters(), model_params.items()):
            if pretrained_model_param[1].size()[0] > 10000:  # pretrained word embedding
                pretrained_word_embedding_size = pretrained_model_param[1].size()[0]
                model_param.data[:pretrained_word_embedding_size].copy_(pretrained_model_param[1])
                print("Load {} pretrained word embedding!".format(pretrained_word_embedding_size))
            else:
                model_param.data.copy_(pretrained_model_param[1])
        print('Loaded model from: {}'.format(filepath))
