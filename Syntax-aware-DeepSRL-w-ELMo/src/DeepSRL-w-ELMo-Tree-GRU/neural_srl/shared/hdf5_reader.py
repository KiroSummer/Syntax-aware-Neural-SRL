import h5py
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn


class hdf5_reader(nn.Module):
    def __init__(self):
        self.filename = ""
        self.data = None
        self.f = None
        self.sentences = None

    def read_from_file(self, filename, sentences):
        self.filename = filename
        self.sentences = sentences
        print("Loading elmo hdf5 from {}".format(self.filename))
        self.f = h5py.File(filename, 'r')
        """for idx, sen in enumerate(sentences):
            if idx == 28:
                embeddings = list(self.f[sen])
                embeddings = torch.autograd.Variable(torch.from_numpy(np.array(embeddings)), requires_grad=False)
                print(embeddings)
                x = (embeddings[0] + embeddings[1] + embeddings[2]) / 3.0
                print(x)
                exit()"""

    def forward(self, x, max_length, sentences_lengths):
        output = []
        for idx, sen_len in zip(x, sentences_lengths):
            sen = self.sentences[idx]  # get the correspoding sentence
            sentence_length = sen.count(' ') + 1
            assert sentence_length == sen_len
            embeddings = list([self.f[sen]])
            embeddings = torch.autograd.Variable(torch.from_numpy(np.array(embeddings)), requires_grad=False)
            embeddings = torch.squeeze(embeddings, 0)
            pad = (0, 0, 0, max_length - embeddings.size()[1])
            embeddings = F.pad(embeddings, pad)
            output.append(embeddings)
        output = torch.stack(output, dim=0)
        return output

