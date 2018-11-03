import torch
import torch.nn.functional as F
from torch import nn


class ELMO(nn.Module):
    def __init__(self, elmo_representations):
        super(ELMO, self).__init__()
        self.elmos = {}
        self.elmo_dim = elmo_representations.data[0].shape[-1]  # 1024
        self.process_elmos(elmo_representations.data)

    def process_elmos(self, elmo_representations):
        for i, elmo in enumerate(elmo_representations):
            assert i not in self.elmos.keys()
            self.elmos[i] = torch.nn.Parameter(torch.from_numpy(elmo), requires_grad=False)  # convert numpy to tensor to Variable

    def forward(self, x, max_length):  # should be list sentences ids. [id0, id1, ..., id[batch -1]]
        output = []
        for i, sentence_id in enumerate(x):
            elmo = self.elmos[sentence_id]
            pad = (0, 0, 0, max_length - elmo.size()[0])
            out = F.pad(elmo, pad)
            output.append(out)
        output = torch.stack(output, dim=0)
        return output

    def save(self, filepath):
        """ Save model parameters to file.
        """
        sorted_elmos = sorted(self.elmos.items(), key=lambda x:x[0])
        print sorted_elmos[:2]
        exit()

