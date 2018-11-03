import numpy as np
from torch.autograd import Variable
import torch
import copy


def block_orth_normal_initializer(input_size, output_size):
    weight = []
    for o in output_size:
        for i in input_size:
            param = torch.FloatTensor(o, i)
            torch.nn.init.orthogonal(param)
            weight.append(param)
    return torch.cat(weight)


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != n_position - 1 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.autograd.Variable(torch.from_numpy(position_enc).type(torch.FloatTensor), requires_grad=False)


def batch_data_variable(corpus_tpf2, batch_position_encoding, batch_x, batch_y, batch_lengths, batch_weights):
    batch_size = len(batch_x)  # batch size
    length = max(batch_lengths)

    words = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)  # padding with 0
    predicates = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    tpf2 = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)  # TPF2 ids
    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    padding_answers = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    labels, lengths = [], []
    pes = []

    b = 0
    for s_words, s_answer, s_length, s_weights in zip(batch_x, batch_y, batch_lengths, batch_weights):
        lengths.append(s_length)
        rel = np.zeros((s_length), dtype=np.int32)
        for i in range(s_length):
            words[b, i] = s_words[1][i]  # word
            predicates[b, i] = s_words[2][i]  # predicate
            rel[i] = s_answer[0][i]
            padding_answers[b, i] = s_answer[0][i]
            masks[b, i] = 1

        sentence_id = s_words[0][0]  # get the dep_labels_ids of each sentence
        predicate_id = np.argmax(s_words[2])
        index = str(sentence_id) + '-' + str(predicate_id)
        if index not in corpus_tpf2:
            print("Error index in tpf2 {}.".format(index))
            exit()
        sentence_tpf2 = corpus_tpf2[index]
        for i in range(s_length):
            tpf2[b, i] = sentence_tpf2[i]

        b += 1
        labels.append(rel)
    # pes = torch.stack(pes, dim=0)
    return words, predicates, tpf2, pes, labels, torch.LongTensor(lengths), masks, padding_answers


if __name__ == "__main__":
    position = sentence_length = 10
    n_dim = 10
    pe = position_encoding_init(position, n_dim)
    print pe
