import numpy as np
from torch.autograd import Variable
import torch


def block_orth_normal_initializer(input_size, output_size):
    weight = []
    for o in output_size:
        for i in input_size:
            param = torch.FloatTensor(o, i)
            torch.nn.init.orthogonal_(param)
            weight.append(param)
    return torch.cat(weight)


def batch_data_variable(dependency_tpf2, batch_x, batch_y, batch_lengths, batch_weights):
    batch_size = len(batch_x)  # batch size
    length = max(batch_lengths)

    words = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)  # padding with 0
    predicates = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    tpf2 = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)  # TPF2 ids
    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    padding_answers = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    labels, lengths = [], []
    sentences_ids = []

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
        sentences_ids.append(sentence_id)

        predicate_id = np.argmax(s_words[2])
        index = str(sentence_id) + '-' + str(predicate_id)
        if index not in dependency_tpf2:
            print("Error index in tpf2 {}.".format(index))
            exit()
        sentence_tpf2 = dependency_tpf2[index]
        for i in range(s_length):
            tpf2[b, i] = sentence_tpf2[i]

        b += 1
        labels.append(rel)

    return words, predicates, tpf2, sentences_ids, labels, torch.LongTensor(lengths), masks, padding_answers
