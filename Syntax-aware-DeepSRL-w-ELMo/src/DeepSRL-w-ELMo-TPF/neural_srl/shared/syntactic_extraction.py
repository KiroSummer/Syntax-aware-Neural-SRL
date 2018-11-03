import numpy as np
import torch
from dictionary import *
from constants import *
from ..SDPLSTM.Tree import *


class SyntacticTree(object):
    def __init__(self, sentence_id):
        self.sentence_id = sentence_id
        self.word_forms = []
        self.pos_forms = []
        self.heads = []
        self.labels = []
        self.labels_id = []


class SyntacticCONLL(object):
    def __init__(self):
        self.file_name = ""
        self.trees = []

    def read_from_file(self, filename):
        self.file_name = filename

        print("Reading conll syntactic trees from {}".format(self.file_name))
        conll_file = open(self.file_name, 'r')
        if conll_file.closed:
            print("Cannot open the syntactic conll file! Please check {}".format(self.file_name))

        sentence_id = 0
        a_tree = SyntacticTree(sentence_id)
        for line in conll_file:
            if line == '\n' or line == '\r\n':  # new sentence
                sentence_id += 1
                self.trees.append(a_tree)
                a_tree = SyntacticTree(sentence_id)
                continue
            tokens = line.strip().split('\t')
            a_tree.word_forms.append(tokens[1])
            a_tree.pos_forms.append(tokens[3])
            # head = int(tokens[6]) if int(tokens[6]) > 0 else -1
            head = int(tokens[6]) - 1
            a_tree.heads.append(head)
            a_tree.labels.append(tokens[7])
        print("Load {} conll syntactic trees.".format(len(self.trees)))

    def get_syntactic_label_dict(self, syn_label_dict=None):
        if syn_label_dict is None:
            syn_label_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)
        else:
            assert syn_label_dict.accept_new is False
        sentences_length = len(self.trees)
        for i in range(sentences_length):
            ith_sentence_length = len(self.trees[i].labels)
            for j in range(ith_sentence_length):
                self.trees[i].labels_id.append(syn_label_dict.add(self.trees[i].labels[j]))
        return syn_label_dict

    def get_tpf2_dict(self, corpus_tensors, tpf2_dict=None):
        dict_tpf = {}
        if tpf2_dict is None:
            tpf2_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)
        else:
            assert tpf2_dict.accept_new is False

        for tensor in corpus_tensors:
            x, _, sentence_length, _ = tensor
            sentence_id, predicate_id = x[0][0], np.argmax(x[2])

            tree = self.trees[sentence_id]
            assert len(tree.heads) == sentence_length
            root, tree = creatTree(tree.heads)  # head: a sentence's heads; sentence base
            sentence_sub_paths = find_sentence_sub_paths(tree, predicate_id)
            tpf_feature_ids = []
            for i, path_pair in enumerate(sentence_sub_paths):
                if i < predicate_id:
                    word_path, predicate_path = path_pair[0], path_pair[1]
                else:
                    predicate_path, word_path = path_pair[0], path_pair[1]
                predicate_tpf = len(predicate_path) - 1
                word_tpf = len(word_path) - 1
                tpf_string = str(predicate_tpf) + '-' + str(word_tpf)
                idx = tpf2_dict.add(tpf_string)
                tpf_feature_ids.append(idx)
            index = str(sentence_id) + '-' + str(predicate_id)
            if index in dict_tpf:
                print(index)
                exit()
            dict_tpf[index] = tpf_feature_ids
        if tpf2_dict.accept_new is True:
            return dict_tpf, tpf2_dict
        else:
            return dict_tpf


class SyntacticRepresentation(object):
    def __init__(self):
        self.file_name = ""
        self.representations = []

    def read_from_file(self, filename):
        self.file_name = filename
        print("Reading lstm representations from {}".format(self.file_name))
        representation_file = open(self.file_name, 'r')
        if representation_file.closed:
            print("Cannot open the representation file! Please check {}".format(self.file_name))
            exit()
        each_sentence_representations = []
        for line in representation_file:
            if line == '\n' or line == "\r\n":  # new sentence
                self.representations.append(each_sentence_representations)
                each_sentence_representations = []
                continue
            line = line.strip()
            line = line.split('\t')
            line = line[1].split(' ')
            rep = np.asarray(line, dtype=np.float32)
            each_sentence_representations.append(rep)
        representation_file.close()
        print("Load LSTM representations done, total {} sentences' representations".format(len(self.representations)))

    def minus_by_the_predicate(self, corpus_tensors):
        has_processed_sentence_id = {}
        for i, data in enumerate(corpus_tensors):
            sentence_id = data[0][0][0]
            predicates = data[0][2]
            predicate_id = predicates.argmax()
            if sentence_id in has_processed_sentence_id:
                continue
            else:
                has_processed_sentence_id[sentence_id] = 1
            for j in range(1, len(self.representations[sentence_id])):  # Root doesn't use.
                self.representations[sentence_id][j] = self.representations[sentence_id][predicate_id] \
                                                       - self.representations[sentence_id][j]

    def check_math_corpus(self, lengths):
        for i, length in enumerate(lengths):
            if len(self.representations[i]) != length + 1:  # 1 means the first one, Root. Actually never use it.
                print i, length, len(self.representations[i])
                print("sentence {} doesn't match: lstm representation {} vs corpus {}" \
                      .format(i, len(self.representations[i])), length)
                exit()
        print("LSTM representation match the corpus!")
