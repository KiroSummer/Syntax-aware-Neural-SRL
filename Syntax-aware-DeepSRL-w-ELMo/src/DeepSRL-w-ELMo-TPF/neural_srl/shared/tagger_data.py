from constants import UNKNOWN_TOKEN
from constants import PADDING_TOKEN

import numpy as np
import random


def tensorize(sentence, max_length):
    """ Input:
      - sentence: The sentence is a tuple of lists (s1, s2, ..., sk)
            s1 is always a sequence of word ids.
            sk is always a sequence of label ids.
            s2 ... sk-1 are sequences of feature ids,
              such as predicate or supertag features.
      - max_length: The maximum length of sequences, used for padding.
  """
    x = np.array([t for t in zip(*sentence[:-1])])
    y = np.array(sentence[-1])
    weights = (y >= 0).astype(float)
    x = x.transpose()  # SRL: (T, 2) T is the sentence length
    y = y.reshape(1, -1)  # SRL:
    return x, np.absolute(y), len(sentence[0]), weights


class TaggerData(object):
    def __init__(self, config, train_sents, dev_sents, word_dict, label_dict, embeddings, embedding_shapes, \
                 feature_dicts=None):
        self.max_train_length = config.max_train_length
        self.max_dev_length = max([len(s[1]) for s in dev_sents]) if len(dev_sents) > 0 else 0
        self.batch_size = config.batch_size
        self.use_se_marker = config.use_se_marker
        self.padding_id = word_dict.str2idx[PADDING_TOKEN]
        self.unk_id = word_dict.str2idx[UNKNOWN_TOKEN]
        print("padding id {}, unknown word id {}".format(self.padding_id, self.unk_id))

        self.train_lengths = self.get_corpus_sentences_lengths(train_sents)
        self.dev_lengths = self.get_corpus_sentences_lengths(dev_sents)

        self.train_predicates_id = []

        self.train_sents = [s for s in train_sents if len(s[1]) <= self.max_train_length]
        self.dev_sents = dev_sents
        self.word_dict = word_dict
        self.label_dict = label_dict
        self.embeddings = embeddings
        self.embedding_shapes = embedding_shapes
        self.feature_dicts = feature_dicts
        self.syntactic_dict = None
        self.tpf2_dict = None

        self.train_tensors = [tensorize(s, self.max_train_length) for s in self.train_sents]
        self.dev_tensors = [tensorize(s, self.max_dev_length) for s in self.dev_sents]

    def get_corpus_sentences_lengths(self, corpus):
        last_sentence_id = -1
        sentence_lengths = []
        for sentence in corpus:
            current_sentence_id = sentence[0]
            if last_sentence_id != current_sentence_id:
                sentence_lengths.append(len(sentence[1]))
                last_sentence_id = current_sentence_id
        return sentence_lengths

    def get_corpus_predicates_Id(self, corpus):
        last_sentence_id = -1
        sentence_predicate_id = []
        for sentence in corpus:
            current_sentence_id = sentence[0]
            if last_sentence_id != current_sentence_id:
                sentence_predicate_id.append(sentence[0][2])
                last_sentence_id = current_sentence_id
        return sentence_predicate_id

    def get_training_data(self, include_last_batch=False):
        """ Get shuffled training samples. Called at the beginning of each epoch.
    """
        # TODO: Speed up: Use variable size batches (different max length).
        train_ids = range(len(self.train_sents))
        random.shuffle(train_ids)

        if not include_last_batch:
            num_batches = len(train_ids) // self.batch_size
            train_ids = train_ids[:num_batches * self.batch_size]

        num_samples = len(self.train_sents)
        tensors = [self.train_tensors[t] for t in train_ids]
        batched_tensors = [tensors[i: min(i + self.batch_size, num_samples)]
                           for i in xrange(0, num_samples, self.batch_size)]
        results = [zip(*t) for t in batched_tensors]

        print("Extracted {} samples and {} batches.".format(num_samples, len(batched_tensors)))
        return results

    def get_development_data(self, batch_size=None):
        if batch_size is None:
            return self.dev_tensors

        num_samples = len(self.dev_sents)
        batched_tensors = [self.dev_tensors[i: min(i + batch_size, num_samples)]
                           for i in xrange(0, num_samples, batch_size)]
        results = [zip(*t) for t in batched_tensors]
        return results

    def get_test_data(self, test_sentences, batch_size=None):
        max_len = max([len(s[0]) for s in test_sentences])
        num_samples = len(test_sentences)
        # print("Max sentence length: {} among {} samples.".format(max_len, num_samples))
        test_tensors = [tensorize(s, max_len) for s in test_sentences]
        self.test_tensors = test_tensors
        if batch_size is None:
            return test_tensors
        batched_tensors = [test_tensors[i: min(i + batch_size, num_samples)]
                           for i in xrange(0, num_samples, batch_size)]
        return [zip(*t) for t in batched_tensors]
