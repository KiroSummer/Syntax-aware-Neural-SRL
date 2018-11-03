from neural_srl.shared import *
from neural_srl.shared.tagger_data import TaggerData
from neural_srl.shared.measurements import Timer
from neural_srl.pytorch.tagger import BiLSTMTaggerModel
from neural_srl.pytorch.util import batch_data_variable
from neural_srl.shared.evaluation import PropIdEvaluator, SRLEvaluator, TaggerEvaluator
from neural_srl.shared.syntactic_extraction import *
from neural_srl.pytorch.util import position_encoding_init

import argparse
import time
import numpy
import os
import shutil
import torch
from torch.nn.functional import log_softmax, softmax
import sys
# -*- coding: utf-8 -*-


def evaluate_tagger(model, batch_position_encoding, batched_dev_data, dev_tpf2, evaluator, writer, global_step):
    predictions = []
    dev_loss = 0
    total_correct, total_prop = 0, 0

    model.eval()
    for i, batched_tensor in enumerate(batched_dev_data):
        x, y, lengths, weights = batched_tensor
        word_inputs_seqs, predicate_inputs_seqs, tpf2_inputs_seqs, pes, answers, input_lengths, masks, padding_answers = \
            batch_data_variable(dev_tpf2, None, x, y, lengths, weights)

        if args.gpu:
            word_inputs_seqs, predicate_inputs_seqs, tpf2_inputs_seqs, input_lengths, masks, \
            padding_answers = \
                word_inputs_seqs.cuda(), predicate_inputs_seqs.cuda(), tpf2_inputs_seqs.cuda(), \
                input_lengths.cuda(), masks.cuda(), padding_answers.cuda()

        output = model.forward(word_inputs_seqs, predicate_inputs_seqs, tpf2_inputs_seqs, pes, input_lengths)
        loss = model.compute_loss(output, padding_answers, masks)

        dev_loss += float(loss.data)  # accumulate the dev loss
        output = torch.cat([output[i][:actual_length].view(actual_length, -1) for i, actual_length in \
                            enumerate(input_lengths)], dim=0)
        if args.gpu:  # convert Variable to numpy
            p = output.data.cpu().numpy()
        else:
            p = output.data.numpy()
        p = numpy.argmax(p, axis=1)
        batch_tokens_size = sum(lengths)
        assert p.shape[0] == batch_tokens_size
        np_answer = numpy.concatenate(answers)
        correct = numpy.equal(p, np_answer).sum()
        denominator = batch_tokens_size  # prop numpy.dot(answer, numpy.ones(answer.shape[0]))
        total_correct += int(correct)
        total_prop += int(denominator)

        # split the output of the Model according the order
        last_index, batch_p = 0, []
        for length in input_lengths:
            batch_p.append(p[last_index:last_index + length])
            last_index += length
        predictions.extend(batch_p)

    print ('Dev loss={:.6f}'.format(dev_loss))
    evaluator.evaluate(predictions)
    print total_correct, " / ", total_prop, " = ", 100.0 * total_correct / total_prop
    if evaluator.accuracy > evaluator.best_accuracy:
        evaluator.best_accuracy = evaluator.accuracy
    writer.write('{}\t{}\t{:.6f}\t{:.2f}\t{:.2f}\n'.format(global_step,
                                                       time.strftime("%Y-%m-%d %H:%M:%S"),
                                                       float(dev_loss),
                                                       float(evaluator.accuracy),
                                                       float(evaluator.best_accuracy)))
    writer.flush()
    if evaluator.has_best:
        model.save(os.path.join(args.model, 'model'))


def train_tagger(args):
    config = configuration.get_config(args.config)
    numpy.random.seed(666)
    torch.manual_seed(666)
    torch.set_printoptions(precision=20)
    ### gpu
    gpu = torch.cuda.is_available()
    if args.gpu and gpu:
        print("GPU available: {}\t GPU ID: {}".format(gpu, args.gpu))
        torch.cuda.manual_seed(666)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    i = 0
    global_step = 0
    epoch = 0
    train_loss = 0.0

    with Timer('Data loading'):
        vocab_path = args.vocab if args.vocab != '' else None
        label_path = args.labels if args.labels != '' else None
        gold_props_path = args.gold if args.gold != '' else None

        print ('Task: {}'.format(args.task))
        if args.task == 'srl':
            # Data and evaluator for SRL.
            data = TaggerData(config,
                              *reader.get_srl_data(config, args.train, args.dev, vocab_path, label_path))
            evaluator = SRLEvaluator(data.get_development_data(),
                                     data.label_dict,
                                     gold_props_file=gold_props_path,
                                     use_se_marker=config.use_se_marker,
                                     pred_props_file=None,
                                     word_dict=data.word_dict)
        else:
            print "Not implemented yet!"
            exit()
            # Data and evaluator for PropId.
            data = TaggerData(config,
                              *reader.get_postag_data(config, args.train, args.dev, vocab_path, label_path))
            evaluator = PropIdEvaluator(data.get_development_data(),
                                        data.label_dict)

        batched_dev_data = data.get_development_data(batch_size=config.dev_batch_size)
        print ('Dev data has {} batches.'.format(len(batched_dev_data)))

    with Timer('Syntactic Information Extracting'):  # extract the syntactic information from file
        train_dep_trees = SyntacticCONLL()
        dev_dep_trees = SyntacticCONLL()
        train_dep_trees.read_from_file(args.train_dep_trees)
        dev_dep_trees.read_from_file(args.dev_dep_trees)

    with Timer("TPF2 generating..."):
        # generate the tree-based position features according the Dependency Tree.
        train_tpf2, data.tpf2_dict = train_dep_trees.get_tpf2_dict(data.train_tensors)
        print("Extract {} training TPF2 features".format(len(train_tpf2)))
        assert len(train_tpf2) == len(data.train_tensors)
        data.tpf2_dict.accept_new = False
        dev_tpf2 = dev_dep_trees.get_tpf2_dict(data.dev_tensors, data.tpf2_dict)
        print("Extract {} dev TPF2 features".format(len(dev_tpf2)))
        assert len(dev_tpf2) == len(data.dev_tensors)

    with Timer('Preparation'):
        if not os.path.isdir(args.model):
            print ('Directory {} does not exist. Creating new.'.format(args.model))
            os.makedirs(args.model)
        else:
            if len(os.listdir(args.model)) > 0:
                print ('[WARNING] Log directory {} is not empty, previous checkpoints might be overwritten' \
                    .format(args.model))
        shutil.copyfile(args.config, os.path.join(args.model, 'config'))
        # Save word and label dict to model directory.
        data.word_dict.save(os.path.join(args.model, 'word_dict'))
        data.label_dict.save(os.path.join(args.model, 'label_dict'))
        # data.syntactic_dict.save(os.path.join(args.model, 'syn_label_dict'))
        data.tpf2_dict.save(os.path.join(args.model, 'tpf2_dict'))
        writer = open(os.path.join(args.model, 'checkpoints.tsv'), 'w')
        writer.write('step\tdatetime\tdev_loss\tdev_accuracy\tbest_dev_accuracy\n')

    with Timer('Building model'):
        model = BiLSTMTaggerModel(data, config=config, gpu_id=args.gpu)
        if args.gpu:
            print "BiLSTMTaggerModel initialize with Cuda!"
            model = model.cuda()
            if args.gpu != "" and not torch.cuda.is_available():
                raise Exception("No GPU Found!")
                exit()
        for param in model.parameters():
            print param.size()

    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.95)  # initialize the optimizer outside the epoch
    # batch_position_encoding = position_encoding_init(200, 100)  # 0: root, 1, ...n, n+1: padding | max length 200
    while epoch < config.max_epochs:
        with Timer("Epoch%d" % epoch) as timer:
            model.train()
            train_data = data.get_training_data(include_last_batch=True)
            for batched_tensor in train_data:  # for each batch in the training corpus
                x, y, lengths, weights = batched_tensor
                word_inputs_seqs, predicate_inputs_seqs, tpf2_inputs_seqs, pes, _, input_lengths, masks, padding_answers = \
                    batch_data_variable(train_tpf2, None, x, y, lengths, weights)

                if args.gpu:
                    word_inputs_seqs, predicate_inputs_seqs, tpf2_inputs_seqs, input_lengths, masks, \
                        padding_answers = \
                        word_inputs_seqs.cuda(), predicate_inputs_seqs.cuda(), tpf2_inputs_seqs.cuda(), \
                        input_lengths.cuda(), masks.cuda(), padding_answers.cuda()

                optimizer.zero_grad()
                output = model.forward(word_inputs_seqs, predicate_inputs_seqs, tpf2_inputs_seqs, pes, input_lengths)
                loss = model.compute_loss(output, padding_answers, masks)
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.data  # should be tensor not Variable, avoiding the graph accumulates

                i += 1
                global_step += 1
                if i % 400 == 0:
                    timer.tick("{} training steps, loss={:.3f}".format(i, float(train_loss / i)))

            train_loss = train_loss / i
            print("Epoch {}, steps={}, loss={:.3f}".format(epoch, i, float(train_loss)))
            i = 0
            epoch += 1
            train_loss = 0.0
            if epoch % config.checkpoint_every_x_epochs == 0:
                with Timer('Evaluation'):
                    evaluate_tagger(model, None, batched_dev_data, dev_tpf2, evaluator, writer, global_step)

    # Done. :)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',
                        type=str,
                        default='',
                        required=True,
                        help='Config file for the neural architecture and hyper-parameters.')

    parser.add_argument('--model',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the directory for saving model and checkpoints.')

    parser.add_argument('--train',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the training data, which is a single file in sequential tagging format.')

    parser.add_argument('--train_lstm',
                        type=str,
                        default='',
                        required=False,
                        help='Path to the training lstm representation, optional')

    parser.add_argument('--train_dep_trees',
                        type=str,
                        default='',
                        required=False,
                        help='Path to the training auto dep trees, optional')

    parser.add_argument('--dev',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the devevelopment data, which is a single file in the sequential tagging format.')

    parser.add_argument('--dev_lstm',
                       type=str,
                       default='',
                       required=False,
                       help='Path to the devevelopment lstm representation, optional')

    parser.add_argument('--dev_dep_trees',
                        type=str,
                        default='',
                        required=False,
                        help='Path to the dev auto dep trees, optional')

    parser.add_argument('--task',
                        type=str,
                        help='Training task (srl or propid). Default is srl.',
                        default='srl',
                        choices=['srl', 'propid'])

    parser.add_argument('--gold',
                        type=str,
                        default='',
                        help='(Optional) Path to the file containing gold propositions (provided by CoNLL shared task).')

    parser.add_argument('--vocab',
                        type=str,
                        default='',
                        help='(Optional) A file containing the pre-defined vocabulary mapping. Each line contains a text string for the word mapped to the current line number.')

    parser.add_argument('--labels',
                        type=str,
                        default='',
                        help='(Optional) A file containing the pre-defined label mapping. Each line contains a text string for the label mapped to the current line number.')

    parser.add_argument('--gpu',
                        type=str,
                        default="",
                        help='(Optional) A argument that specifies the GPU id. Default use the cpu')
    args = parser.parse_args()
    train_tagger(args)

