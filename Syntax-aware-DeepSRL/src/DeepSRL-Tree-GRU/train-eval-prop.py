from neural_srl.shared import *
from neural_srl.shared.tagger_data import TaggerData
from neural_srl.shared.measurements import Timer
from neural_srl.pytorch.tagger import BiLSTMTaggerModel
# from neural_srl.theano.tagger import BiLSTMTaggerModel
from neural_srl.shared.evaluation import PropIdEvaluator, SRLEvaluator, TaggerEvaluator

import argparse
import time
import numpy
import os
import shutil
import torch
import sys


def evaluate_tagger(model, batched_dev_data, evaluator, writer, global_step):
  predictions = None
  dev_loss = 0
  total_correct, total_prop = 0, 0

  model.bilstm.dropout=0.0
  for i, batched_tensor in enumerate(batched_dev_data):
    x, y, _, weights = batched_tensor

    batch_input_lengths = ([sentence_x.shape[0] for sentence_x in x])
    max_length = max(batch_input_lengths)
    # padding
    # input = [numpy.pad(sentence_x, (0, max_length - sentence_x.shape[0]), 'constant') for sentence_x in x]
    word_input = [numpy.pad(sentence_x[:, 0], (0, max_length - sentence_x.shape[0]), 'constant') \
                  for sentence_x in x]  # padding
    predicate_input = [numpy.pad(sentence_x[:, 1], (0, max_length - sentence_x.shape[0]), 'constant') \
                       for sentence_x in x]  # padding
    word_input, predicate_input = numpy.vstack(word_input), numpy.vstack(predicate_input)

    # numpy batch input to Variable
    word_input_seqs = torch.autograd.Variable(torch.from_numpy(word_input.astype('int64')).long())
    predicate_input_seqs = torch.autograd.Variable(torch.from_numpy(predicate_input.astype('int64')).long())

    # First: order the batch by decreasing sequence length
    input_lengths = torch.LongTensor(batch_input_lengths)
    input_lengths, perm_idx = input_lengths.sort(0, descending=True)
    word_input_seqs = word_input_seqs[perm_idx]
    predicate_input_seqs = predicate_input_seqs[perm_idx]
    answer = [None] * len(x)  # resort the answer according to the input
    count = 0
    list_y = list(y)
    for (i, ans) in zip(perm_idx, list_y):
        answer[count] = list_y[i]
        count += 1
    answer = numpy.concatenate(answer)
    answer = torch.autograd.Variable(torch.from_numpy(answer).type(torch.LongTensor))
    answer = answer.view(-1)

    # print answer, answer.size()
    # Then pack the sequences
    # packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths.numpy(), batch_first=True)
    # packed_input = packed_input.cuda() if args.gpu else packed_input
    if args.gpu:
        word_input_seqs, predicate_input_seqs, input_lengths, perm_idx = \
            word_input_seqs.cuda(), predicate_input_seqs.cuda(), input_lengths.cuda(), perm_idx.cuda()
        answer = answer.cuda()
    model.zero_grad()
    output = model.forward(word_input_seqs, predicate_input_seqs, input_lengths, perm_idx,
                           len(x))  # (batch input, batch size)
    loss = model.loss(output, answer)
    """batch_input_lengths = ([sentence_x.shape[0] for sentence_x in x])
    max_length = max(batch_input_lengths)
    input = [numpy.pad(sentence_x, (0, max_length - sentence_x.shape[0]), 'constant') for sentence_x in x]  # padding
    input = numpy.vstack(input)

    input_seqs = torch.autograd.Variable(torch.from_numpy(input.astype('int64')).long())

    # First: order the batch by decreasing sequence length
    input_lengths = torch.LongTensor(batch_input_lengths)
    input_lengths, perm_idx = input_lengths.sort(0, descending=True)
    input_seqs = input_seqs[perm_idx]
    answer = [None] * len(x)  # resort the answer according to the input
    count = 0
    list_y = list(y)
    for (i, ans) in zip(perm_idx, list_y):
        answer[count] = list_y[i]
        count += 1
    answer = numpy.concatenate(answer)
    answer = torch.autograd.Variable(torch.from_numpy(answer).type(torch.LongTensor))

    # print answer, answer.size()
    # Then pack the sequences
    # packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths.numpy(), batch_first=True)
    # packed_input = packed_input.cuda() if args.gpu else packed_input
    if args.gpu:
        input_seqs, input_lengths, perm_idx = input_seqs.cuda(), input_lengths.cuda(), perm_idx.cuda()
        answer = answer.cuda()
    model.zero_grad()
    output = model.forward(input_seqs, input_lengths, perm_idx, len(x))  # (batch input, batch size)
    loss = model.loss(output, answer)"""
    dev_loss += float(loss)

    if args.gpu:
        p = output.data.cpu().numpy()
    else:
        p = output.data.numpy()
    p = numpy.argmax(p, axis=1)
    batch_tokens_size = answer.size()[0]
    p = p.reshape(batch_tokens_size)
    correct = numpy.dot(p, answer)
    dinominator = numpy.dot(answer, numpy.ones(answer.shape[0]))
    total_correct += int(correct)
    total_prop += int(dinominator)

    """for item in zip(x, y, weights):  # a batch
        input = torch.LongTensor(item[0])
        answer = torch.autograd.Variable(torch.from_numpy(item[1]).type(torch.LongTensor))
        answer = answer.view(-1)
        input = input.cuda() if args.gpu else input
        answer = answer.cuda() if args.gpu else answer

        output = model.forward(input)
        # print output
        loss = model.loss(output, answer)
        dev_loss += float(loss)
        if args.gpu:
            p = output.data.cpu().numpy()
        else:
            p = output.data.numpy()
        p = numpy.argmax(p, axis=1)
        p = p.reshape(input.size()[1])
        # print input.size()[1]
        correct = numpy.dot(p, answer)
        dinominator = numpy.dot(answer, item[2])
        total_correct += int(correct)
        total_prop += int(dinominator)
        # print "p shape", p.shape
        # predictions = numpy.concatenate((predictions, p), axis=0) if i > 0 else p"""


  print ('Dev loss={:.6f}'.format(dev_loss))
  # evaluator.evaluate(predictions)
  evaluator.accuracy = float(100.0 * total_correct / total_prop)
  print total_correct, " / ", total_prop, " = ", evaluator.accuracy
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
  torch.cuda.manual_seed(666)
  ### gpu
  gpu = torch.cuda.is_available()
  print("GPU available: ", gpu)

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
      # Data and evaluator for PropId.
      data = TaggerData(config,
                        *reader.get_postag_data(config, args.train, args.dev, vocab_path, label_path))
      evaluator = PropIdEvaluator(data.get_development_data(),
                                  data.label_dict)

    batched_dev_data = data.get_development_data(batch_size=config.dev_batch_size)
    print ('Dev data has {} batches.'.format(len(batched_dev_data)))
  
  with Timer('Preparation'):
    if not os.path.isdir(args.model):
      print ('Directory {} does not exist. Creating new.'.format(args.model))
      os.makedirs(args.model)
    else:
      if len(os.listdir(args.model)) > 0:
        print ('[WARNING] Log directory {} is not empty, previous checkpoints might be overwritten'
             .format(args.model))
    shutil.copyfile(args.config, os.path.join(args.model, 'config'))
    # Save word and label dict to model directory.
    data.word_dict.save(os.path.join(args.model, 'word_dict'))
    data.label_dict.save(os.path.join(args.model, 'label_dict'))
    writer = open(os.path.join(args.model, 'checkpoints.tsv'), 'w')
    writer.write('step\tdatetime\tdev_loss\tdev_accuracy\tbest_dev_accuracy\n')

  with Timer('Building model'):
    model = BiLSTMTaggerModel(data, config=config, gpu_id=args.gpu)
    if args.gpu:
        print "Use Cuda!"
        model = model.cuda()
    if args.gpu != "" and not torch.cuda.is_available():
        raise Exception("No GPU Found!")
        exit()
    for param in model.parameters():
        print param.size()
    """for param in model.params:
      print param, param.name, param.shape.eval()
    loss_function = model.get_loss_function()
    eval_function = model.get_eval_function()"""

  while epoch < config.max_epochs:
    with Timer("Epoch%d" % epoch) as timer:
      train_data = data.get_training_data(include_last_batch=True)
      model.bilstm.dropout = 0.1
      for batched_tensor in train_data:  # for each batch in the training corpus
        x, y, _, weights = batched_tensor

        batch_input_lengths = ([sentence_x.shape[0] for sentence_x in x])
        max_length = max(batch_input_lengths)
        # padding
        # input = [numpy.pad(sentence_x, (0, max_length - sentence_x.shape[0]), 'constant') for sentence_x in x]
        word_input = [numpy.pad(sentence_x[:, 0], (0, max_length - sentence_x.shape[0]), 'constant') \
                 for sentence_x in x]  # padding
        predicate_input = [numpy.pad(sentence_x[:, 1], (0, max_length - sentence_x.shape[0]), 'constant') \
                      for sentence_x in x]  # padding
        word_input, predicate_input = numpy.vstack(word_input), numpy.vstack(predicate_input)

        # numpy batch input to Variable
        word_input_seqs = torch.autograd.Variable(torch.from_numpy(word_input.astype('int64')).long())
        predicate_input_seqs = torch.autograd.Variable(torch.from_numpy(predicate_input.astype('int64')).long())

        # First: order the batch by decreasing sequence length
        input_lengths = torch.LongTensor(batch_input_lengths)
        input_lengths, perm_idx = input_lengths.sort(0, descending=True)
        word_input_seqs = word_input_seqs[perm_idx]
        predicate_input_seqs = predicate_input_seqs[perm_idx]
        answer = [None] * len(x)  # resort the answer according to the input
        count = 0
        list_y = list(y)
        for (i, ans) in zip(perm_idx, list_y):
            answer[count] = list_y[i]
            count += 1
        answer = numpy.concatenate(answer)
        answer = torch.autograd.Variable(torch.from_numpy(answer).type(torch.LongTensor))
        answer = answer.view(-1)

        # print answer, answer.size()
        # Then pack the sequences
        # packed_input = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths.numpy(), batch_first=True)
        # packed_input = packed_input.cuda() if args.gpu else packed_input
        if args.gpu:
            word_input_seqs, predicate_input_seqs, input_lengths, perm_idx =\
                word_input_seqs.cuda(), predicate_input_seqs.cuda(), input_lengths.cuda(), perm_idx.cuda()
            answer = answer.cuda()
        model.zero_grad()
        output = model.forward(word_input_seqs, predicate_input_seqs, input_lengths, perm_idx, len(x))  # (batch input, batch size)
        loss = model.loss(output, answer)
        loss.backward()
        # gradient clipping
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        model.optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
        model.optimizer.step()

        train_loss += loss
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
        evaluate_tagger(model, batched_dev_data, evaluator, writer, global_step)

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

  parser.add_argument('--dev',
                      type=str,
                      default='',
                      required=True,
                      help='Path to the devevelopment data, which is a single file in the sequential tagging format.')

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

