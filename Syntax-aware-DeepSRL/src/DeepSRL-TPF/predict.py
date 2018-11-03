''' Predict and output scores.

   - Reads model param file.
   - Runs data.
   - Remaps label indices.
   - Outputs protobuf file.
'''

from neural_srl.shared import *
from neural_srl.shared.constants import *
from neural_srl.shared.dictionary import Dictionary
from neural_srl.shared.inference import *
from neural_srl.shared.io_utils import *
from neural_srl.shared.tagger_data import TaggerData
from neural_srl.shared.measurements import Timer
from neural_srl.shared.evaluation import PropIdEvaluator, SRLEvaluator
from neural_srl.shared.tensor_pb2 import *
from neural_srl.shared.scores_pb2 import *
from neural_srl.pytorch.tagger import BiLSTMTaggerModel
from neural_srl.pytorch.util import batch_data_variable
from neural_srl.shared.syntactic_extraction import *
from neural_srl.shared.numpy_saver import *

import argparse
import numpy
import os
import sys


def get_scores(config, task, model_path, word_dict_path, label_dict_path, tpf2_dict_path, input_path):
    with Timer('Data loading'):
        print ('Task: {}'.format(task))
        allow_new_words = True
        print ('Allow new words in test data: {}'.format(allow_new_words))

        # Load word and tag dictionary
        word_dict = Dictionary(padding_token=PADDING_TOKEN, unknown_token=UNKNOWN_TOKEN)  # word tokens to Dict
        label_dict = Dictionary()
        tpf2_dict = Dictionary()
        word_dict.load(word_dict_path)
        label_dict.load(label_dict_path)
        tpf2_dict.load(tpf2_dict_path)
        data = TaggerData(config, [], [], word_dict, label_dict, None, None)
        data.tpf2_dict = tpf2_dict

        # Load test data.
        if task == 'srl':
            test_sentences, emb_inits, emb_shapes = reader.get_srl_test_data(
                input_path,
                config,
                data.word_dict,
                data.label_dict,
                allow_new_words)
        else:
            test_sentences, emb_inits, emb_shapes = reader.get_postag_test_data(
                input_path,
                config,
                data.word_dict,
                data.label_dict,
                allow_new_words)

        print ('Read {} sentences.'.format(len(test_sentences)))

        # Add pre-trained embeddings for new words in the test data.
        # if allow_new_words:
        data.embedding_shapes = emb_shapes
        data.embeddings = emb_inits
        # Batching.
        test_data = data.get_test_data(test_sentences, batch_size=config.dev_batch_size)

    with Timer('Syntactic Information Extracting'):  # extract the syntactic information from file
        test_dep_trees = SyntacticCONLL()
        test_dep_trees.read_from_file(args.test_dep_trees)

    with Timer("TPF2 generating..."):
        # generate the tree-based position features according the Dependency Tree.
        data.tpf2_dict.accept_new = False
        test_tpf2 = test_dep_trees.get_tpf2_dict(data.test_tensors, data.tpf2_dict)
        print("Extract {} test TPF2 features".format(len(test_tpf2)))
        assert len(test_tpf2) == len(data.test_tensors)

    with Timer('Model building and loading'):
        model = BiLSTMTaggerModel(data, config=config, gpu_id=args.gpu)
        model.load(model_path)
        for param in model.parameters():
            print param.size()
        if args.gpu:
            print("Initialize the model with GPU!")
            model = model.cuda()

    with Timer('Running model'):
        scores = []
        model.eval()
        for i, batched_tensor in enumerate(test_data):
            x, y, lengths, weights = batched_tensor
            word_inputs_seqs, predicate_inputs_seqs, tpf2_inputs_seqs, pes, answers, input_lengths, masks, padding_answers = \
                batch_data_variable(test_tpf2, None, x, y, lengths, weights)

            if args.gpu:
                word_inputs_seqs, predicate_inputs_seqs, tpf2_inputs_seqs, input_lengths, masks, padding_answers = \
                    word_inputs_seqs.cuda(), predicate_inputs_seqs.cuda(), tpf2_inputs_seqs.cuda(), input_lengths.cuda(), \
                    masks.cuda(), padding_answers.cuda()

            sc = model.forward(word_inputs_seqs, predicate_inputs_seqs, tpf2_inputs_seqs, pes, input_lengths)
            sc = sc.data.cpu().numpy() if args.gpu else sc.data.numpy()
            sc = [sc[j] for j in range(sc.shape[0])]
            scores.extend(sc)

    return scores, data, test_sentences, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the model directory.')

    parser.add_argument('--input',
                        type=str,
                        default='',
                        required=True,
                        help='Path to the input file path (sequetial tagging format).')

    parser.add_argument('--test_dep_trees',
                        type=str,
                        default='',
                        required=False,
                        help='Path to the test auto dep trees, optional')

    parser.add_argument('--input_scores',
                        type=str,
                        default='xxx.npy',
                        required=False,
                        help='Path to the input file scores, optional')

    parser.add_argument('--task',
                        type=str,
                        help='Training task (srl or propid). Default is srl.',
                        default='srl',
                        choices=['srl', 'propid'])

    parser.add_argument('--gold',
                        type=str,
                        default='',
                        help='(Optional) Path to the file containing gold propositions (provided by CoNLL shared task).')

    parser.add_argument('--inputprops',
                        type=str,
                        default='',
                        help='(Optional) Path to the predicted predicates in CoNLL format. Ignore if using gold predicates.')

    parser.add_argument('--output',
                        type=str,
                        default='',
                        help='(Optional) Path for output predictions.')

    parser.add_argument('--outputprops',
                        type=str,
                        default='',
                        help='(Optional) Path for output predictions in CoNLL format. Only used when task is {propid}.')

    parser.add_argument('--proto',
                        type=str,
                        default='',
                        help='(Optional) Path to the proto file path (for reusing predicted scores).')
    parser.add_argument('--gpu',
                        type=str,
                        default="",
                        help='(Optional) A argument that specifies the GPU id. Default use the cpu')

    args = parser.parse_args()
    config = configuration.get_config(os.path.join(args.model, 'config'))

    # Detect available ensemble models.
    num_ensemble_models = 1
    for i in range(20):
        model_path = os.path.join(args.model, 'model{}'.format(i))
        if os.path.exists(model_path):
            num_ensemble_models = i + 1
        else:
            break
    if num_ensemble_models == 1:
        print ('Using single model.')
    else:
        print ('Using an ensemble of {} models'.format(num_ensemble_models))

    ensemble_scores = None
    for i in range(num_ensemble_models):
        if num_ensemble_models == 1:
            model_path = os.path.join(args.model, 'model')
            word_dict_path = os.path.join(args.model, 'word_dict')
        else:
            model_path = os.path.join(args.model, 'model{}.npz'.format(i))
            word_dict_path = os.path.join(args.model, 'word_dict{}'.format(i))
        label_dict_path = os.path.join(args.model, 'label_dict')
        tpf2_dict_path = os.path.join(args.model, 'tpf2_dict')
        print("model dict path: {}, word dict path: {}, label dict path: {}".format( \
            model_path, word_dict_path, label_dict_path))

        # Compute local scores.
        scores, data, test_sentences, test_data = get_scores(config,
                                                             args.task,
                                                             model_path,
                                                             word_dict_path,
                                                             label_dict_path,
                                                             tpf2_dict_path,
                                                             args.input)
        ensemble_scores = numpy.add(ensemble_scores, scores) if i > 0 else scores

    if num_ensemble_models == 1:  # save the results.
        score_saver = NumpySaver(ensemble_scores)
        save_path = os.path.join(args.input_scores)
        score_saver.save(save_path)

    # Getting evaluator
    gold_props_file = args.gold if args.gold != '' else None
    pred_props_file = args.inputprops if args.inputprops != '' else None

    if args.task == 'srl':
        evaluator = SRLEvaluator(data.get_test_data(test_sentences, batch_size=None),
                                 data.label_dict,
                                 gold_props_file,
                                 use_se_marker=config.use_se_marker,
                                 pred_props_file=pred_props_file,
                                 word_dict=data.word_dict)
    else:
        evaluator = PropIdEvaluator(data.get_test_data(test_sentences, batch_size=None),
                                    data.label_dict)

    if args.proto != '':
        print 'Writing to proto {}'.format(args.proto)
        pb_file = open(args.proto, 'wb')
    else:
        pb_file = None

    with Timer("Decoding"):
        transition_params = get_transition_params(data.label_dict.idx2str)
        num_tokens = None

        # Collect sentence length information
        for (i, batched_tensor) in enumerate(test_data):
            _, _, nt, _ = batched_tensor
            num_tokens = numpy.concatenate((num_tokens, nt), axis=0) if i > 0 else nt

        # Decode.
        if num_ensemble_models > 1:
            ensemble_scores = numpy.divide(ensemble_scores, 1.0 * num_ensemble_models)

        predictions = []
        line_counter = 0
        for i, slen in enumerate(num_tokens):
            sc = ensemble_scores[i][:slen, :]

            if args.task == 'srl':  # viterbi decode in SRL
                pred, _ = viterbi_decode(sc, transition_params)
            else:
                pred = numpy.argmax(sc, axis=1)

            batch_pred = numpy.array(pred)
            predictions.append(batch_pred)

            # Construct protobuf message
            if pb_file != None:
                sample_id = line_counter
                sent_sc = SentenceScoresProto(sentence_id=sample_id,
                                              scores=TensorProto(dimensions=DimensionsProto(
                                                  dimension=sentence_scores.shape),
                                                  value=sentence_scores.flatten()))
                write_delimited_to(pb_file, sent_sc)
            line_counter += 1

        if pb_file != None:
            pb_file.close()
    # Evaluate
    # predictions = numpy.stack(predictions, axis=0)
    evaluator.evaluate(predictions)

    if args.task == 'srl' and args.output != '':
        print ('Writing to human-readable file: {}'.format(args.output))
        # _, _, nt, _ = evaluator.data
        print_to_readable(predictions, -1, data.label_dict, args.input, args.output)

    if args.task == 'propid':
        write_predprops_to(predictions, data.label_dict, args.input, args.output, args.gold,
                           args.outputprops)
