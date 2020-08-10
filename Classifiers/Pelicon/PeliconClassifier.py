"""
PeliconClassifier.py: It trains or tests a model.

Usage:
    PeliconClassifier.py [options]

Options:
    --train-path=<file>                         training file
    --test-path=<file>                          testing file
    --classifier-name=<str>                     name of classifier
    --model-path=<file>                         training model file
    --export-results-path=<file>                testing report file
"""

#===========================#
#        Imports            #
#===========================#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import os.path
from docopt import docopt
import joblib
import os
import sys

#===========================#
#        Variables          #
#===========================#

# This is to fix the error: `RecursionError: maximum recursion depth exceeded in comparison`
# https://github.com/nltk/nltk/issues/1971
sys.setrecursionlimit(100000)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#===========================#
#          Classes          #
#===========================#

class InputExample(object):
    '''A single training/test example for simple sequence classification.'''

    def __init__(self, guid, text_a, text_b=None, label=None):
        '''Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        '''
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    '''A single set of features of data.'''

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    '''Base class for data converters for sequence classification data sets.'''

    def get_train_examples(self, data_dir):
        '''Gets a collection of `InputExample`s for the train set.'''
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        '''Gets a collection of `InputExample`s for the dev set.'''
        raise NotImplementedError()

    def get_labels(self):
        '''Gets the list of labels for this data set.'''
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):

        # '''Reads a tab separated value file.'''
        # with open(input_file, 'r', encoding='utf-8') as f:
        #     reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
        #     lines = []
        #     for line in reader:
        #         lines.append(line)
        #     return lines

        '''Or use pandas instead, like a normal human being, to avoid a gazillion of errors.'''
        lines = []
        dataset_df = pd.read_csv(input_file, sep='\t')
        # Iterate over each row
        for index, rows in dataset_df.iterrows():
            item = [rows.text, str(rows.label)]
            lines.append(item)
        return lines


class SemEvalProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        '''See base class.'''
        # logger.info('Reading training examples from: {}'.format(data_dir))
        return self._create_examples(self._read_csv(data_dir), 'train')

    def get_eval_examples(self, data_dir):
        '''See base class.'''
        return self._create_examples(self._read_csv(data_dir), 'eval')

    # def get_test_examples(self, data_dir):
    #     return self._create_examples(self._read_csv(data_dir), 'test')

    def get_labels(self):
        '''See base class.'''
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        '''Creates examples for the training and dev sets.'''
        examples = []
        # if set_type == 'dev' or set_type == 'test':
        if set_type == 'train' or set_type == 'eval':
            for (i, line) in enumerate(lines):

                if i == 0:
                    continue
                guid = '%s-%s' % (set_type, i)
                text_a = line[0]
                text_b = None
                # label = '0' if line[1] == 'OFF' else '1'
                # label = '1' if line[1] == '1' else '0'
                label = line[1]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        elif set_type == 'test':
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                guid = line[0]
                text_a = line[1]
                text_b = None
                label = '0'
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples


#===========================#
#        Functions          #
#===========================#

def get_name_of_train_file():
    """
    Returns the name of the train dataset.
    :return: string
    """
    return str(str(os.path.basename(args['--model-path'])).split(args['--classifier-name'])[1]).split('.model')[0]


def get_name_of_test_file():
    """
    Returns the name of the test dataset.
    :return: string
    """
    return str(os.path.basename(args['--test-path'])).split('Dataset')[0]


def write_analytical_results(classifier_name, texts, y_preds):
    """
    Creates an analytical report to be used later for the inter-agreement calculation of the classifiers.
    :param classifier_name:     name of the classifier
    :param texts:               test file
    :param y_preds:             predictions file
    """
    column_name = classifier_name + get_name_of_train_file() + get_name_of_test_file()                                                      # Get the name of column
    file_name = classifier_name + '/' + classifier_name + get_name_of_train_file() + get_name_of_test_file() + 'AnalyticalReport.csv'       # Get the name of the export file
    texts_df = pd.DataFrame(texts)                                                                                                         # Convert pd.Series into pd.Dataframe
    texts_df = texts_df.drop(texts_df.index[[0]])                                                                                           # Remove the first item since it was removed from the testing dataset
    predictions_df = pd.DataFrame(pd.Series(y_preds, name=column_name))                                                                     # Convert numpy.darray into pd.Dataframe
    predictions_df.index += 1                                                                                                               # Fix the index discrepancy
    final_df = pd.concat([texts_df, predictions_df], axis=1)                                                                                # Concat the dataframes
    final_df.to_csv(file_name, sep='\t', encoding='utf-8', index=False)                                                                     # Write dataframe to .csv
    print('======> Analytical report saved to file: ' + str(file_name))                                                                     # Inform the user


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    '''Loads a data file into a list of `InputBatch`s.'''

    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}
    else:
        label_map = {'0': i for i in range(len(examples))}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with '- 3'
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with '- 2'
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where 'type_ids' are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the 'sentence vector'. Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        # if ex_index < 5:
        #     print('\n__________________________________')
        #     logger.info('*** Example ***')
        #     logger.info('guid: %s' % (example.guid))
        #     logger.info('tokens: %s' % ' '.join([str(x) for x in tokens]))
        #     logger.info('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
        #     logger.info('input_mask: %s' % ' '.join([str(x) for x in input_mask]))
        #     logger.info('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
        #     logger.info('label: %s (id = %d)' % (example.label, label_id))
        #     print('__________________________________\n')

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    '''Truncates a sequence pair in place to the maximum length.'''

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def train(args):

    train_path = args['--train-path']
    model_path = args['--model-path']
    classifier_name = args['--classifier-name']

    # ------------------------------------------------------------------- #
    #   Parser values
    # ------------------------------------------------------------------- #

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--train-path',
                        # required=True,
                        type=str,
                        help='The training data path. Should contain the .csv files for the task.',
                        # default=TRAINING_PATH
                        )

    parser.add_argument('--model-path',
                        # required=True,
                        type=str,
                        help='The path of the model.',
                        )

    parser.add_argument('--bert_model',
                        default='bert-large-uncased',
                        type=str,
                        help='Bert pre-trained model selected in the list: bert-base-uncased, '
                             'bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, '
                             'bert-base-multilingual-cased, bert-base-chinese.')

    parser.add_argument('--task_name',
                        default='SEMEVAL',
                        type=str,
                        help='The name of the task to train.')

    parser.add_argument('--classifier-name',
                        default=classifier_name,
                        type=str,
                        help='Name of the classifier.')

    ## Other parameters
    parser.add_argument('--max_seq_length',
                        default=128,
                        type=int,
                        help='The maximum total input sequence length after WordPiece tokenization. \n'
                             'Sequences longer than this will be truncated, and sequences shorter \n'
                             'than this will be padded.')

    parser.add_argument('--do_train',
                        action='store_true',
                        default=True,
                        help='Whether to run training.')

    parser.add_argument('--do_lower_case',
                        action='store_true',
                        default=True,
                        help='Set this flag if you are using an uncased model.')

    parser.add_argument('--train_batch_size',
                        default=8,
                        type=int,
                        help='Total batch size for training.')

    parser.add_argument('--eval_batch_size',
                        default=8,
                        type=int,
                        help='Total batch size for eval.')

    parser.add_argument('--learning_rate',
                        default=2e-5,
                        type=float,
                        help='The initial learning rate for Adam.')

    parser.add_argument('--num_train_epochs',
                        default=3.0,
                        type=float,
                        help='Total number of training epochs to perform.')

    parser.add_argument('--warmup_proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10%% of training.')

    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='Whether not to use CUDA when available')

    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local_rank for distributed training on gpus')

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed for initialization')

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')

    parser.add_argument('--fp16',
                        action='store_true',
                        help='Whether to use 16-bit float precision instead of 32-bit')

    parser.add_argument('--loss_scale',
                        type=float,
                        default=0,
                        help='Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n'
                             '0 (default value): dynamic loss scaling.\n'
                             'Positive power of 2: static loss scaling value.\n')

    # ------------------------------------------------------------------- #
    #   Checking the validity of parameters etc.
    # ------------------------------------------------------------------- #

    args = parser.parse_args()

    processors = {
        'semeval': SemEvalProcessor,
    }
    num_labels_task = {
        'semeval': 2,
    }  # Set the number of labels

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    # logger.info('device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'.format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError('Invalid gradient_accumulation_steps parameter: {}, should be >= 1'.format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError('At least one of `do_train` or `do_eval` must be True.')

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError('Output directory ({}) already exists and is not empty.'.format(args.output_dir))
    # os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(train_path)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              args.local_rank),
                                                          num_labels=num_labels)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                'Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.')

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                'Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.')

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    # print('Parameters Check: Passed.')

    # ------------------------------------------------------------------- #
    #   Now the training starts
    # ------------------------------------------------------------------- #

    # Skip if the trained model already exists
    if not os.path.isfile(model_path):
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        if args.do_train:
            train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)

            # print('\n__________________________________')
            # logger.info('Running Training: ')
            # logger.info('    Num examples = %d', len(train_examples))
            # logger.info('    Batch size = %d', args.train_batch_size)
            # logger.info('    Num steps = %d', num_train_steps)
            # print('__________________________________\n')

            all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
            train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

            model.train()
            for _ in trange(int(args.num_train_epochs), desc='Epoch'):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    loss = model(input_ids, segment_ids, input_mask, label_ids)
                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # modify learning rate with special warm up BERT uses
                        lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

        # ------------------------------------------------------------------- #
        #   Save the trained model
        # ------------------------------------------------------------------- #

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = model_path
        if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)
        # print('\nModel saved to:', str(output_model_file))

    else:
        print('======> Trained model already exists.')


def test(args):

    test_path = args['--test-path']
    model_path = args['--model-path']
    output_path = args['--export-results-path']
    classifier_name = args['--classifier-name']

    # Get a dataframe of the texts for the analytical report.
    texts_df = pd.read_csv(test_path, sep='\t').text

    # ------------------------------------------------------------------- #
    #   Parser values
    # ------------------------------------------------------------------- #

    parser = argparse.ArgumentParser()

    parser.add_argument('--test-path',
                        type=str,
                        help='The input data dir. Should contain the .csv files for the task.',
                        default=test_path)

    parser.add_argument('--model-path',
                        type=str,
                        help='The input data dir. Should contain the .csv files for the task.',
                        default=model_path)

    parser.add_argument('--export-results-path',
                        default=output_path,
                        type=str,
                        help='Path to save the results')

    parser.add_argument('--classifier-name',
                        default=classifier_name,
                        type=str,
                        help='Name of the classifier.')

    parser.add_argument('--task_name',
                        default='SEMEVAL',
                        type=str,
                        help='The name of the task to train.')

    parser.add_argument('--bert_model',
                        default='bert-large-uncased',
                        type=str,
                        help='Bert pre-trained model selected in the list: bert-base-uncased, '
                             'bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, '
                             'bert-base-multilingual-cased, bert-base-chinese.')

    ## Other parameters
    parser.add_argument('--max_seq_length',
                        default=128,
                        type=int,
                        help='The maximum total input sequence length after WordPiece tokenization. \n'
                             'Sequences longer than this will be truncated, and sequences shorter \n'
                             'than this will be padded.')

    parser.add_argument('--do_train',
                        action='store_true',
                        default=True,
                        help='Whether to run training.')

    parser.add_argument('--do_eval',
                        action='store_true',
                        default=True,
                        help='Whether to run eval on the dev set.')

    parser.add_argument('--do_test',
                        action='store_true',
                        default=False,
                        help='Whether to run test on the dev set.')

    parser.add_argument('--do_lower_case',
                        action='store_true',
                        default=True,
                        help='Set this flag if you are using an uncased model.')

    parser.add_argument('--train_batch_size',
                        default=8,
                        type=int,
                        help='Total batch size for training.')

    parser.add_argument('--eval_batch_size',
                        default=8,
                        type=int,
                        help='Total batch size for eval.')

    parser.add_argument('--learning_rate',
                        default=2e-5,
                        type=float,
                        help='The initial learning rate for Adam.')

    parser.add_argument('--num_train_epochs',
                        default=3.0,
                        type=float,
                        help='Total number of training epochs to perform.')

    parser.add_argument('--warmup_proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10%% of training.')

    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='Whether not to use CUDA when available')

    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local_rank for distributed training on gpus')

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed for initialization')

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')

    parser.add_argument('--fp16',
                        action='store_true',
                        help='Whether to use 16-bit float precision instead of 32-bit')

    parser.add_argument('--loss_scale',
                        type=float,
                        default=0,
                        help='Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n'
                             '0 (default value): dynamic loss scaling.\n'
                             'Positive power of 2: static loss scaling value.\n')

    # ------------------------------------------------------------------- #
    #   Checking the validity of parameters etc.
    # ------------------------------------------------------------------- #

    args = parser.parse_args()

    processors = {
        'semeval': SemEvalProcessor,
    }
    num_labels_task = {
        'semeval': 2,
    } # Set the number of labels

    task_name = args.task_name.lower()
    # task_name = 'semeval'

    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    # logger.info('device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'.format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    # ------------------------------------------------------------------- #
    #   Testing the model
    # ------------------------------------------------------------------- #

    # Load the model
    model_state_dict = torch.load(model_path)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict, num_labels=num_labels)
    model.to(device)

    # eval_examples = processor.get_eval_examples(args.validation_set_path)
    eval_examples = processor.get_eval_examples(args.test_path)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer)

    # print('\n__________________________________')
    # logger.info('Running Evaluation: ')
    # logger.info('    Number of examples = %d', len(eval_examples))
    # logger.info('    Number of features = %d', len(eval_features))
    # logger.info('    Batch size = %d', args.eval_batch_size)
    # print('__________________________________\n')

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    predicted_all = []
    true_all = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc='Evaluating'):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        preds = np.argmax(logits, axis=1).tolist()
        true = label_ids.tolist()
        predicted_all.extend(preds)
        true_all.extend(true)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    # print('\n__________________________________')
    # print('\nResults:\n')
    # print('\nWeighted: ',
    #       '\nPrecision: ', precision_score(true_all, predicted_all, average='weighted'),
    #       '\nRecall: ', recall_score(true_all, predicted_all, average='weighted'),
    #       '\nF1: ', f1_score(true_all, predicted_all, average='weighted'),
    #       '\nAccuracy: ', accuracy_score(true_all, predicted_all),
    #
    #       '\nMacro: ',
    #       '\nPrecision: ', precision_score(true_all, predicted_all, average='macro'),
    #       '\nRecall: ', recall_score(true_all, predicted_all, average='macro'),
    #       '\nF1: ', f1_score(true_all, predicted_all, average='macro'),
    #       '\nAccuracy: ', accuracy_score(true_all, predicted_all))
    # print('__________________________________\n')
    #
    # with open(output_path, 'w') as f:
    #     f.write('\nResults:\n')
    #     f.write('\nWeighted:')
    #     f.write('\nPrecision: ' + str(precision_score(true_all, predicted_all, average='weighted')))
    #     f.write('\nRecall: ' + str(recall_score(true_all, predicted_all, average='weighted')))
    #     f.write('\nF1: ' + str(f1_score(true_all, predicted_all, average='weighted')))
    #     f.write('\nAccuracy: ' + str(accuracy_score(true_all, predicted_all)))
    #
    #     f.write('\n\nMacro:')
    #     f.write('\nPrecision: ' + str(precision_score(true_all, predicted_all, average='macro')))
    #     f.write('\nRecall: ' + str(recall_score(true_all, predicted_all, average='macro')))
    #     f.write('\nF1: ' + str(f1_score(true_all, predicted_all, average='macro')))
    #     f.write('\nAccuracy: ' + str(accuracy_score(true_all, predicted_all)))
    # print("Report saved to file: " + str(output_path))

    with open(output_path, 'w') as f:
        f.write('\nResults:\n')
        f.write('\nWeighted:')
        f.write('\nPrecision: ' + str(precision_score(true_all, predicted_all, average='weighted')))
        f.write('\nRecall: ' + str(recall_score(true_all, predicted_all, average='weighted')))
        f.write('\nF1: ' + str(f1_score(true_all, predicted_all, average='weighted')))
        f.write('\nAccuracy: ' + str(accuracy_score(true_all, predicted_all)))

        f.write('\n\nMacro:')
        f.write('\nPrecision: ' + str(precision_score(true_all, predicted_all, average='macro')))
        f.write('\nRecall: ' + str(recall_score(true_all, predicted_all, average='macro')))
        f.write('\nF1: ' + str(f1_score(true_all, predicted_all, average='macro')))
        f.write('\nAccuracy: ' + str(accuracy_score(true_all, predicted_all)))
    print('======> Report saved to file: ' + str(output_path))

    # Write the analytical results to .csv
    write_analytical_results(classifier_name, texts_df, predicted_all)


#===========================#
#           Main            #
#===========================#

if __name__ == '__main__':

    args = docopt(__doc__)

    # print('\nArguments: ')
    # print(args)

    if args['--test-path']:
        test(args)
    else:
        train(args)


#===========================#
#       End of Script       #
#===========================#
