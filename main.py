import sys, os
import pickle
import argparse
from logging import getLogger, StreamHandler, DEBUG
import time

import numpy as np
import theano

import train, predict, validate

import predictor
import alphabet
import util

theano.config.exception_verbosity='high'

RECURSION_LIMIT = 50000

def parse_lines_task_1(lines):
    """ Parse lines for the task 1
    
    Args:
        lines (list of str)
    
    Returns:
        tuple: a 3-tuple as (source (str), target (str), features (list of str))
        Note that this function does not perform any ordering on features
    """
    
    result = []
    
    for line in lines:
        t = line.split('\t')
        features = t[2][0:-1].split(';') # [0:-1] to remove \n at the last pos
        result.append((t[0], t[1], features))
        
    return result

def parse_lines_task_2(lines):
    """ Parse lines for the task 1
    
    Args:
        lines (list of str)
    
    Returns:
        tuple: a 3-tuple as (source (str), target (str), features (list of str))
        Note that this function does not perform any ordering on features
    """

    result = []
    
    for line in lines:
        t = line.split('\t')
        features = t[2][0:-1].split(';') # [0:-1] to remove \n at the last pos
        result.append((t[0], t[1], features))
        
    return result

def model_info(args):
    model = pickle.load(open(args.model_path, 'rb'))    
    model['predictor'] = None
    print(model)

if __name__ == '__main__':
    ####
    # Parse arguments
    ####
    parser = argparse.ArgumentParser(
        description='analyze and solve the SIGMORPHON 2017 Shared Task'
    )

    parser.add_argument(
        'command', help='command name (train, predict, validate, or model_info)', type=str
    )
    
    parser.add_argument(
        'lang', help='lang-name to be processed', type=str
    )
    
    parser.add_argument(
        '--resource', help='resource type: high, mid, or low (default: high)',
         type=str, default='high'
    )

    parser.add_argument(
        '--embedding-dim', help='dimension for embeddings (default: 300)',
         type=int, default=300
    )
    
    parser.add_argument(
        '--hidden-dim', help='dimension for hidden layers (default: 100)',
         type=int, default=100
    )

    parser.add_argument(
        '--context-dim', help='dimension for context vectors (default: 100)',
         type=int, default=100
    )
        
    parser.add_argument(
        '--max-iter', help='maximum for training iteration (default: 20)',
         type=int, default=20
    )
    
    parser.add_argument(
        '--task', help='task number (default: 1)',
         type=str, default='1'
    )
    
    parser.add_argument(
        '--model-path', help='path to model; required for prediction; resume training for train'
    )
    
    parser.add_argument(
        '--optimizer', help='optimizer; adam, adamax, adam_l1, adamax_l1',
         type=str, default='adamax'
    )
    
    parser.add_argument(
        '--activation', help='activation function; tanh, relu',
         type=str, default='tanh'
    )

    parser.add_argument(
        '--model-save-interval', help='model save interval',
         type=int, default=5000
    )
    
    parser.add_argument(
        '--debug-print', help='print debug info',
         type=bool, default=False
    )
    
    parser.add_argument(
        '--base-dir', help='path to "all" directory in the CoNLL dataset',
        type=str, default= '../all'
    )

    sys.setrecursionlimit(RECURSION_LIMIT) # IMPORTANT!
    
    args = parser.parse_args()

    logger = getLogger(__name__)
    handler = StreamHandler()

    if args.debug_print:
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)

    logger.addHandler(handler)

    dev_suffix = ''
    out_directory_path = ''
    model_directory_path = ''
    validate_directory_path = ''
    
    if args.task == '1':
        dev_suffix = '-dev'
        test_suffix = '-covered-test'
        out_directory_path = './out/task1/'
        model_directory_path = './model/task1/'
        validate_directory_path = './validate/task1/'
    elif args.task == '2':
        dev_suffix = '-covered-dev'
        out_directory_path = './out/task2/'
        model_directory_path = './model/task2/'
        validate_directory_path = './validate/task2/'
    else:
        raise ValueError('task number must be 1 or 2') 

    if not os.path.exists(out_directory_path):
        os.makedirs(out_directory_path)

    if not os.path.exists(model_directory_path):
        os.makedirs(model_directory_path)

    if not os.path.exists(validate_directory_path):
        os.makedirs(validate_directory_path)

    train_file_name = args.base_dir + '/task' + args.task + '/' + args.lang + '-train-' + args.resource
    dev_file_name = args.base_dir + '/task'  + args.task + '/' + args.lang + dev_suffix  
    test_file_name = args.base_dir + '/task'  + args.task + '/' + args.lang + test_suffix

    # ../all/task1/albanian-train-high
    logger.debug('Start')
    
    if args.command == 'train':
        f = open(train_file_name)
        lines = f.readlines()

        if args.task == '1':
            data = parse_lines_task_1(lines)

        else:
            data = parse_lines_task_2(lines)

        train.train(args, data, model_directory_path, logger = logger)

    elif args.command == 'predict':        
        f_test = open(test_file_name)
        lines = f_test.readlines()

        if args.task == '1':
            test_data = parse_lines_task_1(lines)
        else:
            test_data = parse_lines_task_2(lines)

        predict.predict(args, test_data, model_directory_path,
            out_directory_path, logger = logger)
    elif args.command == 'model_info':
        model_info(args)
    elif args.command == 'validate':
        f_dev = open(dev_file_name)
        lines = f_dev.readlines()

        if args.task == '1':
            dev_data = parse_lines_task_1(lines)
        else:
            dev_data = parse_lines_task_2(lines)

        validate.validate(args, dev_data, model_directory_path,
            out_directory_path, validate_directory_path,
            logger = logger)
    else:
        raise ValueError('task number must be train or test')