import os
import pickle

import numpy as np
import util

def distance(str1, str2):
    """Simple Levenshtein implementation for evalm.
    
    Copied from evalm.py in CoNLL 2017

    Author: Ryan Cotterell
    """
    m = np.zeros([len(str2)+1, len(str1)+1])
    for x in range(1, len(str2) + 1):
        m[x][0] = m[x-1][0] + 1
    for y in range(1, len(str1) + 1):
        m[0][y] = m[0][y-1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
    return int(m[len(str2)][len(str1)])

def validate(args, data, model_directory_path, out_directory_path,
    validate_directory_path,
    *, logger = None):

    alphabet_file_path = model_directory_path + args.lang + '-' + args.resource + '-alphabet'
    alph = pickle.load(open(alphabet_file_path, 'rb'))    
    

    """
    Create data
    """
    dev_data = []

    for d in data:
        v = alph.create_index_vector(d)
        dev_data.append(v)
    
    dim = alph.get_dimension()
    
    """
    Loading
    """
    # if args.model_path == None:
    #     raise ValueError('model path unspecified')
    
    max_len = 150
    round = 0
    
    validate_file_path = validate_directory_path + args.lang + '-' + args.resource + '.txt'
    validate_file = open(validate_file_path, 'r+')
    
    content = eval('[' + validate_file.read() + ']')
    
    if len(content) > 0:
        round = content[-1][0]
    
    for iteration_num in range(max_len):
        round += args.model_save_interval
    
        model_file_path = model_directory_path + args.lang + '-' + args.resource + '-' + str(round) + '-model'
    
        if not os.path.isfile(model_file_path):
            break
    
        p = pickle.load(open(model_file_path, 'rb'))
        # logger.debug('Predictor loaded')
        # logger.debug(p)
        predictor_instance = p['predictor']

        """
        Main
        """
        correct = 0
        distance_loss = 0

        for i in range(len(dev_data)):
            correct_str = data[i][1]
            
            in_v = util.convert_to_one_hot(dev_data[i][0], dim)

            predicted = predictor_instance.predict(in_v)
            t = util.convert_to_index(predicted)
            result_str = alph.serialize_output(t)
        
        
            if result_str == correct_str:
                correct += 1
            distance_loss += distance(result_str, correct_str)
        
        accuracy = correct / len(dev_data)
        averaged_distance_loss = distance_loss / len(dev_data)
        
        result = str((round, p['clock_time_history'][iteration_num], p['interval_loss_history'][iteration_num], p['total_loss'], accuracy, averaged_distance_loss)) + ','
        
        validate_file.write(result)
        validate_file.write('\n')

        logger.debug(result)

    validate_file.close()