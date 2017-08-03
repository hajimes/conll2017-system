import pickle

import util

def predict(args, data, model_directory_path, out_directory_path,
    *, logger = None):


    """
    Loading
    """
    if args.model_path == None:
        raise ValueError('model path unspecified')
    
    p = pickle.load(open(args.model_path, 'rb'))
    logger.debug('Predictor loaded')
    logger.debug(p)
    predictor_instance = p['predictor']

    alphabet_file_path = model_directory_path + args.lang + '-' + args.resource + '-alphabet'
    alph = pickle.load(open(alphabet_file_path, 'rb'))


    """
    Create data
    """
    test_data = []

    for d in data:
        v = alph.create_index_vector(d)
        test_data.append(v)
    
    dim = alph.get_dimension()
    
    
    """
    Main
    """
    out_path = out_directory_path + args.lang + '-' + args.resource + '-out'
    out_file = open(out_path, 'w')

    for i in range(len(test_data)):
        in_v = util.convert_to_one_hot(test_data[i][0], dim)

        predicted = predictor_instance.predict(in_v)
        t = util.convert_to_index(predicted)
        result_str = alph.serialize_output(t)
        
        out_file.write(data[i][0])
        out_file.write('\t')
        out_file.write(result_str)
        out_file.write('\t')
        out_file.write(';'.join(data[i][2]))
        out_file.write('\n')

    logger.debug('Prediction: done')