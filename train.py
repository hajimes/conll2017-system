import pickle
import time

import alphabet
import predictor
import util

def train(args, data, model_directory_path, *, logger = None):
    alphabet_file_path = model_directory_path + args.lang + '-' + args.resource + '-alphabet'
    
    if args.model_path == None:
        alph = alphabet.make_alphabet(alphabet_file_path, data, logger = logger)
        dim = alph.get_dimension()
        predictor_instance = predictor.Predictor(dim, dim,
            args.embedding_dim, args.hidden_dim, args.context_dim,
            optimization_method=args.optimizer,
            activation_method=args.activation)

        model = {
            'predictor': predictor_instance,
            'lang': args.lang,
            'resource': args.resource,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'context_dim': args.context_dim,
            'optimizer': args.optimizer,
            'activation': args.activation,
            'interval': args.model_save_interval,
            'round': 0,
            'interval_loss_history': [],
            'clock_time_history': [],
            'total_loss': 0.0
        }
    else:
        model = pickle.load(open(args.model_path, 'rb'))    
        alphabet_file_path = model_directory_path + model['lang'] + '-' + model['resource'] + '-alphabet'
        alph = pickle.load(open(alphabet_file_path, 'rb'))
        dim = alph.get_dimension()

    ###
    # Train phase
    ###
    logger.debug('Start: format train_data')
    
    train_data = []
    for d in data:
        v = alph.create_index_vector(d)
        train_data.append(v)

    logger.debug('End: format train_data')

    ###
    # Fit
    ###    

    clock_time_history = []
    
    current_loss = 0.0
    averaged_loss = 0.0
    interval_loss = 0.0
    
    logger.info('Training started')
    start = time.time()
    for epoch in range(args.max_iter):
        epoch_loss = 0.0
        
        for i in range(len(train_data)):
            in_v = util.convert_to_one_hot(train_data[i][0], dim)
            out_v = util.convert_to_one_hot(train_data[i][1], dim)

            current_loss = model['predictor'].train(in_v, out_v)
            model['total_loss'] += current_loss
            epoch_loss += current_loss
            interval_loss += current_loss

            model['round'] += 1
            
            if (model['round'] % model['interval']) == 0:
                result = model['predictor'].predict(in_v)
                formatted_out_seq = alph.format_output(result)

                end = time.time()

                elapsed = end-start

                logger.debug('Period: %d' % (model['round']))
                logger.debug('Elapsed (s): %d' % elapsed)
                logger.debug(train_data[i][0])
                logger.debug(train_data[i][1])
                logger.debug(alph.format_output(out_v))
                logger.debug(formatted_out_seq)
                logger.debug(alph.serialize_output(formatted_out_seq))

                logger.info('Round %d finished. Loss incurred during this interval: %e' % (model['round'], interval_loss))
                model['clock_time_history'].append(elapsed)
                model['interval_loss_history'].append(interval_loss)

                model_file_path = model_directory_path + args.lang + '-' + args.resource + '-' + str(model['round']) + '-model'        

                pickle.dump(model, open(model_file_path, 'wb'))

                interval_loss = 0.0
                start = time.time()
                
        logger.info('Epoch %d finished. Loss incurred during this epoch: %e' % (epoch, epoch_loss))