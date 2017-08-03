import sys, os
import pickle
import argparse
from logging import getLogger, StreamHandler, DEBUG

LANGS = ['albanian',
'arabic',
'armenian',
'basque',
'bengali',
'bulgarian',
'catalan',
'czech',
'danish',
'dutch',
'english',
'estonian',
'faroese',
'finnish',
'french',
'georgian',
'german',
'haida',
'hebrew',
'hindi',
'hungarian',
'icelandic',
'irish',
'italian',
'khaling',
'kurmanji',
'latin',
'latvian',
'lithuanian',
'lower-sorbian',
'macedonian',
'navajo',
'northern-sami',
'norwegian-bokmal',
'norwegian-nynorsk',
'persian',
'polish',
'portuguese',
'quechua',
'romanian',
'russian',
'scottish-gaelic',
'serbo-croatian',
'slovak',
'slovene',
'sorani',
'spanish',
'swedish',
'turkish',
'ukrainian',
'urdu',
'welsh']

if __name__ == '__main__':
    ####
    # Parse arguments
    ####
    parser = argparse.ArgumentParser(
        description='analyze and solve the SIGMORPHON 2017 Shared Task'
    )

    parser.add_argument(
        '--resource', help='resource type: high, mid, or low (default: high)',
         type=str, default='high'
    )
    
    parser.add_argument(
        '--task', help='task number (default: 1)',
         type=str, default='1'
    )
    
    parser.add_argument(
        '--debug-print', help='print debug info',
         type=bool, default=False
    )

    parser.add_argument(
        '--create-screen', help='create screen',
         type=bool, default=False
    )
    
    parser.add_argument(
        '--start-gpu', help='start-gpu (for create screen)',
         type=int, default=0
    )

    parser.add_argument(
        '--end-gpu', help='end-gpu, *exclusive* (for create screen)',
         type=int, default=16
    )
    
    args = parser.parse_args()

    logger = getLogger(__name__)
    handler = StreamHandler()

    if args.debug_print:
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)

    logger.addHandler(handler)
    
    
    validate_directory_path = './validate/task%s/' % (args.task)
    
    best_results = []
    
    total = 0
    total_accuracy = 0.0
    total_distance_loss = 0.0
    
    for lang in LANGS:
        validate_file_path = validate_directory_path + lang + '-' + args.resource + '.txt'
        
        if not os.path.isfile(validate_file_path):
            logger.debug('MISSING LANG: %s' % (lang))
            continue

        logger.debug(lang)
        f = open(validate_file_path, 'r')
        result_list = eval('[' + f.read() + ']')
        f.close()
        result_list = sorted(result_list, key=lambda item: item[0])
        result_list = sorted(result_list, key=lambda item: item[5])
        result_list = sorted(result_list, key=lambda item: item[4], reverse=True)
        
        best = result_list[0]
        
        model_file_name = lang + '-' + args.resource + '-' + str(best[0]) + '-model'
        if not args.create_screen:
            print('%s\t%f\t%f' % (model_file_name, best[4], best[5]))
        
        best_results.append((lang, model_file_name))
        
        total += 1
        total_accuracy += best[4]
        total_distance_loss += best[5]
        
    if not args.create_screen:
        logger.debug('# of languages: %d', (total))
        logger.debug('Avg. accuracy: %f', (total_accuracy / total))
        logger.debug('Avg. Levenshtein distance: %f', (total_distance_loss / total))
    
    if args.create_screen:
        current_gpu = args.start_gpu
        for (lang, best_model) in best_results:
        
            template = 'screen -dmS cuda%d.predict.%s.%s bash -c "THEANO_FLAGS=\'device=cuda%d\' python3 main.py predict %s --resource %s --debug-print True --model-path model/task%s/%s"'
        
            print(template % (current_gpu, lang, args.resource, current_gpu, lang, args.resource, args.task, best_model))
            
            current_gpu += 1
            
            if current_gpu == args.end_gpu:
                current_gpu = args.start_gpu