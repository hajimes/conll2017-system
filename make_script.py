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

# num_gpus = 7
num_gpus = 15

start_lang = 'georgian'

current = 0

cmd = 'train'

resource = 'low'

template = '"THEANO_FLAGS=\'device=cuda%d\' python3 main.py %s %s --resource %s --debug-print True --hidden-dim 200 --embedding-dim 300 --context-dim 200 --optimizer=adamax --max-iter 100 --model-save-interval 1000"'

THEANO_FLAGS=\'device=cuda0\' python3 main.py foo barabra --resource high --debug-print True --hidden-dim 200 --embedding-dim 300 --context-dim 200 --optimizer=adamax --max-iter 100 --model-save-interval 1000

# template = '"THEANO_FLAGS=\'device=cuda%d\' python3 main.py train %s --resource medium --debug-print True --hidden-dim 200 --embedding-dim 300 --context-dim 200 --optimizer=adamax --max-iter 150 --model-save-interval 10000"'

# template = '"THEANO_FLAGS=\'device=cuda%d\' python3 main.py --task 2 train %s --resource high --debug-print True --hidden-dim 200 --embedding-dim 300 --context-dim 200 --optimizer=adamax --max-iter 40 --model-save-interval 10000"'
#
# template = '"THEANO_FLAGS=\'device=cuda%d\' python3 main.py --task 1 train %s --resource high --debug-print True --hidden-dim 200 --embedding-dim 300 --context-dim 200 --optimizer=adamax --max-iter 2500 --model-save-interval 10000"'

# validate medium
# template = '"THEANO_FLAGS=\'device=cuda%d\' python3 main.py %s %s --resource %s --debug-print True --model-save-interval 10000"'


# # predict
# template = '"THEANO_FLAGS=\'device=cuda%d\' python3 main.py %s %s --resource %s --debug-print True --model-path model/task1/english-low-10000-model"'


for i in range(len(LANGS)):
    if current == 0 and LANGS[i] != start_lang:
        continue
    
    if current == num_gpus:
        break
    
    gpu_id = current
    
    print(('screen -dmS cuda%d.%s.%s.%s bash -c ' + template) % (gpu_id, cmd, LANGS[i], resource, gpu_id, cmd, LANGS[i], resource))
    
    current += 1