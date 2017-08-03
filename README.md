# CoNLL-SIGMPRHON Shared Task 2017 system

The code is so messy that other developers can not use this library.

But the basic usage to replicate the result is as followings (CUDA GPU required):

Train (`[dir_to_all]` represents the path to the directory "`all`" in the official shared task dataset):

```
THEANO_FLAGS=\'device=cuda0\' python3 main.py train french --resource high --debug-print True --hidden-dim 200 --embedding-dim 300 --context-dim 200 --optimizer=adamax --max-iter 100 --model-save-interval 10000 --base-dir [dir_to_all]
```

This will create a file like ``model/task1/french-high-30000-model''.

Prediction:

```
THEANO_FLAGS=\'device=cuda0\' python3 main.py predict french --resource high --debug-print True --model-path model/task1/french-high-30000-model --base-dir [dir_to_all]
```

This will create an output file like ``out/task1/french-high-out''

The directory ``results'' contains our final results submitted to the shared task.

License: new BSD.