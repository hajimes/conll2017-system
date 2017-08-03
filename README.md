# CoNLL-SIGMPRHON Shared Task 2017 system

A system submitted (team name: UTNII) to [the CoNLL-SIGMORPHON-2017 Shared Task: Universal Morphological Reinflection](https://sites.google.com/view/conll-sigmorphon2017/home).

Hajime Senuma and Akiko Aizawa. 2017. Seq2seq for Morphological Reinflection: When Deep Learning Fails. In Proceedings of the CoNLL SIGMORPHON 2017 Shared Task: Universal Morphological Reinflection, pages 100–109, Stroudsburg, PA, USA. Association for Computational Linguistics. <https://doi.org/10.18653/v1/K17-2011>

The code is so messy that other developers can not use this library. But the basic usage to replicate the result is as followings (CUDA GPU required):

Trainining (`[dir_to_all]` represents the path to the directory "`all`" in the official shared task dataset):

```bash
THEANO_FLAGS=\'device=cuda0\' python3 main.py train french --resource high \
--debug-print True --hidden-dim 200 --embedding-dim 300 --context-dim 200 \
--optimizer=adamax --max-iter 100 --model-save-interval 10000 \
--base-dir [dir_to_all]
```

This will create a file like `model/task1/french-high-30000-model`.

Prediction:

```bash
THEANO_FLAGS=\'device=cuda0\' python3 main.py predict french --resource high \
--debug-print True --model-path model/task1/french-high-30000-model \
--base-dir [dir_to_all]
```

This will create an output file like `out/task1/french-high-out'.

The directory `results` contains our final results submitted to the shared task.

License: new BSD.