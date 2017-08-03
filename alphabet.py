import pickle

import numpy as np

class Alphabet(object):
    """ An object to maintain the relation between
        symbold id, features, and characters.
    """
    
    def __init__(self):        
        self._vocabulary = dict()
        self._inv_vocabulary = []
        self._features = dict()
        self._inv_features = []
        
    def analyze_vacabulary_task_1(self, data):
        """ Analyze the usage of characters in a dataset (for task 1)
    
        Args:
            data
    
        Returns:
            dict: a vacabulary
        """

        self._vocabulary = dict()
        self._inv_vocabulary = []
        count = 0
    
        for datum in data:
            for c in datum[0]:
                if not ord(c) in self._vocabulary:
                    self._vocabulary[ord(c)] = count
                    self._inv_vocabulary.append(ord(c))
                    count += 1
            for c in datum[1]:
                if not ord(c) in self._vocabulary:
                    self._vocabulary[ord(c)] = count
                    self._inv_vocabulary.append(ord(c))
                    count += 1
    
        return (self._vocabulary, self._inv_vocabulary)

    def analyze_features_task_1(self, data):
        self._features = dict()
        self._inv_features = []

        count = 0
    
        for datum in data:
            for f in datum[2]:
                if not f in self._features:
                    self._features[f] = count
                    self._inv_features.append(f)
                    count += 1

        return (self._features, self._inv_features)
        
    def format_output(self, out_seq):
        result = []

        for i in range(len(out_seq)):
            index = np.argmax(out_seq[i])
            result.append(index)

        return result

    def serialize_output(self, formatted_out_seq):
        result = ''

        v_offset = 3 + len(self._features)

        for i in range(len(formatted_out_seq)):        
            v = formatted_out_seq[i]
        
            v -= v_offset
            if v < 0:
                result += ''
            elif v == len(self._inv_vocabulary):
                result += ''
            else:
                c = chr(self._inv_vocabulary[v])
                result += c

        return result

    def get_dimension(self):
        """ Calculate a total dimension from a vocabulary and a feature set
        """
        return 4 + len(self._vocabulary) + len(self._features)
        

    def create_index_vector(self, datum):
        """ Parse a datum and output an index vector for NNs
    
        Args:
            datum
            vocabulary: dict created by analyze_vacabulary_task_1
            features: dict created by analyze_features_task_1
        Returns:
            tuple: 2-tuple of an input vector and an output vector
        """
    
        in_vec = []
        out_vec = []
    
        START_SYMBOL = 0
        END_SYMBOL = 1
    
        f_offset = 2
        v_offset = 2 + len(self._features) + 1 # 1 for UNK feature

        UNK_FEATURE = f_offset + len(self._features)
        UNK_CHARACTER = v_offset + len(self._vocabulary)
    
        in_vec.append(START_SYMBOL)
    
        for f in datum[2]:
            if f in self._features:
                index = self._features[f] + f_offset
                in_vec.append(index)
            else:
                in_vec.append(UNK_FEATURE)

        for c in datum[0]:
            if ord(c) in self._vocabulary:
                index = self._vocabulary[ord(c)] + v_offset
                in_vec.append(index)
            else:
                in_vec.append(UNK_CHARACTER)
    
        in_vec.append(END_SYMBOL)
    
        out_vec.append(START_SYMBOL)
        for c in datum[1]:
            if ord(c) in self._vocabulary:
                index = self._vocabulary[ord(c)] + v_offset
                out_vec.append(index)
            else:
                out_vec.append(UNK_CHARACTER)
        out_vec.append(END_SYMBOL)
        
        return (in_vec, out_vec)
        
        
def make_alphabet(alphabet_file_path, data, *, logger = None):
    alph = Alphabet()
    vocabulary, inv_vocabulary = alph.analyze_vacabulary_task_1(data)
    logger.info('# of characters: %d' % (len(vocabulary)))

    features, inv_features = alph.analyze_features_task_1(data)
    logger.info('# of features: %d' % (len(features)))

    v_offset = 3 + len(features)
    
    # START, END, UNK_FEATURE, UNK_CHARACTER
    logger.info('# of symbols: %d' % alph.get_dimension())
    
    dim = alph.get_dimension()
    
    pickle.dump(alph, open(alphabet_file_path, 'wb'))

    return alph