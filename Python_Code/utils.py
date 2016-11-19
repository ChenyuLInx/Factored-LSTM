import numpy as np

'''
This code used some functions from Zhe Gan(zhegan27 on github)'s modified version of
https://github.com/yoonkim/CNN_sentence
'''

def dict2Mat(word_dict, k = 300):
    '''
    Convert word-embedding dictonary to array of embeddings
    '''
    vocab_size = len(word_dict)
    W = np.zeros(shape=(vocab_size, k))
    n = 0
    for key in word_dict:
        W[n] = word_dict[key]
        n = n + 1
    return W


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def load_words(loc='./words.txt'):
    """
    Load words into list
    """
    train = []
    with open(loc, 'rb') as f:
        for line in f:
            train.append(line.strip())
    return train

def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that do not occur in the pretrained word embedding, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pretrained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)