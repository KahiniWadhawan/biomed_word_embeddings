import argparse
import numpy as np
import sys

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def distance(W, vocab, ivocab, input_term1, input_term2):
    for idx, term in enumerate(input_term1.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result1 = np.copy(W[vocab[term], :])
            else:
                vec_result1 += W[vocab[term], :] 
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return
    
    vec_norm1 = np.zeros(vec_result1.shape)
    d1 = (np.sum(vec_result1 ** 2,) ** (0.5))
    vec_norm1 = (vec_result1.T / d1).T

    for idx, term in enumerate(input_term2.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result2 = np.copy(W[vocab[term], :])
            else:
                vec_result2 += W[vocab[term], :]
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return
    vec_norm2 = np.zeros(vec_result2.shape)
    d2 = (np.sum(vec_result2 ** 2,) ** (0.5))
    vec_norm2 = (vec_result2.T / d2).T
    dist = np.dot(vec_norm1.T, vec_norm2.T)
    print('cosine dist :: ', input_term1, input_term2, dist)
    return dist

def similarity(input_term1, input_term2):
    W, vocab, ivocab = generate()
    dist = distance(W, vocab, ivocab, input_term1, input_term2)

