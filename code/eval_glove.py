import gensim
import logging

import os, sys
import csv
import scipy
from scipy import stats
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PAIR_FILES_DIR          = "results/subset_original_files_with_mean"
OUPUT_COSINE_SIM_FILES_DIR = "results/scores/glove"

# -------------------------------------------------------------------
# ---------------------------HELPER FUNCTIONS------------------------
# -------------------------------------------------------------------

def generate(vocab_file, vectors_file):
    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(vectors_file, 'r') as f:
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
            # print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result1 = np.copy(W[vocab[term], :])
            else:
                vec_result1 += W[vocab[term], :]
        else:
            # print('Word: %s  Out of dictionary!\n' % term)
            return sys.maxint

    vec_norm1 = np.zeros(vec_result1.shape)
    d1 = (np.sum(vec_result1 ** 2, ) ** (0.5))
    vec_norm1 = (vec_result1.T / d1).T

    for idx, term in enumerate(input_term2.split(' ')):
        if term in vocab:
            # print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result2 = np.copy(W[vocab[term], :])
            else:
                vec_result2 += W[vocab[term], :]
        else:
            # print('Word: %s  Out of dictionary!\n' % term)
            return sys.maxint
    vec_norm2 = np.zeros(vec_result2.shape)
    d2 = (np.sum(vec_result2 ** 2, ) ** (0.5))
    vec_norm2 = (vec_result2.T / d2).T
    dist = np.dot(vec_norm1.T, vec_norm2.T)
    # print('cosine dist :: ', input_term1, input_term2, dist)
    return dist

def similarity(input_term1, input_term2):
    W, vocab, ivocab = generate()
    dist = distance(W, vocab, ivocab, input_term1, input_term2)

# -------------------------------------------------------------------
# -------------------------------------------------------------------

# This is the model parameters from Glove used to compute cosine similarity
glove_dir = 'GloVe/'
glove_vocab_file = 'vocab.txt'
glove_vocab_file = os.path.join(glove_dir, glove_vocab_file)
glove_vectors_file = 'vectors.txt'
glove_vectors_file = os.path.join(glove_dir, glove_vectors_file)
W, VOCAB, IVOCAB = generate(vocab_file=glove_vocab_file, vectors_file=glove_vectors_file)

def perform_tests_mean(input_file, output_cosine_sim_file):
    res_tests = []

    total = 0
    unseen = 0
    human_means = {}
    glove_cosine_sim = {}
    filename = os.path.join(PAIR_FILES_DIR, input_file)
    output_cosine_sim_file = os.path.join(OUPUT_COSINE_SIM_FILES_DIR, output_cosine_sim_file)
    output_cosine_sim_file_handle = open(output_cosine_sim_file, 'wb')
    output_cosine_sim_file_writer = csv.writer(output_cosine_sim_file_handle, delimiter=' ',quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    output_cosine_sim_file_writer.writerow(["Cosine Similarity", "Term 1", "Term 2"])

    with open(filename, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile.read().splitlines())
        # Skip the first row of header of the CSV file
        csv_reader.next()
        for row in csv_reader:
            mean_score = float(row[0])
            term1 = row[1]
            term2 = row[2]
            human_means[(term1, term2)] = mean_score
            g_cosine_sim = distance(W, VOCAB, IVOCAB, term1, term2)
            if g_cosine_sim == sys.maxint:
                unseen += 1
            else:
                output_cosine_sim_file_writer.writerow([g_cosine_sim, term1, term2])
                glove_cosine_sim[(term1, term2)] = g_cosine_sim

            total += 1

        output_cosine_sim_file_handle.close()
        print("total, seen, unseen :: ", total, total - unseen, unseen)
        human_means_lst = []
        glove_cosine_sim_lst = []
        pairs_found = []
        for key in human_means:
            if key in glove_cosine_sim.keys():
                pairs_found.append(key)
                glove_cosine_sim_lst.append(float(glove_cosine_sim[key]))
                human_means_lst.append(float(human_means[key]))

        # computing spearman's and pearson's coefficient
        spearman_coeff = stats.spearmanr(human_means_lst, glove_cosine_sim_lst)
        res_tests.append(spearman_coeff[0])
        pearson_coeff = stats.pearsonr(human_means_lst, glove_cosine_sim_lst)
        res_tests.append(pearson_coeff[0])
        print("spearman_coeff :: ", spearman_coeff)
        print("pearson coeff ::", pearson_coeff)
        print("for file :: ", filename)
        print("---------------------------------------------------------------")

    return res_tests

def perform_tests_wo_mean(input_file, output_cosine_sim_file):
    total = 0
    unseen = 0
    human_means = {}
    glove_cosine_sim = {}
    filename = os.path.join(PAIR_FILES_DIR, input_file)

    output_cosine_sim_file = os.path.join(OUPUT_COSINE_SIM_FILES_DIR, output_cosine_sim_file)
    output_cosine_sim_file_handle = open(output_cosine_sim_file, 'wb')
    output_cosine_sim_file_writer = csv.writer(output_cosine_sim_file_handle, delimiter=' ', quotechar=' ',
                                               quoting=csv.QUOTE_MINIMAL)
    output_cosine_sim_file_writer.writerow(["Cosine Similarity", "Term 1", "Term 2"])

    with open(filename, 'rb') as csvfile:
        csv_reader = csv.reader(csvfile.read().splitlines())
        # Skip the first row of header of the CSV file
        csv_reader.next()
        for row in csv_reader:
            mean_score = float(row[0])
            term1 = row[1]
            term2 = row[2]
            human_means[(term1, term2)] = mean_score
            g_cosine_sim = distance(W, VOCAB, IVOCAB, term1, term2)
            if g_cosine_sim == sys.maxint:
                unseen += 1
            else:
                # If any cosine_sim is zero, then entropy becomes zero, due to the divide by zero operation.
                # Hence, replacing any zero value with a very small value 0.00001
                if g_cosine_sim == 0.0 or g_cosine_sim < 0 :
                    g_cosine_sim = 0.00001
                output_cosine_sim_file_writer.writerow([g_cosine_sim, term1, term2])
                glove_cosine_sim[(term1, term2)] = g_cosine_sim

            total += 1

        output_cosine_sim_file_handle.close()
        print("total, seen, unseen :: ", total, total - unseen, unseen)
        human_means_lst = []
        glove_cosine_sim_lst = []
        pairs_found = []
        for key in human_means:
            if key in glove_cosine_sim.keys():
                pairs_found.append(key)
                glove_cosine_sim_lst.append(float(glove_cosine_sim[key]))
                human_means_lst.append(float(human_means[key]))

        # compute scipy.stats.entropy
        entropy = scipy.stats.entropy(human_means_lst, glove_cosine_sim_lst)
        print("entropy :: ", entropy)
        print("for file :: ", filename)
        print("---------------------------------------------------------------")

    return entropy

# --------------------------------------------------------------------------------------------
# write results to csv
# --------------------------------------------------------------------------------------------
combined_results_file = 'eval_glove_subset.txt'
combined_results_file = os.path.join(OUPUT_COSINE_SIM_FILES_DIR, combined_results_file)
with open(combined_results_file, 'w') as outfile:
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    res_mean_similar = perform_tests_mean('coverdSimilarPairs.csv', 'CosineScoresSimilar.csv')
    res_mean_related = perform_tests_mean('coverdRelatedPairs.csv', 'CosineScoresRelated.csv')
    res_wo_mean_analogy = perform_tests_wo_mean('coverdAnalogyPairs.csv', 'CosineScoresAnalogy.csv')
    res_wo_mean_bmass = perform_tests_wo_mean('coverdBmassPairs.csv', 'CosineScoresBmass.csv')

    print("res_mean_similar :: ", res_mean_similar)
    print("res_mean_related :: ", res_mean_related)
    print("res_wo_mean_analogy :: ", res_wo_mean_analogy)
    print("res_wo_mean_bmass :: ", res_wo_mean_bmass)

    outfile.write('spearman_coveredSimilarPairs - ' + str(res_mean_similar[0]) + '\n')
    outfile.write('pearson_coveredSimilarPairs  - ' + str(res_mean_similar[1]) + '\n')
    outfile.write('spearman_coveredRelatedPairs - ' + str(res_mean_related[0]) + '\n')
    outfile.write('pearson_coveredRelatedPairs  - ' + str(res_mean_related[1]) + '\n')
    outfile.write('entropy_coveredAnalogyPairs  - ' + str(res_wo_mean_analogy) + '\n')
    outfile.write('entropy_coveredBmassPairs    - ' + str(res_wo_mean_bmass) + '\n')

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

