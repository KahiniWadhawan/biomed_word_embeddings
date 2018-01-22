import gensim
import logging

import scipy
from scipy import stats
import os
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

MODEL_DIR = "word2vec/data/"
model_bin_file_name = 'full_pubmed_lowercase-vector-ng10-hs0-dim200-textwin30-subsample4-mincount5-alpha0.05.bin'
model_bin_file = os.path.join(MODEL_DIR, model_bin_file_name)
model = gensim.models.KeyedVectors.load_word2vec_format(model_bin_file, binary=True)

PAIR_FILES_DIR          = "results/subset_original_files_with_mean"

OUPUT_COSINE_SIM_FILES_DIR = "results/scores/word2vec"

def perform_tests_mean(input_file, output_cosine_sim_file):
    res_tests = []

    total = 0
    unseen = 0
    human_means = {}
    wordvec_cosine_sim = {}
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
            try:
                cosine_sim = model.similarity(term1, term2)
                output_cosine_sim_file_writer.writerow([cosine_sim, term1, term2])
                wordvec_cosine_sim[(term1, term2)] = cosine_sim

            except KeyError:
                # print "gensim similarity function got key error"
                unseen += 1

            total += 1

        output_cosine_sim_file_handle.close()
        print("total, seen, unseen :: ", total, total - unseen, unseen)
        human_means_lst = []
        wordvec_cosine_sim_lst = []
        pairs_found = []
        for key in human_means:
            if key in wordvec_cosine_sim.keys():
                pairs_found.append(key)
                wordvec_cosine_sim_lst.append(float(wordvec_cosine_sim[key]))
                human_means_lst.append(float(human_means[key]))

        # computing spearman's and pearson's coefficient
        spearman_coeff = stats.spearmanr(human_means_lst, wordvec_cosine_sim_lst)
        res_tests.append(spearman_coeff[0])
        pearson_coeff = stats.pearsonr(human_means_lst, wordvec_cosine_sim_lst)
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
    wordvec_cosine_sim = {}
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
            try:
                cosine_sim = model.similarity(term1, term2)
                # If any cosine_sim is zero, then entropy becomes zero, due to the divide by zero operation.
                # Hence, replacing any zero value with a very small value 0.00001
                if cosine_sim == 0.0 or cosine_sim < 0 :
                    cosine_sim = 0.00001
                output_cosine_sim_file_writer.writerow([cosine_sim, term1, term2])
                wordvec_cosine_sim[(term1, term2)] = cosine_sim

            except KeyError:
                # print "gensim similarity function got key error"
                unseen += 1

            total += 1

        output_cosine_sim_file_handle.close()
        print("total, seen, unseen :: ", total, total - unseen, unseen)
        human_means_lst = []
        wordvec_cosine_sim_lst = []
        pairs_found = []
        for key in human_means:
            if key in wordvec_cosine_sim.keys():
                pairs_found.append(key)
                wordvec_cosine_sim_lst.append(float(wordvec_cosine_sim[key]))
                human_means_lst.append(float(human_means[key]))

        # compute scipy.stats.entropy
        entropy = scipy.stats.entropy(human_means_lst, wordvec_cosine_sim_lst)
        print("entropy :: ", entropy)
        print("for file :: ", filename)
        print("---------------------------------------------------------------")

    return entropy

# --------------------------------------------------------------------------------------------
# write results to csv
# --------------------------------------------------------------------------------------------
combined_results_file = 'eval_word2vec_subset.txt'
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

