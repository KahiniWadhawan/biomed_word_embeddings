import gensim, logging
#from gensim.models import word2vec
import scipy
from scipy import stats
import os
import csv
import argparse

from glove_eval import distance

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#file_DIR = "exps/"
file_DIR = "./"
f1 = os.path.join(file_DIR,'full_pubmed_lowercase-vector-ng10-hs0-dim200-textwin30-subsample4-mincount5-alpha0.05.bin')
f2 = os.path.join(file_DIR,'full-pubmed-text_lower-vectors-phrase-ng10-hs0-dim200-textwin30-subsample4-mincount5-alpha0.05.bin')
#file_list = [f1,f2]
file_list = [f2]


def perform_glove_tests(args):
    # list to return results
    res_tests = []
    W, vocab, ivocab = generate()
    # ------------------------------------------------------------------------
    # Word similarity test
    # print(model.similarity('uterus', 'ovary'))
    # ------------------------------------------------------------------------------
    filename_lst = ['UMNSRS_similarity_mod449_word2vec.csv', 'UMNSRS_relatedness_mod458_word2vec.csv',
                    'MayoSRS.csv', 'MiniMayoSRS.csv']

    for filename in filename_lst:
        row_count = 0
        total = 0
        correct = 0
        unseen = 0
        human_means = {}
        glove_cosine_sim = {}

        with open(filename, 'rb') as csvfile:
            # csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            csv_reader = csv.reader(csvfile.read().splitlines())
            for row in csv_reader:
                if row_count != 0:
                    if filename.startswith('UMNSRS'):
                        mean_score = float(row[0])
                        term1 = row[2]
                        term2 = row[3]
                    elif filename == 'MayoSRS.csv':
                        mean_score = float(row[0])
                        term1 = row[3]
                        term2 = row[4]
                    elif filename == 'MiniMayoSRS.csv':
                        mean_score = float(row[0])   #physicians
                        term1 = row[4]
                        term2 = row[5]

                    human_means[(term1, term2)] = mean_score

                    try:
                        g_cosine_sim = distance(W, vocab, ivocab, term1, term2)
                        glove_cosine_sim[(term1, term2)] = g_cosine_sim


                    except KeyError:
                        print "gensim similarity function got key error"
                        unseen += 1
                row_count += 1
                total += 1

            #---------------------------------------------------------------
            # Word2vec
            #---------------------------------------------------------------
            print("total, seen, unseen :: ", total, total - unseen, unseen)
            human_means_lst = []
            glove_cosine_sim_lst = []
            pairs_found = []
            for key in human_means:
                if key in glove_cosine_sim.keys():
                    print('key found in both lsts ::', key )
                    pairs_found.append(key)
                    glove_cosine_sim_lst.append(float(glove_cosine_sim[key]))
                    human_means_lst.append(float(human_means[key]))
            # perform ks test to check whether dist is normal or not
            # print "humans dist :: ",stats.kstest(human_means_lst, 'norm')
            # print "cosine sim dist :: ",stats.kstest(wordvec_cosine_sim_lst, 'norm')
            # computing spearman's coefficient
            spearman_coeff = stats.spearmanr(human_means_lst, glove_cosine_sim_lst)
            res_tests.append(spearman_coeff[0])
            pearson_coeff = stats.pearsonr(human_means_lst, glove_cosine_sim_lst)
            res_tests.append(pearson_coeff[0])
            print"glove spearman_coeff :: ", spearman_coeff
            print "glove pearson coeff ::", pearson_coeff
            print "for file :: ", filename
            print "---------------------------------------------------------------"
            #-----------------------------------------------------------------------

    return res_tests

def perform_w2v_tests(filename):
    # list to return results
    res_tests = []
    #model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=True)
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
    # model.save_word2vec_format('data/merge300k-vector.txt', binary=False)
    # model['brain']  #raw NumPy vector of a word
    # ------------------------------------------------------------------------
    # Word similarity test
    # print(model.similarity('uterus', 'ovary'))
    # ------------------------------------------------------------------------------
    filename_lst = ['UMNSRS_similarity_mod449_word2vec.csv', 'UMNSRS_relatedness_mod458_word2vec.csv',
                    'MayoSRS.csv', 'MiniMayoSRS.csv']

    for filename in filename_lst:
        row_count = 0
        total = 0
        correct = 0
        unseen = 0
        human_means = {}
        wordvec_cosine_sim = {}

        with open(filename, 'rb') as csvfile:
            # csv_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            csv_reader = csv.reader(csvfile.read().splitlines())
            for row in csv_reader:
                if row_count != 0:
                    if filename.startswith('UMNSRS'):
                        mean_score = float(row[0])
                        term1 = row[2]
                        term2 = row[3]
                    elif filename == 'MayoSRS.csv':
                        mean_score = float(row[0])
                        term1 = row[3]
                        term2 = row[4]
                    elif filename == 'MiniMayoSRS.csv':
                        mean_score = float(row[0])   #physicians
                        term1 = row[4]
                        term2 = row[5]

                    human_means[(term1, term2)] = mean_score
                    try:
                        cosine_sim = model.similarity(term1, term2)
                        wordvec_cosine_sim[(term1, term2)] = cosine_sim

                    except KeyError:
                        print "gensim similarity function got key error"
                        unseen += 1
                row_count += 1
                total += 1

            print("total, seen, unseen :: ", total, total - unseen, unseen)
            human_means_lst = []
            wordvec_cosine_sim_lst = []
            pairs_found = []
            for key in human_means:
                if key in wordvec_cosine_sim.keys():
                    print('key found in both lsts ::', key )
                    pairs_found.append(key)
                    wordvec_cosine_sim_lst.append(float(wordvec_cosine_sim[key]))
                    human_means_lst.append(float(human_means[key]))
            # perform ks test to check whether dist is normal or not
            # print "humans dist :: ",stats.kstest(human_means_lst, 'norm')
            # print "cosine sim dist :: ",stats.kstest(wordvec_cosine_sim_lst, 'norm')
            # computing spearman's coefficient
            spearman_coeff = stats.spearmanr(human_means_lst, wordvec_cosine_sim_lst)
            res_tests.append(spearman_coeff[0])
            pearson_coeff = stats.pearsonr(human_means_lst, wordvec_cosine_sim_lst)
            res_tests.append(pearson_coeff[0])
            print"spearman_coeff :: ", spearman_coeff
            print "pearson coeff ::", pearson_coeff
            print "for file :: ", filename
            print "---------------------------------------------------------------"

    return res_tests

# --------------------------------------------------------------------------------------------
# writing to csv
# --------------------------------------------------------------------------------------------
with open('evals.csv', 'wb') as csvfile:
    '''fieldnames = ['analogy', 'odd-one-out',
                  'spearman-wordsim-353', 'pearson-wordsim-353',
                  'spearman-simlex999', 'pearson-simlex999',
                  'spearman-mayosrs', 'pearson-mayosrs',
                  'spearman-minimayosrs', 'pearson-minimayosrs',
                  'spearman-umnsrs-sim', 'pearson-umnsrs-sim',
                  'spearman-umnsrs-rel', 'pearson-umnsrs-rel']'''

    fieldnames = ['spearman-umnsrs-sim', 'pearson-umnsrs-sim',
                  'spearman-umnsrs-rel', 'pearson-umnsrs-rel',
                  'spearman-mayosrs', 'pearson-mayosrs',
                  'spearman-minimayosrs', 'pearson-minimayosrs']

    parser = argparse.ArgumentParser()
    parser.add_argument('--w2v', default= True, type=bool)
    parser.add_argument('--glove', default= True, type=bool)
    args = parser.parse_args()

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for bin_file in file_list:
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        print "for file : ", bin_file
        #res = perform_w2v_tests(args, bin_file)
        res = perform_glove_tests(args)
        print "res :: ", res
        dict = {'spearman-umnsrs-sim': res[0], 'pearson-umnsrs-sim': res[1],
                'spearman-umnsrs-rel': res[2], 'pearson-umnsrs-rel': res[3],
                'spearman-mayosrs': res[4], 'pearson-mayosrs': res[5],
                'spearman-minimayosrs': res[6], 'pearson-minimayosrs': res[7]
                }
        writer.writerow(dict)
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"





