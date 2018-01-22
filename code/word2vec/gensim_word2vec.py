# import modules & set up logging
import gensim, logging
#from gensim.models import word2vec
import scipy 
from scipy import stats 
import os
import numpy as np 

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#------------------------------------------------------------------------------------------------------
#reading dim bin files 
#-----------------------------------------------------------------------------------------------------

file_DIR = "/lustre/janus_scratch/kawa8312/word_vectors_bins/word_vectors_merge_full/lingpipe" 
f1 = os.path.join(file_DIR,'merge_full-vector-neg15-dim300-textwin11-subsample5.bin')


def perform_tests(filename):
	model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=True, unicode_errors='ignore')
	#model.save_word2vec_format('data/merge300k-vector.txt', binary=False)

	#------------------------------------	
	# things you can do 
	#------------------------------------

	# get raw numpy vector of a word - word embedding given a word 
	#print model['brain'] 
	
	# Word similarity test - computing the word similarity score between 2 words 
	print(model.similarity('uterus', 'ovary'))
	
        # Word analogy test :- king + women - man = queen. topn gives top 5 nearest neighbor words
        #print(model.most_similar(positive=['king', 'women'], negative=['man'], topn=5))
	
	# Nearest words to a given word can be tweaked out of analogy above 
        topn = 20 
	your_word_vector = np.array(model['brain'], dtype='f')
	most_similar_words = model.most_similar( [ your_word_vector ], [], topn)
        print " 20 most similar words :: ",most_similar_words



perform_tests(f1)
