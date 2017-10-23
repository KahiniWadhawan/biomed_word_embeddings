from gensim.models import word2vec

model = word2vec.Word2Vec.load_word2vec_format('../data/merge300k-vector.bin', binary=True)
model.save_word2vec_format('../data/merge300k-vector.txt', binary=False)
