DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/full-pubmed-text_lower.text
VECTOR_DATA=$DATA_DIR/full_pubmed_lowercase-vector-ng10-hs0-dim200-textwin30-subsample4-mincount5-alpha0.05.bin

pushd ${SRC_DIR} && make; popd

echo -----------------------------------------------------------------------------------------------------
#echo -- Training vectors...
time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 200 -window 30 -negative 10 -hs 0 -sample 1e-4 -threads 24 -binary 1 -min-count 5 -alpha 0.05
  
echo -----------------------------------------------------------------------------------------------------
echo -- distance...
$BIN_DIR/distance $DATA_DIR/$VECTOR_DATA
