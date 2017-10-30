DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/full-pubmed-text_lower.text
PHRASES_DATA=$DATA_DIR/full-pubmed-text-lower-phrases
PHRASES_VECTOR_DATA=$DATA_DIR/full-pubmed-text_lower-vectors-phrase-ng10-hs0-dim200-textwin30-subsample4-mincount5-alpha0.05.bin

pushd ${SRC_DIR} && make; popd

if [ ! -e $PHRASES_VECTOR_DATA ]; then
  
  if [ ! -e $PHRASES_DATA ]; then
    
    if [ ! -e $TEXT_DATA ]; then
      wget http://mattmahoney.net/dc/text8.zip -O $DATA_DIR/text8.gz
      gzip -d $DATA_DIR/text8.gz -f
    fi
    echo -----------------------------------------------------------------------------------------------------
    echo -- Creating phrases...
    time $BIN_DIR/word2phrase -train $DATA_DIR/full-pubmed-text_lower.text -output $PHRASES_DATA -threshold 500 -debug 2 -min-count 5
    
  fi

  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors from phrases...
  time $BIN_DIR/word2vec -train $PHRASES_DATA -output $PHRASES_VECTOR_DATA -cbow 0 -size 200 -window 30 -negative 10 -hs 0 -sample 1e-4 -threads 24 -binary 1 -mincount 5 -alpha 0.05
  
fi

echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $PHRASES_VECTOR_DATA
