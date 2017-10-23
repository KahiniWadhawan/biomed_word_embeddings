DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src

TEXT_DATA=$DATA_DIR/merge_genia_full
PHRASES_DATA=$DATA_DIR/merge_genia_full-phrases
PHRASES_VECTOR_DATA=$DATA_DIR/merge300k-vectors-phrase-textwin11-hs1-dim300-subsample5.bin

pushd ${SRC_DIR} && make; popd

if [ ! -e $PHRASES_VECTOR_DATA ]; then
  
  if [ ! -e $PHRASES_DATA ]; then
    
    if [ ! -e $TEXT_DATA ]; then
      wget http://mattmahoney.net/dc/text8.zip -O $DATA_DIR/text8.gz
      gzip -d $DATA_DIR/text8.gz -f
    fi
    echo -----------------------------------------------------------------------------------------------------
    echo -- Creating phrases...
    time $BIN_DIR/word2phrase -train $DATA_DIR/merge_genia_full -output $PHRASES_DATA -threshold 500 -debug 2 -min-count 10
    
  fi

  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors from phrases...
  time $BIN_DIR/word2vec -train $PHRASES_DATA -output $PHRASES_VECTOR_DATA -cbow 0 -size 300 -window 11 -negative 0 -hs 1 -sample 1e-5 -threads 24 -binary 1 -mincount 10
  
fi

echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $PHRASES_VECTOR_DATA
