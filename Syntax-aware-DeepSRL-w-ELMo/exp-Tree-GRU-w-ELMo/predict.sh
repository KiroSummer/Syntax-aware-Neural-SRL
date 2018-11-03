#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

MODEL_PATH="./conll05_model"

INPUT_PATH="../data/srl/conll05.devel.txt"
GOLD_PATH="../data/srl/conll05.devel.props.gold.txt"
TEST_ELMO="../data/conll05_elmo_hdf5/conll05.dev.sen.txt.hdf5"
TEST_DEP_PATH="/data/qrxia/EMNLP2018/PTB_CONLL05_analysis/conll05_autodep/conll05.srl.dev.stanford.pos.conll.autodep.txt"
OUTPUT_PATH="../temp/conll05.devel.out"

#INPUT_PATH="../data/srl/conll05.test.wsj.txt"
#GOLD_PATH="../data/srl/conll05.test.wsj.props.gold.txt"
#TEST_ELMO="../data/conll05_elmo_hdf5/conll05.test.wsj.sen.txt.hdf5"
#TEST_DEP_PATH="/data/qrxia/EMNLP2018/PTB_CONLL05_analysis/conll05_autodep/ptb.english.conll.test.wsj.index.biaffine.autodep.txt"
#OUTPUT_PATH="../temp/conll05.test.wsj.out"

#INPUT_PATH="../data/srl/conll05.test.brown.txt"
#GOLD_PATH="../data/srl/conll05.test.brown.props.gold.txt"
#TEST_ELMO="../data/conll05_elmo_hdf5/conll05.test.brown.sen.txt.hdf5"
#TEST_DEP_PATH="/data/qrxia/EMNLP2018/PTB_CONLL05_analysis/conll05_autodep/conll05.srl.test.brown.stanford.pos.conll.autodep.txt"
#OUTPUT_PATH="../temp/conll05.test.brown.out"

#INPUT_PATH="../data/srl/conll05.test.both.txt"
#GOLD_PATH="../data/srl/conll05.test.both.props.gold.txt"
#TEST_ELMO="../data/conll05_elmo_hdf5/conll05.test.both.sen.txt.hdf5"
#TEST_DEP_PATH="/data/qrxia/EMNLP2018/PTB_CONLL05_analysis/conll05_autodep/ptb.english.conll.test.both.index.biaffine.autodep.txt"
#OUTPUT_PATH="../temp/conll05.test.both.out"

if [ "$#" -gt 0 ]
then
CUDA_VISIBLE_DEVICES=$1 python2 ../src/baseline-w-ELMo-hdf5-full-formulation-TreeGRU/predict.py \
  --model="$MODEL_PATH" \
  --input="$INPUT_PATH" \
  --input_elmo="$TEST_ELMO" \
  --output="$OUTPUT_PATH" \
  --gold="$GOLD_PATH" \
  --gpu=$1
  # --input_dep_trees="$TEST_DEP_PATH" \
else
THEANO_FLAGS="optimizer=fast_compile,floatX=float32" python python/predict.py \
  --model="$MODEL_PATH" \
  --input="$INPUT_PATH" \
  --output="$OUTPUT_PATH" \
  --gold="$GOLD_PATH"
fi

