export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

CONFIG="srl_config.json"
MODEL="conll05_model"

TRAIN_PATH="../data/srl/conll05.train.txt"
#TRAIN_ELMO="../data/conll05_elmo/conll05_train_elmo_2.npy"
TRAIN_ELMO_HDF5="../data/conll05_elmo_hdf5/conll05.train.sen.txt.hdf5"
TRAIN_DEP_TREES="/data/qrxia/EMNLP2018/PTB_CONLL05_analysis/conll05_autodep/ptb.english.conll.conll05.srl.train.index.biaffine.autodep.txt"

DEV_PATH="../data/srl/conll05.devel.txt"
#DEV_ELMO="../data/conll05_elmo/conll05_dev_elmo_2.npy"
DEV_ELMO_HDF5="../data/conll05_elmo_hdf5/conll05.dev.sen.txt.hdf5"
DEV_DEP_TREES="/data/qrxia/EMNLP2018/PTB_CONLL05_analysis/conll05_autodep/conll05.srl.dev.stanford.pos.conll.autodep.txt"
GOLD_PATH="../data/srl/conll05.devel.props.gold.txt"

gpu_id=$1
CUDA_VISIBLE_DEVICES=$gpu_id python2 ../src/baseline-w-ELMo-hdf5-full-formulation-TreeGRU//train.py \
   --config=$CONFIG \
   --model=$MODEL \
   --train=$TRAIN_PATH \
   --train_elmo=$TRAIN_ELMO_HDF5 \
   --train_dep_trees=$TRAIN_DEP_TREES \
   --dev=$DEV_PATH \
   --dev_elmo=$DEV_ELMO_HDF5 \
   --dev_dep_trees=$DEV_DEP_TREES \
   --gold=$GOLD_PATH \
   --gpu=$1
