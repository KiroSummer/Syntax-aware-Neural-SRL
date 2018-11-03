# Syntax-aware Neural Semantic Role Labeling
This repository contains codes and configs for training models presented in : [Syntax-aware Neural Semantic Role Labeling](XXX)

We use the deep srl as our basic model. [Deep Semantic Role Labeling](https://github.com/luheng/deep_srl)

## Train
To train our syntax-aware neural models, you should set the train.sh and config.json. Then, run
```bash
nohup ./train.sh 0 > log.txt 2>&1 &
```
where 0 is the GPU id.
We also give an example in ./Syntax-aware-DeepSRL-w-ELMo/exp-Tree-GRU-w-ELMo
