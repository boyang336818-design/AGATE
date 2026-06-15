# AGATE
Code for the paper ["Bidirectional Semantic Accumulation for Deep Graph Representation Learning via Autoencoders"]

The base code is a PyTorch implementation of the Variational Graph Auto-Encoder model described in the paper:
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)

## 🚀 Code Availability

The core code for this paper has been uploaded. We are currently organizing and cleaning up the remaining scripts, and the complete codebase will be released shortly. 

Stay tuned!


### Requirements
- Python 3
- PyTorch 0.4 

### To train a model run the following command
```bash
cd gae
python train.py --model="gcn_ae" --dataset-str="citeseer" --dw=1 --epochs=200 --walk-length=30 --window-size=30 --number-walks=50 --lr_dw=0.01
python train.py --model="gcn_ae" --dataset-str="Pubmed" --dw=1 --epochs=2000 --walk-length=80 --window-size=70 --number-walks=30 --lr_dw=0.01
python train.py --model="gcn_ae" --dataset-str="cora" --dw=1 --epochs=200 --walk-length=30 --window-size=30 --number-walks=50 --lr_dw=0.01
python trainnew.py --model="gcn_ae" --dataset-str="photo" --dw=1 --epochs=200 --walk-length=30 --window-size=30 --number-walks=50 --lr_dw=0.01
python trainnew.py --model="gcn_ae" --dataset-str="computers" --dw=1 --epochs=200 --walk-length=30 --window-size=30 --number-walks=50 --lr_dw=0.01
python trainnew.py --model="gcn_ae" --dataset-str="arxiv" --dw=1 --epochs=200 --walk-length=30 --window-size=30 --number-walks=50 --lr_dw=0.01
```
To train TORC run the following command
```bash
cd gae
python train_TORC.py --model="gcn_ae" --dataset-str="cora" --dw=1 --epochs=200 --walk-length=30 --window-size=30 --number-walks=50 --lr_dw=0.01
```

## Citation

If you find this code useful in your research, please consider citing our paper:
@article{XIE2026114168,
title = {Bidirectional semantic accumulation for deep graph representation learning via autoencoders},
journal = {Pattern Recognition},
volume = {180},
pages = {114168},
year = {2026},
author = {Chengxin Xie and Qiya Song and Meng Liu and Yuan Luo and Jingui Huang}
}
