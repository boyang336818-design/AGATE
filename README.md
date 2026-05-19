# AGATE
Code for the paper ["Bidirectional Semantic Accumulation for Deep Graph Representation Learning via Autoencoders"]


The base code is a PyTorch implementation of the Variational Graph Auto-Encoder model described in the paper:
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)


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
python trainnew.py --model="gcn_ae" --dataset-str="ogbn-arxiv" --dw=1 --epochs=200 --walk-length=30 --window-size=30 --number-walks=50 --lr_dw=0.01
```
To train TORC run the following command
```bash
cd gae
python train_TORC.py --model="gcn_ae" --dataset-str="cora" --dw=1 --epochs=200 --walk-length=30 --window-size=30 --number-walks=50 --lr_dw=0.01
```
