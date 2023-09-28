# LiveTransForM

## About
**LiveTransForm** joint semi-supervised JT-VAE (SSVAE) and artificial neural network (ANN) model for design of TM complexes as described in the manuscript "A Joint Semi-Supervised Variational Autoencoder and Transfer Learning Model for Designing Molecular Transition Metal Complexes".

It allows for the embedding of molecules into a continuous latent space, and subsequent used transfer learning transition metal complexes spin state prediction 

This repository contains the code we used in training of SSVAE, ANN as described in the manuscript. We have also included utility tools for decoding and encoding molecules, so that you can fit and train your own models.

## Installation
```sh
git clone https://github.com/TYanglab-NTHU/LiveTransForM
cd LiveTransForM
```

```sh
conda env create -f environment.yml
conda activate LiveTransForm
```

Installation should take under ten minutes in most cases.

## Ligand Generation!
There is an IPython Notebook that you can open using Jupyter and/or other notebooks (not tested) named `script/Genlig.ipynb`. It contains steps how to generate ligand.

## Usage

# Training of Semi-Supervised Junction Tree VAE

## Deriving Vocabulary
If you are running our code on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run
```
cd ../fast_jtnn
python mol_tree.py -i ./../data/jtvae_smi.txt -v ./../data/vocab.txt
```
This gives you the vocabulary of cluster labels over the dataset `jtvae_smi.txt`.

## Training
Step 1: Preprocess the data:
```
cd ../script
python preprocess_vae.py --train ./../data/jtvae_smi.txt --split 100 --jobs 40 --output ./../data/vae_training_data
```
This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets.

Step 2: Train VAE model with KL annealing.
```
python vae_train.py --train ./../data/vae_training_data --vocab ./../data/jtvae_smi.txt --save_dir vae_model/
```
Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 700` means that beta will not increase within first 700 training steps. It is recommended because using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.002 --kl_anneal_iter 460` means beta will increase by 0.002 every 460 training steps (batch updates). You should observe that the KL will decrease as beta increases.

`--max_beta 0.016 ` sets the maximum value of beta to be 0.016.

`--save_dir vae_model`: the model will be saved in vae_model/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters.

# Training of ANN model fro Spin State Prediction

## Training
```
python tl_SSmodel.py -v ../data/latent_vec.csv --ss_label -i ../data/TL_model_data.pkl --ss_label../data/TL_model_label.json --output ./../data/SS_VAE_model
```

# Usage of ANN model fro Spin State Prediction
```
python Find_ss.py -x <input xyz file> -s 0.01 -m 50
```
`-x xyzfile` specifies the file xyzfile to be mutated.

`-s` sets the initial step size.

`-m` sets the maximum step size.

## License
License details can be found in the LICENSE file.
# Requirements
* RDKit (version >= 2022.09.5)
* Python (version >= 3.7.13)
* PyTorch (version >= 1.13.0)
* Openbabel (version >= 3.1.0)

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.
