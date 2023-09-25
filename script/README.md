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
