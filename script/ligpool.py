import sys
sys.path.append('../')
import torch
import torch.nn as nn

import math, random, sys
import argparse
from fast_jtnn import *
import rdkit

def checksmile(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    smi = Chem.MolToSmiles(mol,kekuleSmiles=True,isomericSmiles=True)
    return smi

validation_denticity = [1,2,3,4,5,6]

def main_sample(output, num, hidden_size=450, latent_size=56, prop_size=2, depthT=20, depthG=3, scs_limit=False, denticity=False, vocab='../data/data_vocab.txt', model_lfs_path='../data/model/JTVAE_model.epoch-89'):
    vocab = [x.strip("\r\n ") for x in open(vocab)]
    vocab = Vocab(vocab)
    if not scs_limit:
        while True:
            user_input = input("Please assign the SCScore for the ligand generation (or press enter if you don't want any limit): \n")
            if user_input.lower() == '':
                print("No limit set for the SCScore.")
                break
            try:
                scs_limit = float(user_input)
                if 1 <= scs_limit <= 5:
                    print(f"SCScore set to: {scs_limit}")
                    # You can perform further processing based on the SCScore here
                    break
                else:
                    print("SCScore between 1 ~ 5. Please try again.")
            except:
                pass
    if not denticity:
        while True:
            user_input = input("Please assign the denticity for the ligand generation (or press enter if you don't want any limit): \n"
                               "[1,2,3,4,5,6]\n")
            if user_input.lower() == '':
                print("No limit set for the denticity.")
                break
            try:
                denticity = int(user_input)
                if 1 <= denticity <= 6:
                    print(f"Denticity set to: {denticity}")
                    # You can perform further processing based on the SCScore here
                    break
                else:
                    print("Denticity between [1,2,3,4,5,6]. Please try again.")
            except:
                pass
    model_lfs = JTPropVAE(vocab, int(hidden_size), int(latent_size),int(prop_size),int(depthT),int(depthG))
    dict_buffer = torch.load(model_lfs_path, map_location='cuda:0')
    model_lfs.load_state_dict(dict_buffer)
    model_lfs.cuda()
    model_lfs.eval()
    decode_smiles_set = set()
    while len(decode_smiles_set) < num:
        decode_smi = model_lfs.sample_prior()
        try:
            tree_batch = [MolTree(decode_smi)]
            _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, vocab, assm=False)
            tree_vecs, _, mol_vecs = model_lfs.encode(jtenc_holder, mpn_holder)
            z_tree_, z_mol_ = model_lfs.T_mean(tree_vecs), model_lfs.G_mean(mol_vecs)
            z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
            lfs_pred,scs_pred = model_lfs.propNN(z_vecs_).squeeze(0)
            lfs_pred = torch.clamp(lfs_pred, min=0, max=1).item()
            scs_pred = torch.clamp(scs_pred, min=1, max=5).item()
            denticity_predict_check = model_lfs.denticity_NN(z_vecs_)
            smiles = checksmile(decode_smi)

            if scs_limit:
                if scs_pred <= scs_limit:
                    _, denticity_predict_check = torch.max(denticity_predict_check,1)
                    denticity_predict_check = (denticity_predict_check + 1).item()
                    if denticity:
                        if denticity_predict_check == denticity:
                            if smiles not in decode_smiles_set:
                                decode_smiles_set.add(smiles)
                    else:
                        if smiles not in decode_smiles_set:
                            decode_smiles_set.add(smiles)
            else:
               if smiles not in decode_smiles_set:
                    decode_smiles_set.add(smiles)
        except:
            pass
    with open(output,'w') as fout:
        for smile in decode_smiles_set:
            fout.write(smile+'\n')


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--nsample', type=int, required=True)
    parser.add_argument('-o','--output_file', required=True)
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--prop_size', type=int, default=2)
    parser.add_argument('--scs_limit',type=int, default=False)
    parser.add_argument('--denticity_limit',type=int, default=False)


    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    args = parser.parse_args()
    
    main_sample(args.output_file, args.nsample, args.hidden_size, args.latent_size, args.prop_size,args.depthT, args.depthG, args.scs_limit, args.denticity_limit)