import torch
import torch.nn as nn
import sys
from collections import defaultdict
sys.path.append('../')
import numpy as np
from fast_jtnn import *
from optparse import OptionParser
from rdkit import Chem
from tl_SSmodel import PropNN
import numpy as np
from rdkit import Chem
from SS_utils import *
from openbabel import openbabel as ob

hidden_size_lfs = 450
hidden_size_ss = 56
latent_size = 56
depthT = 20
depthG = 3
prop_size = 2

class Genlig():
    def __init__(self,vocab,hidden_size_lfs=hidden_size_lfs,hidden_size_ss=hidden_size_ss,latent_size=latent_size,depthT=depthT,depthG=depthG,prop_size=prop_size):
        self.hidden_size_lfs = hidden_size_lfs
        self.hidden_size_ss = hidden_size_ss
        self.latent_size = latent_size
        self.depthT = depthT
        self.depthG = depthG
        self.prop_size = prop_size
        self.vocab = [x.strip("\r\n ") for x in open(vocab)]
        self.vocab = Vocab(self.vocab)
        self._restored_lfs = False
        self._restored_ss = False
        self.zeff_dict = {'Mn2':0.21187286929865048, 'Mn3':0.47965663466222397,
            'Fe2':0.7091855764024294, 'Fe3':0.9769693417660021,
            'Co2':1.2064982835062081, 'Co3':1.474282048869781,
            }
        self.ionization_energy = {'Mn2':0.1489329431875911,'Mn3':1.1052894371297863,
            'Fe2':-0.01555472321511081, 'Fe3':1.3070142633261608,
            'Co2':0.1397790435114532,'Co3':1.1086797703431708,
            }
        self.ionic_radi = {'Mn2':-0.4696827439384181,'Mn3':-0.7082050151547933,
            'Fe2':-0.6286975914160016,'Fe3':-1.0262347101099603,
            'Co2':-0.5226876930976125,'Co3':-0.8009636761833837
            }
        self.ss_dict = {'HS': 1, 'LS': 0, 1: 'HS', 0:'LS'}
        
    def restore(self):
        model_lfs_path = './vae_model/model.epoch-89'
        model_lfs = JTPropVAE(self.vocab, int(self.hidden_size_lfs), int(self.latent_size),int(prop_size),int(self.depthT),int(self.depthG))
        dict_buffer = torch.load(model_lfs_path, map_location='cuda:0')
        model_lfs.load_state_dict(dict_buffer)
        model_lfs.cuda()
        model_lfs.eval()
        self._restored_lfs = True
        self.model_lfs = model_lfs
        model_ss_path = '/work/scorej41075/JTVAE_horovod/SS_model_modelfinal/ratio-7-1-2_nolabel/batchsize_128_lr_0.001_dropout_50_lr-iter_10/model.epoch-422'
        dict_buffer = torch.load(model_ss_path, map_location='cuda:0')
        model_ss = PropNN(28,56,0.5)
        model_ss.load_state_dict(dict_buffer)
        model_ss.cuda()
        model_ss.eval()
        self._restored_ss = True
        self.model_ss = model_ss
    
    def ss_check(self,input_ss):
        try:
            if type(input_ss) == str:
                spinstate = str.upper(input_ss)
                spinstate = self.ss_dict[spinstate]
            if type(input_ss) == int or type(input_ss) == float:
                spinstate = int(input_ss)
        except:
            raise ValueError('Have you assign the spinstate?')
        return spinstate

    def zvec_from_smiles(self,smis):
        self.restore()
        z_vecs_tot = []
        tree_batch = [MolTree(smi) for smi in smis]
        _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
        z_tree, z_mol = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
        z_vecs = torch.cat((z_tree,z_mol),dim=1)
        prop_pred = self.model_lfs.propNN(z_vecs)
        lfs_pred = torch.clamp(prop_pred[0], min=0, max=1)
        lfs_check,scs_check = [prop.item() for prop in lfs_pred],[prop[1].item() for prop in prop_pred]
        denticity_predict = self.model_lfs.denticity_NN(z_vecs)
        _, denticity_predict = torch.max(denticity_predict,1)
        denticity_predict = denticity_predict + 1
        denticity_count = denticity_predict.sum()
        for z_vec,denticity in zip(z_vecs,denticity_predict):
            z_vecs_tot.append([z_vec,denticity])
        return z_vecs_tot,denticity_count.item(),denticity_predict.tolist(),lfs_check,scs_check
    
    def smile2zvec(self,smis):
        tree_batch = [MolTree(smi) for smi in smis]
        _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
        z_tree, z_mol = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
        z_vecs = torch.cat((z_tree,z_mol),dim=1).cuda()
        return z_vecs
    
        
    def obtain_ss_sorted(self,z_vecs_tot,metal):
        self.restore()
        axil_tot,equ_tot,combintaion = sort_z_vecs(z_vecs_tot)
        axil_batch_tensor = torch.stack(axil_tot, dim=0).sum(dim=0)
        equ_batch_tensor = torch.stack(equ_tot, dim=0).sum(dim=0)
        complex_predict = self.model_ss.ss_NN(torch.cat((axil_batch_tensor.cuda(),equ_batch_tensor.cuda(),torch.Tensor([self.zeff_dict[metal]]).cuda()),dim=0))
        softmaxfun = nn.Softmax(dim=0)
        complex_predict = softmaxfun(complex_predict)
        _, complex_predict = torch.max(complex_predict,0)
        spin_state = self.ss_dict[complex_predict.item()]
        return spin_state

    def obtain_ss_unique(self,axil_smis,equ_vecs,metal):
        self.restore()
        if len(axil_smis) != 2 or len(equ_vecs) != 4:
            return None
        axil_vecs = self.smile2zvec(axil_smis)
        equ_vecs = self.smile2zvec(equ_vecs)
        axil_batch_tensor = torch.stack(tuple(axil_vecs), dim=0).sum(dim=0)
        equ_batch_tensor = torch.stack(tuple(equ_vecs), dim=0).sum(dim=0)
        zeff = torch.tensor(self.zeff_dict[metal])
        ionic_radi = torch.tensor(self.ionic_radi[metal])
        ionization_energy = torch.tensor(self.ionization_energy[metal])
        metal_inform_batch = torch.stack([zeff, ionic_radi, ionization_energy]).to(torch.float32)
        complex_predict = self.model_ss.ligand_NN(torch.cat((axil_batch_tensor.cuda(),equ_batch_tensor.cuda(),metal_inform_batch.cuda()),dim=0))
        _, complex_predict = torch.max(complex_predict,0)
        spin_state = self.ss_dict[complex_predict.item()]
        return spin_state

    def gen_comp(self, scs_limit,metal,desire_ss='',desire_denticity=6,zvec_from_smiles=False):
        # Initialize variables
        self.restore()
        smis = []                  # List of SMILES strings
        denticity_origin = []      # List of predicted denticity values for each iteration
        scs_origin = []            # List of predicted SCS values for each iteration
        lfs_origin = []            # List of predicted LFS values for each iteration
        scs_check = []             # List of actual SCS values for each iteration
        lfs_check = []             # List of actual LFS values for each iteration
        denticity_count = 0        # Counter for number of denticity values predicted
        lfs_pred_tot = 0           # Total predicted LFS value for all predicted denticity values
        z_vecs_tot = []            # List of z vectors for each iteration
        flag_lig = True                # Flag to continue iteration
        flag_ss = True
        if desire_ss != '':
            desire_ss = self.ss_check(desire_ss)
        print('Searching desire ligand')
        while flag_ss:
            while flag_lig:
                # Generate z vectors and predict denticity and property values
                while denticity_count != desire_denticity:   # Loop until 6 denticity values are predicted
                    # Generate random z vectors
                    z_tree = torch.randn(1, 28).cuda()
                    z_mol = torch.randn(1, 28).cuda()
                    z_vecs = torch.cat((z_tree, z_mol), dim=1)
                    # Predict LFS and SCS values using the model_lfs
                    prop_pred = self.model_lfs.propNN(z_vecs)
                    prop_pred = prop_pred.squeeze(0)
                    lfs_pred = torch.clamp(prop_pred[0], min=0, max=1)
                    scs_pred = prop_pred[1].item()
                    # Predict denticity using the model
                    denticity_output = self.model_lfs.denticity_NN(z_vecs)
                    _, denticity_predict = torch.max(denticity_output, 1)
                    lfs_pred = torch.clamp(prop_pred.squeeze(0)[0], min=0, max=1)
                    denticity_predict = denticity_predict + 1
                    # Add z vector and predicted values to lists
                    if scs_pred < scs_limit:
                        scs_origin.append(scs_pred)
                        lfs_origin.append(lfs_pred.item())
                        denticity_origin.append(denticity_predict.item())
                        denticity_count += denticity_predict.item()
                        z_vecs_tot.append([z_vecs, denticity_predict.item()])
                        for i in range(denticity_predict):
                            lfs_pred_tot += lfs_pred.item()
                        if denticity_count > 6:   # Reset variables if not all 6 denticity values are predicted
                            denticity_count = 0
                            lfs_pred_tot = 0
                            z_vecs_tot = []
                            scs_origin = []
                            lfs_origin = []
                            denticity_origin = []
                # Decode z vectors to SMILES strings and store in list
                smis = [self.model_lfs.decode(*torch.split(z[0],28,dim=1), prob_decode=False) for z in sorted(z_vecs_tot, key=lambda x: x[1:])]
                # Attempt to create MolTree objects from SMILES strings and calculate denticiy values
                try:
                    tree_batch = [MolTree(smi) for smi in smis]
                    _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                    tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                    z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                    z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                    prop_pred_ = self.model_lfs.propNN(z_vecs_)
                    lfs_pred = torch.clamp(prop_pred_[0], min=0, max=1)
                    lfs_check,scs_check = [[lfs.item() for lfs in lfs_pred],[prop[1].item() for prop in prop_pred_]]     
                    denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                    _, denticity_predict_check = torch.max(denticity_predict_check,1)
                    denticity_predict_check = denticity_predict_check + 1
                    decode_error_lfs = [(check - origin) for origin, check in zip(lfs_origin,lfs_check)]
                    decode_error_scs = [(check - origin) for origin, check in zip(scs_origin,scs_check)]
                    if denticity_predict_check.tolist() == denticity_origin:
                        flag_lig = False
                    else:
                        denticity_count = 0
                        lfs_pred_tot = 0
                        z_vecs_tot = []
                        scs_origin = []
                        lfs_origin = []
                        denticity_origin = []
                except Exception as e:
                    print(e)
                    pass                
            spinstate = self.obtain_ss_sorted(z_vecs_tot,metal)
            if desire_ss == '' or spinstate == desire_ss:
                flag_ss = False
                return smis,scs_origin,lfs_origin,denticity_origin,decode_error_lfs,decode_error_scs,spinstate
            else:
                denticity_count = 0
                lfs_pred_tot = 0
                z_vecs_tot = []
                scs_origin = []
                lfs_origin = []
                denticity_origin = []
                    
    # def lig_pool(self,desire_denticity,scs_limit,numligs):
    #     self.restore()
    #     decode_smis = []                  # List of SMILES strings
    #     denticity_predict = 0
    #     flag = True                # Flag to continue iteration
    #     smis = set()
    #     with open('/home/scorej41075/program/SS_model/data/success_smi_2.csv','r') as f1:
    #         exit_smis = f1.read().splitlines()
    #     success_smis = set()
    #     for smi in exit_smis:
    #         try:
    #             mol = Chem.MolFromSmiles(smi)  
    #             smile = Chem.MolToSmiles(mol,canonical=True)    
    #             success_smis.add(smile)  
    #         except:
    #             pass
    #     print("Starting generate ligands")
    #     while flag:
    #         if len(decode_smis) >= int(numligs):
    #             flag = False
    #             print('Decode Finished')
    #         # Generate z vectors and predict denticiy values and property values
    #         while desire_denticity != denticity_predict:   # Loop until 6 denticiy values are predicted
    #             z_tree = torch.randn(1,28).cuda()
    #             z_mol = torch.randn(1,28).cuda()
    #             z_vecs = torch.cat((z_tree,z_mol),dim=1)
    #             prop_pred = self.model_lfs.propNN(z_vecs.cuda())
    #             denticity_predict = self.model_lfs.denticity_NN(z_vecs)
    #             _, denticity_predict = torch.max(denticity_predict,1)
    #             denticity_predict = denticity_predict.item() + 1

    #         prop_pred = prop_pred.squeeze(0)
    #         lfs_pred = torch.clamp(prop_pred[0],min=0,max=1)
    #         scs_pred = prop_pred[1].item()
    #         if scs_pred < scs_limit:
    #             # Decode z vectors to SMILES strings and store in list
    #             smi = self.model_lfs.decode(*torch.split(z_vecs,28,dim=1), prob_decode=False)
    #         # Decode z vectors to SMILES strings and store in list
    #             try:
    #                 tree_batch = [MolTree(smi)]
    #                 _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
    #                 tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
    #                 z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
    #                 z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
    #                 prop_pred_ = self.model_lfs.propNN(z_vecs_)
    #                 prop_pred_ = prop_pred_.squeeze(0)
    #                 denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
    #                 _, denticity_predict_check = torch.max(denticity_predict_check,1)
    #                 denticity_predict_check = denticity_predict_check + 1
    #                 if denticity_predict == denticity_predict_check:
    #                     if smi not in smis and smi not in success_smis:
    #                         smis.add(smi)
    #                         decode_smis.append([smi,lfs_pred.item(),scs_pred,denticity_predict])
    #             except Exception as e:
    #                 print(e)
    #                 pass
    #         denticity_predict = 0
    #     return decode_smis
    
    def combine_smis(self,smis_input,scs_limit,metal,desire_ss = ''):
        print("Starting generate ligands")
        z_vecs_tot_init,denticity_count,denticity_init,lfs_check,scs_check = self.zvec_from_smiles(smis_input)
        print(denticity_init)
        denticity_desire = 6 - denticity_count
        denticity_gen = 0
        scs_origin = []
        lfs_origin = []
        denticity_origin = []
        z_vecs_tot = []
        flag_combine = True
        if desire_ss != '':
            desire_ss = self.ss_check(desire_ss)
        while flag_combine:
            z_vecs_tot_init_copy = z_vecs_tot_init.copy()
            lfs_check_copy = lfs_check.copy()
            scs_check_copy = scs_check.copy()    
            denticity_copy = denticity_init.copy()
            while denticity_gen != denticity_desire:
                z_tree = torch.randn(1, 28).cuda()
                z_mol = torch.randn(1, 28).cuda()
                z_vecs = torch.cat((z_tree, z_mol), dim=1)
                # Predict LFS and SCS values using the model_lfs
                prop_pred = self.model_lfs.propNN(z_vecs)
                prop_pred = prop_pred.squeeze(0)
                lfs_pred = torch.clamp(prop_pred[0], min=0, max=1)
                scs_pred = prop_pred[1].item()
                # Predict denticity using the model
                denticity_output = self.model_lfs.denticity_NN(z_vecs)
                _, denticity_predict = torch.max(denticity_output, 1)
                lfs_pred = torch.clamp(prop_pred.squeeze(0)[0], min=0, max=1)
                denticity_predict = denticity_predict + 1
                # Add z vector and predicted values to lists
                if scs_pred < scs_limit:
                    scs_origin.append(scs_pred)
                    lfs_origin.append(lfs_pred.item())
                    denticity_origin.append(denticity_predict.item())
                    denticity_gen += denticity_predict.item()
                    z_vecs_tot.append([z_vecs, denticity_predict.item()])
                    if denticity_gen > denticity_desire:   # Reset variables if not all 6 denticity values are predicted
                        denticity_gen = 0
                        z_vecs_tot = []
                        scs_origin = []
                        lfs_origin = []
                        denticity_origin = [] 
            smis = [self.model_lfs.decode(*torch.split(z[0],28,dim=1), prob_decode=False) for z in z_vecs_tot]
        # Attempt to create MolTree objects from SMILES strings and calculate denticiy values
            try:
                tree_batch = [MolTree(smi) for smi in smis]
                _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                prop_pred_ = self.model_lfs.propNN(z_vecs_)
                lfs_pred = torch.clamp(prop_pred_[0], min=0, max=1)
                lfs_check,scs_check = [[lfs.item() for lfs in lfs_pred],[prop[1].item() for prop in prop_pred_]]     
                denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                _, denticity_predict_check = torch.max(denticity_predict_check,1)
                denticity_predict_check = denticity_predict_check + 1
                # decode_error_lfs = [(check - origin) for origin, check in zip(lfs_origin,lfs_check)]
                # decode_error_scs = [(check - origin) for origin, check in zip(scs_origin,scs_check)]
                print(denticity_predict_check.tolist())
                print(denticity_origin)
                if denticity_predict_check.tolist() == denticity_origin:
                    for z_vec,lfs,scs,denticity,in zip(z_vecs_tot,lfs_check,scs_check,denticity_origin):
                        z_vecs_tot_init_copy.append(z_vec)
                        lfs_check_copy.append(lfs)
                        scs_check_copy.append(scs)
                        denticity_copy.append(denticity)
                    spin_state = gen_complex.obtain_ss_sorted(z_vecs_tot_init_copy,metal)
                    if desire_ss == '' or self.ss_dict(spin_state) == desire_ss:
                        flag_combine = False
                    for smi in smis:
                        smis_input.append(smi)
                    return spin_state,smis_input,lfs_check_copy,scs_check_copy,denticity_copy
            except Exception as e:
                print(e)
                pass 
            scs_origin = []
            lfs_origin = []
            denticity_origin = []
            z_vecs_tot = []
            denticity_gen = 0
    
    def SpinStatePrediction_from_xyz(self,xyz_file,metal=None):
        if not metal:
            raise ValueError('Didn\'t assign metal center oxidation states')
        obmol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("xyz")
        obConversion.ReadFile(obmol, xyz_file)
        metal_atom_count = 0
        spin_state_prediction = set()
        if len(obmol.Separate()) == 1:
            for atom in ob.OBMolAtomIter(obmol):
                if atom.IsMetal():
                    metal_atom_count += 1
            if metal_atom_count != 1:
                raise ValueError('Sorry, the model is not afford to dinulear metal sysyem')
            else:
                axial_equ_pair = Find_coordinates(obmol)
                for axial_smiles,equ_smiles in axial_equ_pair.items():
                    spin_state = gen_complex.obtain_ss_unique(axial_smiles,equ_smiles[0],'Co3')
                    spin_state_prediction.add(spin_state)
                if spin_state != None:
                    if len(spin_state_prediction) != 1:
                        raise ValueError('The model predicts different spin states in different orientations')
                    else:
                        return spin_state
                else:
                    raise ValueError('Model didn\'t has suitable vocab for the input')

                
        else:
            raise ValueError('Structure might not be octahedral')

    def Single_mutation_from_xyz(self,xyz_file):
        obmol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("xyz")
        obConversion.ReadFile(obmol, xyz_file)
        metal_atom_count = 0
        spin_state_prediction = set()
        if len(obmol.Separate()) == 1:
            for atom in ob.OBMolAtomIter(obmol):
                if atom.IsMetal():
                    metal_atom_count += 1
            if metal_atom_count != 1:
                raise ValueError('Sorry, the model is not afford to dinulear metal sysyem')
            else:
                axial_equ_pair = Find_coordinates(obmol)
                smiles_set = set()
                for axial,equ in axial_equ_pair.items():
                    for i in axial:
                        smiles_set.add(i)
                    for j in equ[0]:
                        smiles_set.add(j)
                



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-v', dest="vocab",default='../data/data_vocab.txt')
    opts,args = parser.parse_args()
    
    vocab = opts.vocab
    gen_complex = Genlig(vocab)
    scs_limit = 3
    gen_complex.restore()
    spinstate = gen_complex.SpinStatePrediction_from_xyz('/work/scorej41075/Build3D_data1/ccdc_database/BILKUC/BILKUC_0_1_tpss_d4_lanl2dz_631gpol_Co3.opt/20230829224830/BILKUC_0_1_tpss_d4_lanl2dz_631gpol_Co3.opt.xyz','Co3')
