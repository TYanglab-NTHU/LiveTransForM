import torch
import torch.nn as nn
import sys
from collections import defaultdict
sys.path.append('../')
import numpy as np
from fast_jtnn import *
from optparse import OptionParser
from rdkit import Chem
import numpy as np
from tl_SSmodel import PropNN
from openbabel import openbabel as ob
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
hidden_size_lfs = 450
hidden_size_ss = 56
latent_size = 56
depthT = 20
depthG = 3
prop_size = 2
valid_oxidation_states = ["Fe2", "Fe3", "Mn2", "Mn3", "Co2", "Co3"]


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
        model_lfs_path = '../data/model/JTVAE_model.epoch-89'
        model_lfs = JTPropVAE(self.vocab, int(self.hidden_size_lfs), int(self.latent_size),int(prop_size),int(self.depthT),int(self.depthG))
        dict_buffer = torch.load(model_lfs_path, map_location='cuda:0')
        model_lfs.load_state_dict(dict_buffer)
        model_lfs.cuda()
        model_lfs.eval()
        self._restored_lfs = True
        self.model_lfs = model_lfs
        model_ss_path = '../data/model/SS_model.epoch-100'
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
    
    def get_vector(self,smile=''):
        smi_target = [smile]
        tree_batch = [MolTree(smi) for smi in smi_target]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
        z_tree_mean = self.model_lfs.T_mean(tree_vecs).cuda()
        z_mol_mean = self.model_lfs.G_mean(mol_vecs).cuda()
        z_tree_log_var = -torch.abs(self.model_lfs.T_var(tree_vecs)).cuda()
        z_mol_log_var = -torch.abs(self.model_lfs.G_var(mol_vecs)).cuda()
        return z_tree_mean,z_mol_mean,z_tree_log_var,z_mol_log_var

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
            
    def IdxTosmiles(self,axial_equ_pair,idx_dict):
        batches = []
        for axial_smiles, equ_smiles in axial_equ_pair.items():
            batch = []
            axial_pair = tuple(idx_dict[key] for key in axial_smiles)
            batch.extend(axial_pair)
            equ_pair = tuple(idx_dict[key] for key in equ_smiles)
            batch.extend(equ_pair)
            batches.append(batch)
        final_batches = [tuple(batch) for batch in batches]  # convert each sub-list to a tuple
        final_batches = list(set(final_batches))  # remove duplicates
        return final_batches
    
    def SpinStatePrediction_from_xyz(self,xyz_file,metal=False):
        if not metal:
            while True:
                user_input = input("Please assign the metal center oxidation state from the following options:\n"
                                "[Fe2, Fe3, Mn2, Mn3, Co2, Co3]\n"
                                "You can also type 'exit' to quit: ")

                if user_input.lower() == 'exit':
                    print("Exiting the program. Goodbye!")
                    break

                if user_input in valid_oxidation_states:
                    print(f"You've chosen the oxidation state: {user_input}")
                    metal = user_input
                    break
                else:
                    print("Invalid input. Please choose from the provided options.")
                    
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
                axial_equ_pair,denticity_dict,idx_dict = Find_coordinates(obmol)
                spin_state_prediction = set()  # Initialize a set to store spin state predictions
                final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict)
                for batch in final_batches:
                    axial_smiles = batch[:2]
                    equ_smiles = batch[2:]
                    spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                    if spin_state != None:
                        spin_state_prediction.add(spin_state)
                if len(spin_state_prediction) != 1:
                    raise ('The model predicts different spin states in different orientations')
                else:
                    print('The model predict SS is %s' %spin_state)
        else:
            raise ValueError('Structure might not be octahedral')

    def Single_mutation_from_xyz(self,xyz_file,metal=False,SS=False,scs_limit=None,step_size=0.01,max_step_size_limit=100):
        obmol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("xyz")
        obConversion.ReadFile(obmol, xyz_file)
        SS_change = False
        # Let user to assign metal oxidation state
        if not metal:
            while True:
                user_input = input("Please assign the metal center oxidation state from the following options:\n"
                                "[Fe2, Fe3, Mn2, Mn3, Co2, Co3]\n"
                                "You can also type 'exit' to quit: ")

                if user_input.lower() == 'exit':
                    print("Exiting the program. Goodbye!")
                    break

                if user_input in valid_oxidation_states:
                    print(f"You've chosen the oxidation state: {user_input}")
                    metal = user_input
                    break
                else:
                    print("Invalid input. Please choose from the provided options.")
        if not scs_limit:
            while True:
                user_input = input("Please assign the SCScore for the mutated ligand (or press enter if you don't want any limit): ")
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

        # Perform single mutation on ligands
        if len(obmol.Separate()) == 1:
            metal_atom_count = 0
            for atom in ob.OBMolAtomIter(obmol):
                if atom.IsMetal():
                    metal_atom_count += 1
            if metal_atom_count != 1:
                raise ValueError('Sorry, the model is not afford to dinulear metal sysyem')
            else:
                axial_equ_pair,denticity_dict,idx_dict = Find_coordinates(obmol)
                spin_state_prediction = set()  # Initialize a set to store spin state predictions
                final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict)
                for batch in final_batches:
                    axial_smiles = batch[:2]
                    equ_smiles = batch[2:]
                    spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                    if spin_state != None:
                        spin_state_prediction.add(spin_state)
                if not SS:
                    if len(spin_state_prediction) != 1:
                        raise ValueError('The model predicts different spin states in different orientations')
                    else:
                        SS = next(iter(spin_state_prediction))
                        print('original spin_state is', SS)

                # Print the available ligands for the user to choose from
                print('Which ligand do you want to mutate?')
                for idx, lig in idx_dict.items():
                    print(f"lig {idx}: {lig}")
                # Get user input for the ligand to mutate
                selected_ligand = False
                while True:
                    user_input = input("Enter the number of the ligand you want to mutate[0,1,2,3....] (or 'exit' to quit): ")
                    if user_input.lower() == 'exit':
                        print("Exiting the selection process.")
                        break
                    try:
                        choice = int(user_input)
                        if 0 <= choice < len(idx_dict.keys()):
                            selected_ligand = idx_dict[choice]
                            print(f"You selected ligand {selected_ligand} for mutation.")
                            break  # Exit the loop if a valid choice is made
                        else:
                            print("Invalid choice. Please enter a valid number.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                if selected_ligand:
                    step_size_limit = 0
                    while not SS_change:
                        count = 0
                        while count <= 10:
                            idx_dict_copy = idx_dict.copy()
                            # change step size in order to generate diverse ligand
                            if count == 10:
                                count = 0
                                step_size += 0.1
                                step_size_limit += 1
                                print('Increase step size!')
                            if step_size_limit > max_step_size_limit:
                                raise ValueError("Couldn't find a suitable mutated ligand, maybe try a different initial step size")
                            count += 1
                            z_tree_mean, z_mol_mean, z_tree_log_var, z_mol_log_var = self.get_vector(selected_ligand)
                            epsilon_tree = create_var(torch.randn_like(z_tree_mean))
                            epsilon_mol = create_var(torch.randn_like(z_mol_mean))
                            z_tree_mean_new = z_tree_mean + torch.exp(z_tree_log_var / 2) * epsilon_tree * step_size
                            z_mol_mean_new = z_mol_mean + torch.exp(z_mol_log_var / 2) * epsilon_mol * step_size
                            smi_new = self.model_lfs.decode(z_tree_mean_new, z_mol_mean_new, prob_decode=False)
                            smi_new = checksmile(smi_new)
                            if smi_new != checksmile(selected_ligand):
                                # Test decode smiles denticity
                                try:
                                    tree_batch = [MolTree(smi_new)]
                                    _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                                    tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                                    z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                                    z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                                    lfs_pred,scs_pred = self.model_lfs.propNN(z_vecs_).squeeze(0)
                                    lfs_pred = torch.clamp(lfs_pred, min=0, max=1).item()
                                    scs_pred = torch.clamp(scs_pred, min=1, max=5).item()
                                    denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                                    if scs_limit:
                                        if scs_pred <= scs_limit:
                                            _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                            denticity_predict_check = (denticity_predict_check + 1).item()
                                            if denticity_predict_check == denticity_dict[selected_ligand]:
                                                idx_dict_copy[choice] = smi_new
                                                spin_state_prediction = set()  # Initialize a set to store spin state predictions
                                                final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict_copy)
                                                for batch in final_batches:
                                                    axial_smiles = batch[:2]
                                                    equ_smiles = batch[2:]
                                                    spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                                                    if spin_state != None:
                                                        spin_state_prediction.add(spin_state)
                                                if len(spin_state_prediction) == 1:
                                                    SS_new = next(iter(spin_state_prediction))
                                                    if SS != SS_new:
                                                        for idx, lig in idx_dict_copy.items():
                                                            print(f"lig {idx}: {lig}")
                                                        SS_change = True
                                                        break
                                    else:
                                        _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                        denticity_predict_check = (denticity_predict_check + 1).item()
                                        if denticity_predict_check == denticity_dict[selected_ligand]:
                                            idx_dict_copy[choice] = smi_new
                                            spin_state_prediction = set()  # Initialize a set to store spin state predictions
                                            final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict_copy)
                                            for batch in final_batches:
                                                axial_smiles = batch[:2]
                                                equ_smiles = batch[2:]
                                                spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                                                if spin_state != None:
                                                    spin_state_prediction.add(spin_state)
                                            if len(spin_state_prediction) == 1:
                                                SS_new = next(iter(spin_state_prediction))
                                                if SS != SS_new:
                                                    for idx, lig in idx_dict_copy.items():
                                                        print(f"lig {idx}: {lig}")
                                                    SS_change = True
                                                    break
                                except:
                                    pass
                    final_point = idx_dict_copy[choice]
                    inital_point = idx_dict[choice]
                    while True:
                        user_input = input("Want to perform interpolation of mutated ligand: (or enter exit to stop) \n  "
                                           "Please enter the number of interpolation point (interger) ")
                        if user_input.lower() == '' or user_input.lower() == 'exit':
                            interpolation = False
                            break
                        try:
                            delta_step = int(user_input)
                            print(f"Number of interpolation set to: {delta_step}")
                            interpolation = True
                            break
                        except:
                            pass
                    if interpolation:
                        print('Interpolation start')
                        mutation_list = []
                        inital_vecs = self.zvec_from_smiles([inital_point])
                        final_vecs = self.zvec_from_smiles([final_point])
                        denticity_ = denticity_dict[selected_ligand]
                        zvecs_inital = inital_vecs[0][0][0]
                        zvecs_final = final_vecs[0][0][0]
                        delta = zvecs_final - zvecs_inital
                        one_piece = delta / delta_step
                        for i in range(delta_step):
                            idx = 0
                            stepsize = 0.05
                            flag = True
                            zvecs = zvecs_inital + one_piece * (i + 1)
                            while flag:
                                if idx == 0 or idx == 10:
                                    idx = 0
                                    stepsize += 0.1
                                try:
                                    smi = self.model_lfs.decode(*torch.split((zvecs).unsqueeze(0),28,dim=1), prob_decode=False)
                                    tree_batch = [MolTree(smi)]
                                    _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                                    tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                                    z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                                    z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                                    denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                                    _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                    denticity_predict_check = (denticity_predict_check + 1).item()
                                    smi_check = checksmile(smi)
                                    if denticity_predict_check != denticity_ or checksmile(smi) == checksmile(inital_point) or checksmile(smi) == checksmile(final_point):
                                        zvecs_mutated = z_vecs_ + torch.randn_like(z_vecs_) * stepsize
                                        smi = self.model_lfs.decode(*torch.split((zvecs_mutated),28,dim=1), prob_decode=False)
                                        tree_batch = [MolTree(smi)]
                                        _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                                        tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                                        z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                                        z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                                        denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                                        _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                        denticity_predict_check = (denticity_predict_check + 1).item()
                                        smi_check = checksmile(smi)
                                        if denticity_predict_check == denticity_ and smi_check != checksmile(inital_point) and smi_check != checksmile(final_point):
                                            flag = False
                                            mutation_list.append([smi_check,i+1])
                                    elif denticity_predict_check == denticity_ and smi_check != checksmile(inital_point) and smi_check != checksmile(final_point):
                                        flag = False
                                        mutation_list.append([smi_check,i+1])
                                    idx += 1 
                                except Exception as e:
                                    print(e)
                                    pass
                        print('Interpolation finished')
                        for i in mutation_list:
                            smiles,idx = i
                            print("Point %s: %s" %(idx,smiles))                        
                
                
                                    
    def Seed_Mutation_from_xyz(self,xyz_file,metal=False,SS=False,scs_limit=None,step_size=0.01,max_step_size_limit=100):
        obmol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInFormat("xyz")
        obConversion.ReadFile(obmol, xyz_file)
        # Let user to assign metal oxidation state
        if not metal:
            while True:
                user_input = input("Please assign the metal center oxidation state from the following options:\n"
                                "[Fe2, Fe3, Mn2, Mn3, Co2, Co3]\n"
                                "You can also type 'exit' to quit: ")

                if user_input.lower() == 'exit':
                    print("Exiting the program. Goodbye!")
                    break

                if user_input in valid_oxidation_states:
                    print(f"You've chosen the oxidation state: {user_input}")
                    metal = user_input
                    break
                else:
                    print("Invalid input. Please choose from the provided options.")
        if not scs_limit:
            while True:
                user_input = input("Please assign the SCScore for the mutated ligand (or press enter if you don't want any limit): ")
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
                
        # Perform seed mutation on ligands
        if len(obmol.Separate()) == 1:
            metal_atom_count = 0
            for atom in ob.OBMolAtomIter(obmol):
                if atom.IsMetal():
                    metal_atom_count += 1
            if metal_atom_count != 1:
                raise ValueError('Sorry, the model is not afford to dinulear metal sysyem')
            else:
                axial_equ_pair,denticity_dict,idx_dict = Find_coordinates(obmol)
                spin_state_prediction = set()  # Initialize a set to store spin state predictions
                final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict)
                for batch in final_batches:
                    axial_smiles = batch[:2]
                    equ_smiles = batch[2:]
                    spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                    if spin_state != None:
                        spin_state_prediction.add(spin_state)
                if not SS:
                    if len(spin_state_prediction) != 1:
                        raise ValueError('The model predicts different spin states in different orientations')
                    else:
                        SS = next(iter(spin_state_prediction))
                        print('original SS is', SS)

                # Print the available ligands for the user to choose from
                print('Seed mutation start')
                idx_dict_copy = idx_dict.copy()
                for idx, lig in idx_dict.items():
                    print(f"lig {idx}: {lig}, Mutation start!")
                    step_size_limit = 0
                    count = 0
                    while count <= 10:
                        # change step size in order to generate diverse ligand
                        if count == 10:
                            count = 0
                            step_size += 0.1
                            step_size_limit += 1
                            print('Increase step size!')
                            print(step_size_limit)
                        if step_size_limit > max_step_size_limit:
                            raise ValueError("Couldn't find a suitable mutated ligand, maybe try a different initial step size")
                        count += 1
                        z_tree_mean, z_mol_mean, z_tree_log_var, z_mol_log_var = self.get_vector(lig)
                        epsilon_tree = create_var(torch.randn_like(z_tree_mean))
                        epsilon_mol = create_var(torch.randn_like(z_mol_mean))
                        z_tree_mean_new = z_tree_mean + torch.exp(z_tree_log_var / 2) * epsilon_tree * step_size
                        z_mol_mean_new = z_mol_mean + torch.exp(z_mol_log_var / 2) * epsilon_mol * step_size
                        smi_new = self.model_lfs.decode(z_tree_mean_new, z_mol_mean_new, prob_decode=False)
                        smi_new = checksmile(smi_new)
                        if smi_new != checksmile(lig):
                            # Test decode smiles denticity
                            try:
                                tree_batch = [MolTree(smi_new)]
                                _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, self.model_lfs.vocab, assm=False)
                                tree_vecs, _, mol_vecs = self.model_lfs.encode(jtenc_holder, mpn_holder)
                                z_tree_, z_mol_ = self.model_lfs.T_mean(tree_vecs), self.model_lfs.G_mean(mol_vecs)
                                z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
                                lfs_pred,scs_pred = self.model_lfs.propNN(z_vecs_).squeeze(0)
                                lfs_pred = torch.clamp(lfs_pred, min=0, max=1).item()
                                scs_pred = torch.clamp(scs_pred, min=1, max=5).item()
                                denticity_predict_check = self.model_lfs.denticity_NN(z_vecs_)
                                if scs_limit:
                                    if scs_pred <= scs_limit:
                                        _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                        denticity_predict_check = (denticity_predict_check + 1).item()
                                        if denticity_predict_check == denticity_dict[lig]:
                                            idx_dict_copy[idx] = smi_new
                                            break
                                else:
                                    _, denticity_predict_check = torch.max(denticity_predict_check,1)
                                    denticity_predict_check = (denticity_predict_check + 1).item()
                                    if denticity_predict_check == denticity_dict[lig]:
                                        idx_dict_copy[idx] = smi_new
                                        break
                            except:
                                pass
                print('\nMutation Finished \n___________________________\n')
                comparsion_ligs = [(i, j) for i, j in zip(idx_dict.values(), idx_dict_copy.values())]
                for idx, (original_lig, mutated_lig) in enumerate(comparsion_ligs):
                    print(f'Lig{idx} : Original: {original_lig}, Mutated: {mutated_lig}')
                    
                spin_state_prediction = set()  # Initialize a set to store spin state predictions
                final_batches = self.IdxTosmiles(axial_equ_pair,idx_dict)
                for batch in final_batches:
                    axial_smiles = batch[:2]
                    equ_smiles = batch[2:]
                    spin_state = self.obtain_ss_unique(axial_smiles, equ_smiles, metal)
                    if spin_state != None:
                        spin_state_prediction.add(spin_state)
                if len(spin_state_prediction) != 1:
                    raise ValueError('The model predicts different spin states in different orientations')
                else:
                    SS = next(iter(spin_state_prediction))
                    print('\nSS after mutation is', SS)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-v', dest="vocab",default='../data/data_vocab.txt')
    parser.add_option('-x', dest="xyz_file")
    parser.add_option('-s',dest="step_size",type=int,default=0.01)
    parser.add_option('-m',dest="max_step_size",type=int,default=100)    
    opts,args = parser.parse_args()
    vocab = opts.vocab
    xyz_file = opts.xyz_file
    step_size = opts.step_size
    max_step_size = opts.max_step_size
    gen_complex = Genlig(vocab)
    gen_complex.restore()
    valid_assign_list = [str(0),str(1),str(2)]
    while True:
        user_input = input("Please choose the mode want to used:\n"
                        "0: SpinStatePrediction_from_xyz\n"
                        "1: SingleMutation_from_xyz\n"
                        "2: SeedMutation_from_xyz\n"
                        "You can also type 'exit' to quit: ")

        if user_input.lower() == 'exit':
            raise ValueError("Exiting the program. Goodbye!")
            
        if user_input in valid_assign_list:
            if user_input == str(0):
                print(f"You've chosen the model : SpinStatePrediction_from_xyz\n")
            elif user_input == str(1):
                print(f"You've chosen the model : SingleMutation_from_xyz\n")
            elif user_input == str(2):
                print(f"You've chosen the model : SeedMutation_from_xyz\n")
            assign = user_input
            break
        else:
            print("Invalid input. Please choose from the provided options.")
    
    if assign == str(0):
        gen_complex.SpinStatePrediction_from_xyz(xyz_file)
    elif assign == str(1):
        gen_complex.Single_mutation_from_xyz(xyz_file,step_size=step_size,max_step_size_limit=max_step_size)
    elif assign == str(2):
        gen_complex.Seed_Mutation_from_xyz(xyz_file)
    