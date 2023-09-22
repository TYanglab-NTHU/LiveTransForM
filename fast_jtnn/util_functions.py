import numpy as np
from openbabel import openbabel as ob
from collections import defaultdict
from rdkit import Chem
from .mol_tree import Vocab, MolTree
from .datautils import tensorize
import math,sys,os
import torch
from torch.autograd import Variable
from .nnutils import create_var

    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def checksmile(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    smi = Chem.MolToSmiles(mol,kekuleSmiles=True,isomericSmiles=True)
    return smi

def sort_z_vecs(z_vecs_tot):
    z_vecs_tot = sorted(z_vecs_tot, key=lambda x: x[1:])
    prefilter = [i[1] for i in z_vecs_tot]
    
    axil_tot = []
    equ_tot = []

    for z_vec in z_vecs_tot:
        z_vec_ = z_vec[0].squeeze(0)
        denticity = z_vec[1]

        if prefilter == [1, 1, 1, 1, 1, 1]:
            if len(axil_tot) < 2:
                axil_tot.append(z_vec_)
            else:
                equ_tot.append(z_vec_)

        elif prefilter == [1, 1, 1, 1, 2]:
            if len(axil_tot) < 2:
                axil_tot.append(z_vec_)
            else:
                for i in range(denticity):
                    equ_tot.append(z_vec_)

        elif prefilter == [1, 1, 1, 3]:
            if denticity == 1:
                if len(axil_tot) < 1:
                    axil_tot.append(z_vec_)
                else:
                    equ_tot.append(z_vec_)
            else:
                axil_tot.append(z_vec_)
                for i in range(denticity - 1):
                    equ_tot.append(z_vec_)

        elif prefilter == [1, 1, 4]:
            if len(axil_tot) < 2:
                axil_tot.append(z_vec_)
            else:
                for i in range(denticity):
                    equ_tot.append(z_vec_)

        elif prefilter == [1, 2, 3]:
            if denticity == 1:
                equ_tot.append(z_vec_)
            else:
                axil_tot.append(z_vec_)
                for i in range(denticity - 1):
                    equ_tot.append(z_vec_)

        elif prefilter == [1, 5]:
            if denticity == 1:
                axil_tot.append(z_vec_)
            else:
                axil_tot.append(z_vec_)
                for i in range(denticity - 1):
                    equ_tot.append(z_vec_)

        elif prefilter == [2, 4] or prefilter == [3, 3]:
            axil_tot.append(z_vec_)
            for i in range(denticity - 1):
                equ_tot.append(z_vec_)

        elif prefilter == [6]:
            for i in range(2):
                axil_tot.append(z_vec_)
            for i in range(4):
                equ_tot.append(z_vec_)

    return axil_tot, equ_tot,prefilter


def Gen_lig(smiles,model_lfs,step_size=0.005):
    idx = 0 
    while idx < 100:
        z_tree_mean,z_mol_mean,z_tree_log_var,z_mol_log_var = model_lfs.get_vector(smiles)
        epsilon_tree = create_var(torch.randn_like(z_tree_mean))
        epsilon_mol = create_var(torch.randn_like(z_mol_mean))
        z_tree_mean_new = z_tree_mean + torch.exp(z_tree_log_var / 2) * epsilon_tree * step_size
        z_mol_mean_new = z_mol_mean + torch.exp(z_mol_log_var / 2) * epsilon_mol * step_size
        smi = model_lfs.decode(z_tree_mean_new, z_mol_mean_new, prob_decode=False)
        smi = checksmile(smi)
        try:
            tree_batch = [MolTree(smi)]
            _, jtenc_holder, mpn_holder = tensorize(tree_batch, model_lfs.vocab, assm=False)
            tree_vecs, _, mol_vecs = model_lfs.encode(jtenc_holder, mpn_holder)
            z_tree_, z_mol_ = model_lfs.T_mean(tree_vecs), model_lfs.G_mean(mol_vecs)
            z_vecs_ = torch.cat((z_tree_,z_mol_),dim=1)
            denticity_predict_check = model_lfs.denticity_NN(z_vecs_)
            _, denticity_predict_check = torch.max(denticity_predict_check,1)
            denticity_predict_check = denticity_predict_check + 1
            denticity_predict_check = denticity_predict_check.item()
        except:
            pass

def Arrange_mutation(ligs_pair, ligs_denticity, selected_ligand, smi_new):
    if len(ligs_denticity) == 1:
        denticity = ligs_denticity[selected_ligand]
        
        if denticity == 1:
            # first orientation
            axial_equ_pair = {}
            axial = [selected_ligand, smi_new]
            equ = [selected_ligand] * 4
            
            axial_equ_pair[tuple(axial)] = [equ]
            # 2th orientation
            axial = [selected_ligand] * 2
            equ = [smi_new]*3 + [selected_ligand]
        
            axial_equ_pair[tuple(axial)] = [equ]
            
            return axial_equ_pair
        
        elif denticity == 2:
            axial_equ_pair = {}
            axial = [selected_ligand, smi_new]
            equ = [selected_ligand] * 3 + smi_new
            return axial_equ_pair
        
        elif denticity == 3:
            axial_equ_pair = {}
            axial = [selected_ligand, smi_new]
            
            

def Find_coordinates(obmol):
    coordinates = []
    metal_coord = []
    obConversion = ob.OBConversion()
    for atom in ob.OBMolAtomIter(obmol):
        if atom.IsMetal():
            db = defaultdict(list)
            idx_dict = defaultdict(list)
            denticity_dict = defaultdict()
            metal_coord.append([atom.GetX(), atom.GetY(), atom.GetZ()])
            metal_bond = 0
            for bond in ob.OBAtomBondIter(atom):
                bond_atom = bond.GetNbrAtom(atom)
                metal_bond += 1
                # Find the atom coordinates that bond with a metal atom
                if not bond_atom.IsMetal():
                    coordinates.append([bond_atom.GetX(),bond_atom.GetY(),bond_atom.GetZ()])
            if metal_bond == 6:
                obmol.DeleteAtom(atom)
                # Find ligands smiles
                for idx,component in enumerate(obmol.Separate()):
                    count = 0 
                    obConversion.SetOutFormat('smi')
                    smile_i = obConversion.WriteString(component).split('\t\n')[0]
                    try:
                        mol = Chem.MolFromSmiles(smile_i)           
                        smile = Chem.MolToSmiles(mol,canonical=True)
                        idx_dict[idx] = smile
                    except:
                        pass
                    for lig_atom in ob.OBMolAtomIter(component):
                        coord_x = lig_atom.x()
                        coord_y = lig_atom.y()
                        coord_z = lig_atom.z()
                        lig_coord = [coord_x, coord_y, coord_z]
                        lig_coord_ = np.array(lig_coord) - np.array(metal_coord)
                        if lig_coord in coordinates:
                            db[idx].extend(lig_coord_)
                            count += 1
                    denticity_dict[smile] = count
                batches = []
                # Find the axial-equatorial pair
                for idx_i, coord_i in db.items():
                    for coord_i_ in coord_i:
                        batch = []
                        axial = []
                        axial.append(idx_i)
                        equ = []
                        for idx_j, coord_j in db.items():
                            for coord_j_ in coord_j:
                                angle = math.degrees(angle_between(coord_i_,coord_j_))
                                if not np.all(coord_i_ == coord_j_):
                                    if angle > 145:
                                        axial.append(idx_j)
                                    else:
                                        equ.append(idx_j)
                        axial = sorted(axial)
                        equ = sorted(equ)
                        if len(axial) == 2 and len(equ) == 4:
                            batch.extend(axial)
                            batch.extend(equ)
                            batches.append(batch)
                            
                final_batches = [tuple(batch) for batch in batches]  # convert each sub-list to a tuple
                final_batches = list(set(final_batches))  # remove duplicates
                axial_equ_pair = {}
                for final_batch in final_batches:
                    axial = final_batch[:2]
                    equ = final_batch[2:]
                    if len(axial) == 2 and len(equ) == 4:
                        if tuple(axial) not in axial_equ_pair:
                            axial_equ_pair[tuple(axial)] = []
                        axial_equ_pair[tuple(axial)].extend(equ)
                return axial_equ_pair,denticity_dict,idx_dict
            else:
                raise ValueError('Structure might not be octahedral')
