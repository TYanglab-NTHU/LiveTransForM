import numpy as np
from openbabel import openbabel as ob
from collections import defaultdict
from rdkit import Chem
import math
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

def Find_coordinates(obmol):
    coordinates = []
    metal_coord = []
    obConversion = ob.OBConversion()
    for atom in ob.OBMolAtomIter(obmol):
        if atom.IsMetal():
            db = defaultdict(list)
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
                for component in obmol.Separate():
                    obConversion.SetOutFormat('smi')
                    smile_i = obConversion.WriteString(component).split('\t\n')[0]
                    try:
                        mol = Chem.MolFromSmiles(smile_i)           
                        smile = Chem.MolToSmiles(mol,canonical=True)
                    except:
                        pass
                    for lig_atom in ob.OBMolAtomIter(component):
                        coord_x = lig_atom.x()
                        coord_y = lig_atom.y()
                        coord_z = lig_atom.z()
                        lig_coord = [coord_x, coord_y, coord_z]
                        lig_coord_ = np.array(lig_coord) - np.array(metal_coord)
                        if lig_coord in coordinates:
                            db[smile].extend(lig_coord_)
                batches = []
                # Find the axial-equatorial pair
                for smi_i, coord_i in db.items():
                    for coord_i_ in coord_i:
                        batch = []
                        axial = []
                        axial.append(smi_i)
                        equ = []
                        for smi_j, coord_j in db.items():
                            for coord_j_ in coord_j:
                                angle = math.degrees(angle_between(coord_i_,coord_j_))
                                if not np.all(coord_i_ == coord_j_):
                                    if angle > 145:
                                        axial.append(smi_j)
                                    else:
                                        equ.append(smi_j)
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
                        axial_equ_pair[tuple(axial)].append(equ)
                        
                return axial_equ_pair
            else:
                raise ValueError('Structure might not be octahedral')
