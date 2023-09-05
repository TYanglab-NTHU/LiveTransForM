import torch
import sys,json
import time
import timeout_decorator
from torch.utils.data import Dataset, DataLoader
from .mol_tree import *
import numpy as np
from .jtnn_enc import JTNNEncoder
from .mpn import MPN
from .jtmpn import JTMPN
import pickle as pickle
import os, random
import torch.utils.data.distributed

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab,prop_path ,batch_size, epoch=0, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [i for i in os.listdir(self.data_folder) if '.pkl' in i]
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.batch = []
        self.batch_size = batch_size
        self.epoch = epoch
        # prop
        self.prop = json.load(open(prop_path,'r'))
        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
                
        for i, fn in enumerate(self.data_files):
            # print(fn)
            f = os.path.join(self.data_folder, fn)
            with open(f, 'rb') as fin:
                fin.seek(0)
                data = pickle.load(fin)
            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch
            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) % self.batch_size != 0:
                batch_to_add = batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0], num_workers=self.num_workers)
                
            if dataloader:
                try:
                    for b in dataloader:
                        moltrees = b[0]
                        prop_ss = []
                        for moltree in moltrees:
                            if int(self.prop[moltree.smiles]['tot']) >= 5:
                                prop_ss.append([float(self.prop[moltree.smiles]['hs']),float(self.prop[moltree.smiles]['SCS']),int(self.prop[moltree.smiles]['denticity'])])
                            else:
                                prop_ss.append([float('nan'),float(self.prop[moltree.smiles]['SCS']),int(self.prop[moltree.smiles]['denticity'])])
                        yield list(b) + [prop_ss]
                    del dataset, dataloader
                except Exception as e:
                    print(('%s failed' %(fn),e))
                
                del data


class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            return tensorize(self.data[idx], self.vocab, assm=self.assm)
        except:
            print(self.data[idx][0].smiles)
            return tensorize(self.data[idx-1], self.vocab, assm=self.assm)
            pass

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder
    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

           
    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1