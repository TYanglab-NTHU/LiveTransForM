import torch,glob
import torch.nn as nn
import torch.optim as optim
import random
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import sys, os,os,json,pickle
sys.path.append('../')
sys.path.append('./SS_utils')
from fast_jtnn import *
import pandas as pd
import numpy as np
from tqdm import tqdm
from optparse import OptionParser
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class PropNN(nn.Module):

    def __init__(self, latent_size, hidden_size,dropout):
        super(PropNN, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.ligand_NN = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + 3, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 2)
          )
        
        self.SS_loss = nn.CrossEntropyLoss(size_average=False)

    def forward(self, axial_vecs,equ_vecs, nuclear_label, ss_label):
        complex_predict = self.ligand_NN(torch.cat((axial_vecs,equ_vecs,nuclear_label),dim=1))
        loss = self.SS_loss(complex_predict,ss_label)
        return loss


class latent_vects(object):
    def __init__(self):
        self.axial_vecs = None
        self.equ_vecs = None
        self.ss_label = None
        self.ligs_vecs = None
        self.metal = None
        self.zeff = None
        self.ref = None
        self.ionic_radi = None
        self.ionization_energy = None
        
def normalize_dict(dictionary):
    values = np.array(list(dictionary.values()))
    mean = np.mean(values)
    std = np.std(values)
    normalized_dict = {key: (value - mean) / std for key, value in dictionary.items()}
    return normalized_dict

class datautil():
    def __init__(self,latent_path,ss_label):
        self.latent_path = latent_path
        self.df_latent = pd.read_csv(self.latent_path)
        with open(ss_label) as f:
            self.ss_label = json.load(f)
        self.ss_dict = {'ls':0.,'hs':1,'sco':2}
        self.zeff_dict = {'Fe2':0.7091855764024294, 'Fe3':0.9769693417660021,
            'Co2':1.2064982835062081, 'Co3':1.474282048869781,
            'Mn2':0.21187286929865048, 'Mn3':0.47965663466222397
            }
        self.ionic_radi = {'Fe2':-0.6286975914160016, 'Fe3':-1.0262347101099603,
            'Co2':-0.5226876930976125,'Co3':-0.8009636761833837,
            'Mn2':-0.4696827439384181,'Mn3':-0.7082050151547933
            }
        self.ionization_energy = {'Fe2':-0.01555472321511081,'Fe3':1.3070142633261608,
            'Mn2':0.1489329431875911,'Mn3':1.1052894371297863,
            'Co2':0.1397790435114532,'Co3':1.1086797703431708
            }
        
    def batch_dataset(self,ligss_file):
        batches = []
        for ligs_file in ligss_file:
            try:
                batches.append([ligs_file.axial,ligs_file.equ,ligs_file.refcode])
            except:
                pass
        return batches
    
    def batch_loader(self,batch):
        z_vecs = latent_vects()
        axial_ligs = batch[0]
        equ_ligs = batch[1]
        refcode = batch[2]
        axial_vecs = self.recover(axial_ligs)
        equ_vecs = self.recover(equ_ligs)
        if len(axial_vecs) == 2 and len(equ_vecs) == 4:  
            z_vecs.axial_vecs = self.get_vecs(axial_vecs)
            z_vecs.equ_vecs = self.get_vecs(equ_vecs)
            z_vecs.ref = refcode
            ## stack lig vecs in order to predict LFS 
            axial_vecs = torch.stack(axial_vecs,dim=0)
            equ_vecs = torch.stack(equ_vecs,dim=0)
            try:
                z_vecs.ligs_vecs = torch.cat((axial_vecs,equ_vecs),dim=0)
                z_vecs.ss_label, z_vecs.zeff,z_vecs.ionic_radi,z_vecs.ionization_energy,z_vecs.metal = self.get_ss_label(refcode)
                if z_vecs.axial_vecs != None and z_vecs.equ_vecs != None:
                    return z_vecs
                else:
                    return None
            except:
                return None
            
    def recover(self,vecs):
        vecs_batch = []
        for vec in vecs:
            vec_match = self.df_latent[self.df_latent.iloc[:,0] == vec]
            if len(vec_match) > 0:
                vecs_batch.append(torch.Tensor(vec_match.iloc[0,1:]))
        return vecs_batch
    
    def get_vecs(self,raw_vecs):
        vecs = torch.stack(raw_vecs,dim=0)
        vecs = vecs.sum(dim=0)
        return vecs
    def get_ss_label(self,refcode):
        ss = torch.tensor(self.ss_dict[list(self.ss_label[refcode].values())[0]]).long()
        zeff = torch.tensor(self.zeff_dict[list(self.ss_label[refcode].keys())[0]])
        ionic_radi = torch.tensor(self.ionic_radi[list(self.ss_label[refcode].keys())[0]])
        ionization_energy = torch.tensor(self.ionization_energy[list(self.ss_label[refcode].keys())[0]])
        metal = list(self.ss_label[refcode].keys())[0]
        return ss, zeff, ionic_radi, ionization_energy, metal
    
    def batch_loader_(self,batch):
        data = []
        axial_ligs = batch[0]
        equ_ligs = batch[1]
        refcode = batch[2]
        axial_vecs = self.recover(axial_ligs)
        equ_vecs = self.recover(equ_ligs)
        data.append([axial_vecs,equ_vecs,refcode])
        return data
            
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-v',"--vecs", dest="vec_path", default='../data/latent_vec.csv')
    parser.add_option("--ss_label", dest="spin_label", default='../data/TL_model_label.json')
    parser.add_option('-i',"--input", dest="input_pickle", default='../data/TL_model_data.pkl')
    parser.add_option("--output", dest="save_dir", default=True)
    parser.add_option("--dropout", dest="dropout", default=0.5)
    parser.add_option("--batch", dest="batch_size", default=128)
    parser.add_option("--hidden", dest="hidden_size", default=56)
    parser.add_option("--latent", dest="latent_size", default=28)
    parser.add_option("--lr_iter", dest="lr_iter", default=5)
    parser.add_option("--lr", dest="lr", default=1e-3)
    parser.add_option('--clip_norm', type=float, default=50.0)
    parser.add_option('--anneal_rate', type=float, default=0.9)

    opts,args = parser.parse_args()
    batch_size = opts.batch_size
    vec_path = opts.vec_path
    save_dir = opts.save_dir
    dropout = opts.dropout
    clip_norm = opts.clip_norm
    learning_rate_iter = opts.lr_iter
    anneal_rate = opts.anneal_rate
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    latent_size = int(opts.latent_size)
    hidden_size = int(opts.hidden_size)
    spin_label = opts.spin_label
    with open(opts.input_pickle,'rb') as f1:
        ref_pickle = pickle.load(f1)
    # Loading data
    datau = datautil(vec_path,spin_label)
    batches = datau.batch_dataset(ref_pickle)
    batches = [datau.batch_loader(batch) for batch in batches]
    batches = list(filter(None,batches))
    # Splitting data to train,validation,tests
    split_ratio = 0.8
    split_ratio_train = 7 / 8
    random_state_value = random.randint(0, 10000)
    trainset_temp, testset = train_test_split(batches, train_size=split_ratio, random_state=random_state_value)
    trainset, valset = train_test_split(trainset_temp, train_size=split_ratio_train, random_state=random_state_value)
        
    # training model
    MAX_EPOCH = 1000
    PRINT_ITER = 100
    model_prop = PropNN(latent_size, hidden_size,dropout)
    model_prop.cuda()
    print ("Model #Params: %dK" % (sum([x.nelement() for x in model_prop.parameters()]) / 1000,))
    lr = float(opts.lr)
    optimizer = optim.Adam(model_prop.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
    best_test_loss = np.inf
    current_patience = 0
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for epoch in tqdm(range(MAX_EPOCH)):
        model_prop.train()
        ## random sampling data
        random.shuffle(trainset)
        train_batches = [trainset[i : i + batch_size] for i in range(0, len(trainset),int(batch_size))]
        if len(train_batches[-1]) != int(batch_size):
            train_batches.pop()
        step = 0
        loss_ = 0
        for batch in train_batches:
            model_prop.zero_grad()
            axial_batch = torch.stack([comp.axial_vecs for comp in batch])
            equ_batch = torch.stack([comp.equ_vecs for comp in batch])
            metal_inform_batch = torch.stack([torch.stack([comp.zeff, comp.ionic_radi, comp.ionization_energy]).to(torch.float32) for comp in batch])
            ss_batch = torch.stack([comp.ss_label for comp in batch])
            loss = model_prop(axial_batch.cuda(),equ_batch.cuda(),metal_inform_batch.cuda(),ss_batch.cuda())
            step += 1
            loss.backward()
            loss = loss.item()
            loss_ += loss / batch_size
            nn.utils.clip_grad_norm_(model_prop.parameters(), clip_norm)
            optimizer.step()
        with open(os.path.join(save_dir,'tl_loss_train'),'a') as f1:
            loss_ = loss_ / step
            s = '%d,%5f' %(epoch,loss_) 
            f1.write(s+'\n')
            sys.stdout.flush()
        if epoch % learning_rate_iter == 0:
            scheduler.step()
        torch.save(model_prop.state_dict(), save_dir + "/model.epoch-" + str(epoch))
        # eval model performance
        model_prop.eval()
        count,correct = 0,0
        loss_ = 0
        confusion_matrix_true = []
        confusion_matrix_predict = []
        for val_batch in valset:
            axial_vecs = val_batch.axial_vecs
            equ_vecs = val_batch.equ_vecs
            metal_inform = torch.stack((val_batch.zeff.to(torch.float32), val_batch.ionic_radi.to(torch.float32), val_batch.ionization_energy.to(torch.float32)))
            ss_batch = val_batch.ss_label.cuda()
            complex_predict = model_prop.ligand_NN(torch.cat((axial_vecs.cuda(),equ_vecs.cuda(),metal_inform.cuda()),dim=0))
            loss = model_prop.SS_loss(complex_predict,ss_batch)
            _, complex_predict = torch.max(complex_predict,0)
            ss_predict = complex_predict.item()
            ss_true = ss_batch.item()
            count += 1
            loss_ += loss.item()
            if ss_predict == ss_true:
                correct += 1
                
            acc = correct/count
            with open(os.path.join(save_dir,'val_acc.csv'),'a') as fwrite:
                s = '%d,%5f' %(epoch,acc) 
                fwrite.write(s+'\n')
                
            with open(os.path.join(save_dir,'tl_loss_val'),'a') as f1:
                loss_ = loss_ / count
                s = '%d,%5f' %(epoch,loss_) 
                f1.write(s+'\n')