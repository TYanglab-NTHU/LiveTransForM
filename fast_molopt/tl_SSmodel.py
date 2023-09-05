import torch,glob
import torch.nn as nn
import torch.optim as optim
import random,itertools
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import sys, os,os,json,csv,pickle
sys.path.append('../')
from fast_jtnn import *
from fast_jtnn.jtprop_vae import JTPropVAE
import pandas as pd
import numpy as np
from tqdm import tqdm
from optparse import OptionParser
from tqdm import tqdm
from rdkit import Chem
from sklearn.model_selection import train_test_split
# from ligand_class import ligands_pair

class ligands_pair(object):
    def __init__(self):
        self.axial = []
        self.equ = []
        self.filepath = ''
        self.refcode = ''
        
    def recover_axil(self,axils):
        for axil in axils:
            self.axial.append(axil)
    def recover_equ(self,equs):
        for equ in equs:
            self.equ.append(equ)

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
        self.zeff_dict = {'Fe0':3.75, 'Fe1':4.1, 'Fe2':6.25, 'Fe3':6.6,
            'Co0':3.9, 'Co1':4.25, 'Co2':6.9, 'Co3':7.25,
            'Mn0':3.1, 'Mn1':5.25, 'Mn2':5.6, 'Mn3':5.95, 'Mn4':6.3
            }
        self.ionic_radi = {'Fe0':140, 'Fe2':75, 'Fe3':60,
            'Co0':152,'Co2':79,'Co3':68.5,
            'Mn0':161,'Mn2':81,'Mn3':72
            }
        self.ionization_energy = {'Fe0':762.47,'Fe2':2957.4,'Fe3':5298,
            'Mn0':717.28,'Mn2':3248.5,'Mn3':4941,
            'Co0':760.40,'Co2':3232.3,'Co3':4947
            }
        
        self.zeff_dict = normalize_dict(self.zeff_dict)
        self.ionic_radi = normalize_dict(self.ionic_radi)
        self.ionization_energy = normalize_dict(self.ionization_energy)
        
        
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
    parser.add_option("-t", "--train", dest="vec_path", default='/work/scorej41075/JTVAE_horovod/JACS_no_label/latent/vecs_epoch_83')
    parser.add_option('-v', dest='vocab',default='/home/scorej41075/program/SS_model/data/vocab_add_hydrogen.csv')
    parser.add_option("-s", "--ss_label", dest="spin_label", default='/home/scorej41075/program/SS_model/data/SS_refcode/6-coord-M_ss_charge_v2.json')
    parser.add_option("-r", "--ref_pickle", dest="ref_pickle", default='/home/scorej41075/program/SS_model/data/SS_refcode/ref_ligs_pair_revise.pkl')
    parser.add_option("-o", "--output", dest="save_dir", default='/work/scorej41075/JTVAE_horovod/SS_model_model')
    parser.add_option("-d", "--dropout", dest="dropout", default=0.5)

    parser.add_option("-b", "--batch", dest="batch_size", default=128)
    parser.add_option("-w", "--hidden", dest="hidden_size", default=56)
    parser.add_option("-l", "--latent", dest="latent_size", default=28)
    parser.add_option("-q", "--lr", dest="lr", default=1e-3)

    opts,args = parser.parse_args()
    batch_size = opts.batch_size
    vec_path = opts.vec_path
    save_dir = opts.save_dir
    dropout = opts.dropout
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    vocab = opts.vocab
    latent_size = int(opts.latent_size)
    hidden_size = int(opts.hidden_size)
    spin_label = opts.spin_label
    with open(opts.ref_pickle,'rb') as f1:
        ref_pickle = pickle.load(f1)
    
    datau = datautil(vec_path,spin_label)
    batches = datau.batch_dataset(ref_pickle)
    batches = [datau.batch_loader(batch) for batch in batches]

    batches = list(filter(None,batches))
    split_ratio = 0.8
    trainset_temp, testset_temp = train_test_split(batches, train_size=split_ratio, random_state=55)

    # Further split the temporary dataset into training and validation
    split_ratio_train = 7 / 8  # 7/8 of the temporary dataset for training
    trainset, valset = train_test_split(trainset_temp, train_size=split_ratio_train, random_state=55)
    trainset_ = trainset
    # Use the remaining part of the temporary dataset as the final test set
    testset = testset_temp

    # trainset,testset = train_test_split(batches,train_size=split_ratio, random_state=55)
    # trainset_,testset_ = train_test_split(batches,train_size=split_ratio, random_state=55)
    hs_label_train,hs_label_test = [],[]
    ls_label_train,ls_label_test = [],[]
    sco_label_train,sco_label_test = [],[]
    sco_label_select = []
    for i in trainset:
        if i.ss_label.item() == 0:
            ls_label_train.append(i)
        elif i.ss_label.item() == 2:
            sco_label_train.append(i)
        elif i.ss_label.item() == 1:
            hs_label_train.append(i)
    for i in testset:
        if i.ss_label.item() == 0:
            ls_label_test.append(i)
        elif i.ss_label.item() == 2:
            sco_label_test.append(i)
        elif i.ss_label.item() == 1:
            hs_label_test.append(i)
    
    
    same_ligset = open('/home/scorej41075/program/SS_model/fast_molopt/same_ligset_ref.csv','r').read().splitlines()
    same_lig_train, same_lig_test = [], []
    same_lig_train_hs, same_lig_train_ls, same_lig_train_sco = [], [], []
    same_lig_test_hs, same_lig_test_ls, same_lig_test_sco = [], [], []

    for i in trainset_:
        if i.ref in same_ligset:
            same_lig_train.append(i)
            if i.ss_label.item() == 0:
                same_lig_train_ls.append(i)
            elif i.ss_label.item() == 2:
                same_lig_train_sco.append(i)
            elif i.ss_label.item() == 1:
                same_lig_train_hs.append(i)    
    for j in testset:
        if j.ref in same_ligset:
            same_lig_test.append(j)
            if j.ss_label.item() == 0:
                same_lig_test_ls.append(j)
            elif j.ss_label.item() == 2:
                same_lig_test_sco.append(j)
            elif j.ss_label.item() == 1:
                same_lig_test_hs.append(j)    
            
            
    # print('same_lig_train: hs_count %s, ls_count %s sco_count %s' %(len(same_lig_train_hs),len(same_lig_train_ls),len(same_lig_train_sco)))
    # print('same_lig_test: hs_count %s, ls_count %s sco_count %s' %(len(same_lig_test_hs),len(same_lig_test_ls),len(same_lig_test_sco)))

    print('train: hs_count %s, ls_count %s sco_count %s' %(len(hs_label_train),len(ls_label_train),len(sco_label_train)))
    print('test: hs_count %s, ls_count %s sco_count %s' %(len(hs_label_test),len(ls_label_test),len(sco_label_test)))

    ls_label_select = random.sample(ls_label_train,len(hs_label_train)-len(ls_label_train))
    # repetitions = len(hs_label_train) // len(sco_label_train)

    # Repeat sco_label_train to match the length of hs_label_train
    # repeated_sco_label_train = list(itertools.islice(itertools.cycle(sco_label_train), len(hs_label_train) - len(sco_label_train)))


    trainset.extend(ls_label_select)
    # trainset.extend(repeated_sco_label_train)
            
    ## SS_model init
    # training model
    MAX_EPOCH = 1000
    PRINT_ITER = 100
    learning_rate_iters = [10]
    softmaxfun = nn.Softmax(dim=0)

    for learning_rate_iter in learning_rate_iters:
    ## obtain probablity output
        ligand_size = 12
        model_prop = PropNN(latent_size, hidden_size,dropout)
        model_prop.cuda()
        print ("Model #Params: %dK" % (sum([x.nelement() for x in model_prop.parameters()]) / 1000,))
        lr = float(opts.lr)
        optimizer = optim.Adam(model_prop.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
        best_test_loss = np.inf
        current_patience = 0
        model_path = '/work/scorej41075/JTVAE_horovod/SS_model_model/ratio-7-1-2_wo_sco_nolabel/batchsize_128_lr_0.001_dropout_50_lr-iter_10/model.epoch-100'
        dict_buffer = torch.load(model_path, map_location='cuda:0')
        model_prop = PropNN(latent_size, hidden_size,0.2)
        model_prop.load_state_dict(dict_buffer)
        model_prop.cuda()
        save_dir_ = save_dir + '/ratio-7-1-2_wo_sco_nolabel/batchsize_128_lr_0.001_dropout_50_lr-iter_%s' %(learning_rate_iter)
        if not os.path.isdir(save_dir_):
            os.makedirs(save_dir_)
        for epoch in tqdm(range(MAX_EPOCH)):
            model_prop.train()
            epoch += 101
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
                nn.utils.clip_grad_norm_(model_prop.parameters(), 50)
                optimizer.step()
                    # print("Loss: %.5f" % (loss))
            with open(os.path.join(save_dir_,'tl_loss_train'),'a') as f1:
                loss_ = loss_ / step
                s = '%d,%5f' %(epoch,loss_) 
                f1.write(s+'\n')
                sys.stdout.flush()
            if epoch % learning_rate_iter == 0:
                scheduler.step()
            torch.save(model_prop.state_dict(), save_dir_ + "/model.epoch-" + str(epoch))
            ## test model
            model_prop.eval()
            count, count_same_lig,count_same_lig_hs,count_same_lig_ls,count_same_lig_sco = 0, 0, 0, 0, 0
            correct,correct_same_lig,correct_same_lig_hs,correct_same_lig_ls,correct_same_lig_sco = 0, 0,0,0,0
            loss_ = 0
            # early stopping
            min_delta = 0.0005
            patience = 50
            confusion_matrix_true = []
            confusion_matrix_predict = []
            for test_batch in valset:
                axial_vecs = test_batch.axial_vecs
                equ_vecs = test_batch.equ_vecs
                metal_inform = torch.stack((test_batch.zeff.to(torch.float32), test_batch.ionic_radi.to(torch.float32), test_batch.ionization_energy.to(torch.float32)))
                ss_batch = test_batch.ss_label.cuda()
                complex_predict = model_prop.ligand_NN(torch.cat((axial_vecs.cuda(),equ_vecs.cuda(),metal_inform.cuda()),dim=0))
                # complex_predict = model_prop.SS_NN(torch.cat((ligands_vecs,metal_inform.cuda()),dim=0))     
                loss = model_prop.SS_loss(complex_predict,ss_batch)
                complex_predict = softmaxfun(complex_predict)
                _, complex_predict = torch.max(complex_predict,0)
                ss_predict = complex_predict.item()
                ss_true = ss_batch.item()
                count += 1
                loss_ += loss.item()
                # if test_batch.ref in same_ligset:
                #     confusion_matrix_true.append(ss_true)
                #     confusion_matrix_predict.append(ss_predict)
                #     count_same_lig += 1  
                #     if ss_predict == ss_true:
                #         correct_same_lig += 1   
            
                if ss_predict == ss_true:
                    correct += 1
                    
            # confusion_matrix1 = confusion_matrix(confusion_matrix_true, confusion_matrix_predict)
            # class_accuracy = np.diag(confusion_matrix1) / confusion_matrix1.sum(axis=1)
            # precision = np.diag(confusion_matrix1) / np.sum(confusion_matrix1, axis=0)
            # recall = np.diag(confusion_matrix1) / np.sum(confusion_matrix1, axis=1)
            # f1_score = 2 * (precision * recall) / (precision + recall)
            # final = list(precision) + list(recall) + list(f1_score)
            # acc_same_lig = correct_same_lig / count_same_lig
        
            acc = correct/count
            with open(os.path.join(save_dir_,'val_acc.csv'),'a') as fwrite:
                s = '%d,%5f' %(epoch,acc) 
                fwrite.write(s+'\n')
                
            with open(os.path.join(save_dir_,'tl_loss_val'),'a') as f1:
                loss_ = loss_ / count
                s = '%d,%5f' %(epoch,loss_) 
                f1.write(s+'\n')
            #     fwrite.write(s+'\n')
    #         if loss_ + min_delta < best_test_loss :
    #             best_test_loss = loss_
    #             current_patience = 0
    #         else:
    #             current_patience += 1
    #             if current_patience >= patience:
    #                 print(f"Early stopping at epoch {epoch+1}.")
    #                 break
    #         # with open(os.path.join(save_dir_,'test_acc_samelig.csv'),'a') as fwrite:
    #         #     s = '%d,%5f,%s' %(epoch,acc_same_lig,str(class_accuracy.tolist()).strip('[]')) 
    #         #     fwrite.write(s+'\n')
    #         # with open(os.path.join(save_dir_,'same_lig_confusion_matrix_test.csv'),'a') as f1:
    #         #     writer = csv.writer(f1)
    #         #     writer.writerow(final)
    save_dir_ = '/work/scorej41075/JTVAE_horovod/SS_model_model/ratio-7-1-2_wo_sco_nolabel/batchsize_128_lr_0.001_dropout_50_lr-iter_10'
    files = glob.glob('/work/scorej41075/JTVAE_horovod/SS_model_model/ratio-7-1-2_wo_sco_nolabel/batchsize_128_lr_0.001_dropout_50_lr-iter_10/model.epoch*')
    def get_epoch_number(file_name):
        return int(file_name.split('-')[-1])

    # Sort the file names using the custom sorting key
    sorted_files = sorted(files, key=get_epoch_number)
    for model_path in sorted_files:
        # model_path = '/work/scorej41075/JTVAE_horovod/SS_model_model/ratio-7-1-2_wo_sco_v2/batchsize_128_lr_0.001_dropout_50_lr-iter_10/model.epoch-389'
        epoch = model_path.split('-')[-1]
        dict_buffer = torch.load(model_path, map_location='cuda:0')
        model_prop = PropNN(latent_size, hidden_size,0.2)
        model_prop.load_state_dict(dict_buffer)
        model_prop.cuda()
        model_prop.eval() 
        count, count_same_lig,count_same_lig_hs,count_same_lig_ls,count_same_lig_sco = 0, 0, 0, 0, 0
        correct,correct_same_lig,correct_same_lig_hs,correct_same_lig_ls,correct_same_lig_sco = 0, 0, 0, 0, 0
        confusion_matrix_true = []
        confusion_matrix_predict = []
        for train_batch in trainset_:
            axial_vecs = train_batch.axial_vecs
            equ_vecs = train_batch.equ_vecs
            metal_inform = torch.stack((train_batch.zeff.to(torch.float32), train_batch.ionic_radi.to(torch.float32), train_batch.ionization_energy.to(torch.float32)))
            ss_batch = train_batch.ss_label
            complex_predict = model_prop.ligand_NN(torch.cat((axial_vecs.cuda(),equ_vecs.cuda(),metal_inform.cuda()),dim=0))
            # complex_predict = model_prop.SS_NN(torch.cat((ligands_vecs,metal_inform.cuda()),dim=0))     
            complex_predict = softmaxfun(complex_predict)
            _, complex_predict = torch.max(complex_predict,0)
            ss_predict = complex_predict.item()
            ss_true = ss_batch.item()
            count += 1
            # if train_batch.ref in same_ligset:
            confusion_matrix_true.append(ss_true)
            confusion_matrix_predict.append(ss_predict)
            count_same_lig += 1   
            if ss_predict == ss_true:
                correct_same_lig += 1   
                        
            if ss_predict == ss_true:
                correct += 1
        acc = correct/count

        acc_same_lig = correct_same_lig / count_same_lig
        confusion_matrix1 = confusion_matrix(confusion_matrix_true, confusion_matrix_predict)
        class_accuracy = np.diag(confusion_matrix1) / (confusion_matrix1.sum(axis=1))
        precision = np.diag(confusion_matrix1) / np.sum(confusion_matrix1, axis=0)
        recall = np.diag(confusion_matrix1) / np.sum(confusion_matrix1, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        final = list(precision) + list(recall) + list(f1_score)
        with open(os.path.join(save_dir_,'confusion_matrix_train.csv'),'a') as f1:
            writer = csv.writer(f1)
            writer.writerow(final)
                
        with open(os.path.join(save_dir_,'train_acc.csv'),'a') as fwrite:
            s = '%s,%5f' %(epoch,acc) 
            fwrite.write(s+'\n')
        with open(os.path.join(save_dir_,'train_acc_samelig.csv'),'a') as fwrite:
            s = '%s,%5f,%s' %(epoch,acc_same_lig,str(class_accuracy.tolist()).strip('[]')) 
            fwrite.write(s+'\n')
            
        # for train_batch in trainset_:
        #     axial_vecs = train_batch.axial_vecs
        #     equ_vecs = train_batch.equ_vecs
        #     metal_inform = torch.stack((train_batch.zeff.to(torch.float32), train_batch.ionic_radi.to(torch.float32), train_batch.ionization_energy.to(torch.float32)))
        #     ss_batch = train_batch.ss_label
        #     complex_predict = model_prop.ligand_NN(torch.cat((axial_vecs.cuda(),equ_vecs.cuda(),metal_inform.cuda()),dim=0))
        #     # complex_predict = model_prop.SS_NN(torch.cat((ligands_vecs,metal_inform.cuda()),dim=0)) 
        #     complex_predict = softmaxfun(complex_predict)
        #     _, complex_predict = torch.max(complex_predict,0)
        #     ss_predict = complex_predict.item()
        #     ss_true = ss_batch.item()
        #     count += 1   
        #     if ss_predict == ss_true:
        #         correct += 1
        # acc = correct/count
        # with open(os.path.join(save_dir_,'train_acc_tuning.csv'),'a') as fwrite:
        #     s = '%d,%5f' %(epoch,acc) 
        #     fwrite.write(s+'\n')
        # if loss_ + min_delta < best_test_loss :
        #     best_test_loss = loss_
        #     current_patience = 0
        # else:
        #     current_patience += 1
        #     if current_patience >= patience:
        #         print(f"Early stopping at epoch {epoch+1}.")
        #         break
        count = 0
        correct = 0
        confusion_matrix_true = []
        confusion_matrix_predict = []
        for test_batch in testset:
            axial_vecs = test_batch.axial_vecs
            equ_vecs = test_batch.equ_vecs
            metal_inform = torch.stack((test_batch.zeff.to(torch.float32), test_batch.ionic_radi.to(torch.float32), test_batch.ionization_energy.to(torch.float32)))
            ss_batch = test_batch.ss_label
            complex_predict = model_prop.ligand_NN(torch.cat((axial_vecs.cuda(),equ_vecs.cuda(),metal_inform.cuda()),dim=0))
            # complex_predict = model_prop.SS_NN(torch.cat((ligands_vecs,metal_inform.cuda()),dim=0))     
            complex_predict = softmaxfun(complex_predict)
            _, complex_predict = torch.max(complex_predict,0)
            ss_predict = complex_predict.item()
            ss_true = ss_batch.item()
            count += 1
            # if test_batch.ref in same_ligset:
            confusion_matrix_true.append(ss_true)
            confusion_matrix_predict.append(ss_predict)
            #     count_same_lig += 1   
            #     if ss_predict == ss_true:
            #         correct_same_lig += 1   
            if ss_predict == ss_true:
                correct += 1
            
            
        acc = correct/count
        with open(os.path.join(save_dir_,'test_acc.csv'),'a') as fwrite:
            s = '%s,%5f' %(epoch,acc) 
            fwrite.write(s+'\n')

        acc_same_lig = correct_same_lig / count_same_lig
        confusion_matrix1 = confusion_matrix(confusion_matrix_true, confusion_matrix_predict)
        class_accuracy = np.diag(confusion_matrix1) / (confusion_matrix1.sum(axis=1))
        precision = np.diag(confusion_matrix1) / np.sum(confusion_matrix1, axis=0)
        recall = np.diag(confusion_matrix1) / np.sum(confusion_matrix1, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        final = list(precision) + list(recall) + list(f1_score)
        with open(os.path.join(save_dir_,'confusion_matrix_test.csv'),'a') as f1:
            writer = csv.writer(f1)
            writer.writerow(final)

        with open(os.path.join(save_dir_,'test_acc_samelig.csv'),'a') as fwrite:
            s = '%s,%5f,%s' %(epoch,acc_same_lig,str(class_accuracy.tolist()).strip('[]')) 
            fwrite.write(s+'\n')
        # if loss_ + min_delta < best_test_loss :
        #     best_test_loss = loss_
        #     current_patience = 0
        # else:
        #     current_patience += 1
        #     if current_patience >= patience:
        #         print(f"Early stopping at epoch {epoch+1}.")
        #         break
        
        # results = []
        # results_same = []
        # for test_batch in testset_:
        #     axial_vecs = test_batch.axial_vecs
        #     equ_vecs = test_batch.equ_vecs
        #     metal_inform = torch.stack((test_batch.zeff.to(torch.float32), test_batch.ionic_radi.to(torch.float32), test_batch.ionization_energy.to(torch.float32)))
        #     ss_batch = test_batch.ss_label
        #     complex_predict = model_prop.ligand_NN(torch.cat((axial_vecs.cuda(),equ_vecs.cuda(),metal_inform.cuda()),dim=0))
        #     complex_predict = softmaxfun(complex_predict)
        #     _, complex_predict = torch.max(complex_predict,0)
        #     ss_predict = complex_predict.item()
        #     ss_true = ss_batch.item()
        #     results.append([ss_predict,ss_true])
        #     if test_batch.ref in same_ligset:
        #         results_same.append([ss_predict,ss_true])

            
        # with open('/work/scorej41075/JTVAE_horovod/SS_model/TL_model_label_v2_3_lr_20_7:3_randomstate_55/confusion_metrix-48_samelig-test.csv','w') as f1:
        #     writer = csv.writer(f1)
        #     writer.writerows(results)

        # # results = []

        # for train_batch in trainset:
        #     axial_vecs = train_batch.axial_vecs
        #     equ_vecs = train_batch.equ_vecs
        #     metal_inform = torch.stack((train_batch.zeff.to(torch.float32), train_batch.ionic_radi.to(torch.float32), train_batch.ionization_energy.to(torch.float32)))
        #     ss_batch = train_batch.ss_label
        #     complex_predict = model_prop.ligand_NN(torch.cat((axial_vecs.cuda(),equ_vecs.cuda(),metal_inform.cuda()),dim=0))
        #     complex_predict = softmaxfun(complex_predict)
        #     _, complex_predict = torch.max(complex_predict,0)
        #     ss_predict = complex_predict.item()
        #     ss_true = ss_batch.item()
        #     results.append([ss_predict,ss_true])
        # with open('/work/scorej41075/JTVAE_horovod/SS_model/TL_model_version3/confusion_metrix-451-train.csv','w') as f1:
        #     writer = csv.writer(f1)
        #     writer.writerows(results)
            
    # ## LFS_model init by default
    # vocab = [x.strip("\r\n ") for x in open(vocab)]
    # vocab = Vocab(vocab)
    # model_lfs = JTPropVAE(vocab, int(450), int(56),int(2),int(20), int(3))
    # model_path = os.path.join('/home','scorej41075','program','SS_model','model_version1','model.epoch-83')
    # dict_buffer = torch.load(model_path, map_location='cuda:0')
    # model_lfs.load_state_dict(dict_buffer)
    # model_lfs.cuda()
    # model_lfs.eval()
    # idx = 76
    # model_path = save_dir + '/model.epoch-%s' %idx
    # model_path = '/work/scorej41075/JTVAE_horovod/SS_model/TL_model_ratio7:3_label_v2/model.epoch-874'
    # dict_buffer = torch.load(model_path, map_location='cuda:0')
    # model_prop = PropNN(latent_size, hidden_size,0,0.2)
    # model_prop.load_state_dict(dict_buffer)
    # model_prop.cuda()
    # model_prop.eval()
    # train_acc = []
    # lfs_ref = []
    # for train_batch in trainset:
    #     axial_vecs = train_batch.axial_vecs
    #     equ_vecs = train_batch.equ_vecs
    #     zeff = train_batch.zeff.unsqueeze(0)
    #     ligs_vecs = train_batch.ligs_vecs
    #     ss_label = train_batch.ss_label
    #     ref = train_batch.ref
    #     metal = train_batch.metal
    #     lfs = 0 
    #     for lig_vec in ligs_vecs:
    #         lfs_pred = model_lfs.propNN(lig_vec.cuda())
    #         lfs_pred = torch.clamp(lfs_pred[0],min=0,max=1)
    #         lfs += lfs_pred.item()              
    #     complex_predict = model_prop.ss_NN(torch.cat((axial_vecs.cuda(),equ_vecs.cuda(),zeff.unsqueeze(0).cuda()),dim=0))
    #     complex_predict = softmaxfun(complex_predict)
    #     _, complex_predict = torch.max(complex_predict,0)
    #     ss_predict = complex_predict.item()
    #     ss_true = ss_label.item()
    #     lfs_ref.append([ref,metal,ss_predict,ss_true,lfs])
    #     correct += 1
    # acc = correct/count
    # train_acc.append([acc,idx])
    # with open('/home/scorej41075/program/SS_model/TL_model_SS_unique/lfs_acc.csv','w') as f1:
    #     writer = csv.writer(f1)
    #     writer.writerows(lfs_ref)
        
        
    # # Initialize variables
    # smis = []                  # List of SMILES strings
    # denticity_origin = []      # List of predicted denticiy values for each iteration
    # denticity_check = []       # List of actual denticiy values for each iteration
    # prob_decode = False        # Flag for decoding probabilistic output
    # denticity_count = 0        # Counter for number of denticiy values predicted
    # lfs_pred_tot = 0           # Total predicted property value for all predicted denticiy values
    # z_vecs_tot = []            # List of z vectors for each iteration
    # flag = True                # Flag to continue iteration
    # decode_count = 0           # Counter for number of decoding iterations
    # data = set()               # Set of unique SMILES strings generated

    # while flag:
    #     # Generate z vectors and predict denticiy values and property values
    #     while denticity_count != 6:   # Loop until 6 denticiy values are predicted
    #         z_tree = torch.randn(1,28).cuda()
    #         z_mol = torch.randn(1,28).cuda()
    #         z_vecs = torch.cat((z_tree,z_mol),dim=1)
    #         prop_pred = model_lfs.propNN(z_vecs.cuda())
    #         denticity_output = model_lfs.denticity_NN(z_vecs)
    #         _, denticity_predict = torch.max(denticity_output,1)
    #         lfs_pred = torch.clamp(prop_pred.squeeze(0)[0],min=0,max=1)
    #         denticity_predict = denticity_predict + 1
    #         denticity_origin.append(denticity_predict.item())
    #         denticity_count += denticity_predict.item()
    #         for i in range(s):
    #             lfs_pred_tot += lfs_pred.item()
    #         if denticity_count != 6:   # Reset variables if not all 6 denticiy values are predicted
    #             denticity_count = 0
    #             lfs_pred_tot = 0
    #             z_vecs_tot = []
    #             denticity_origin = []
                
    #     # Add final z vector and predicted denticiy value to list
    #     z_vecs_tot.append([z_vecs,denticity_predict.item()])

    #     # Decode z vectors to SMILES strings and store in list
    #     smis = [model_lfs.decode(*torch.split(z[0],28,dim=1), prob_decode=False) for z in z_vecs_tot]

    #     # Attempt to create MolTree objects from SMILES strings and calculate denticiy values
    #     try:
    #         tree_batch = [MolTree(smi) for smi in smis]
    #         _, jtenc_holder, mpn_holder = datautils.tensorize(tree_batch, model_lfs.vocab, assm=False)
    #         tree_vecs, _, mol_vecs = model_lfs.encode(jtenc_holder, mpn_holder)
    #         z_tree_means = model_lfs.T_mean(tree_vecs)
    #         z_mol_means = model_lfs.G_mean(mol_vecs)
    #         for z_tree_vec,z_mol_vec in zip(z_tree_means,z_mol_means):
    #             denticity = model_lfs.denticity_NN(torch.cat((z_tree_vec,z_mol_vec),dim=0))
    #             denticity_predict = softmaxfun(denticity)
    #             _, denticity_predict = torch.max(denticity_predict,0)
    #             denticity_check.append(denticity_predict.item() + 1)
    #     except:
    #         pass
    #     if denticity_check == denticity_origin:
    #         if smis != []:
    #             data.add(smis[0])
    #         if len(data) > 500:
    #             flag = False
    #     # print(denticity_origin)
    #     # print(denticity_check)
    #     if denticity_check == denticity_origin:
    #         if smis != []:
    #             data.add(smis[0])
    #         if len(data) > 500:
    #             flag = False
    # ## combination [1,1,1,1,1,1],[1,1,1,1,2],[1,1,1,3],[1,1,4],[1,2,3],[1,5],[2,4],[3,3],[6]
    # axial_tot,equ_tot,combintaion = sort_z_vecs(z_vecs_tot)
    # axial_batch_tensor = torch.stack(axial_tot, dim=0).sum(dim=0)
    # equ_batch_tensor = torch.stack(equ_tot, dim=0).sum(dim=0)
    # complex_predict = model_prop.ss_NN(torch.cat((axial_batch_tensor.cuda(),equ_batch_tensor.cuda(),torch.Tensor([6.25]).cuda()),dim=0))
    # softmaxfun = nn.Softmax(dim=0)
    # complex_predict = softmaxfun(complex_predict)
    # _, complex_predict = torch.max(complex_predict,0)
    # zeff_dict = {'Fe0':3.75, 'Fe1':4.1, 'Fe2':6.25, 'Fe3':6.6,
    #         'Co0':3.9, 'Co1':4.25, 'Co2':6.9, 'Co3':7.25,
    #         'Mn0':3.1, 'Mn1':5.25, 'Mn2':5.6, 'Mn3':5.95, 'Mn4':6.3
    #         }
    # print(smis)
    # print(denticity_check)
    # print(complex_predict.item(),lfs_pred_tot)  
    # print(decode_count)
