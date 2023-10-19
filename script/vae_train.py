import argparse, sys, pickle, os, rdkit, math, time
import numpy as np
from re import I
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
sys.path.append('../')
from fast_jtnn import *
from fast_jtnn.jtprop_vae import JTPropVAE

t = time.strftime('%Y%m%d%H%M%S')

def main_vae_train(args):
  
    train = args.train
    vocab = args.vocab
    prop_path = args.prop_path
    save_dir = args.save_dir
    load_epoch = args.load_epoch
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    latent_size = args.latent_size
    prop_size = args.prop_size
    depthT = args.depthT
    depthG = args.depthG
    lr = args.lr
    clip_norm = args.clip_norm
    beta = args.beta
    step_beta = args.step_beta
    max_beta = args.max_beta
    warmup = args.warmup
    epoch = args.epoch
    anneal_rate = args.anneal_rate
    anneal_iter = args.anneal_iter
    kl_anneal_iter = args.kl_anneal_iter
    print_iter = args.print_iter
    save_iter = args.save_iter

    vocab = [x.strip("\n") for x in open(vocab)] 
    vocab = Vocab(vocab)

    model = JTPropVAE(vocab, int(hidden_size), int(latent_size), int(prop_size), int(depthT), int(depthG))
    print(model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)
    
    if load_epoch > 0:
        model.load_state_dict(torch.load(save_dir + "/model.epoch-" + str(load_epoch)))

    print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
    scheduler.step()

    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = load_epoch
    beta = beta
    meters = np.zeros(7)
    nancount = 0        
    for epoch in range(epoch):
        loader = MolTreeFolder(train, vocab,prop_path, batch_size, epoch=epoch, num_workers=4)
        # print('epoch number is %s' %(epoch))
        for batch in loader:
            batch_new0 = batch[0]
            batch_new1 = batch[1]
            batch_new2 = batch[2]
            batch_new3 = batch[3]
            batch_new4 = torch.Tensor(batch[4])
            batch_new1 = tuple([i.cuda() if ii < 4 else i for ii,i in enumerate(batch_new1)])
            batch_new2 = tuple([i.cuda() if ii < 4 else i for ii,i in enumerate(batch_new2)])
            batch_new30 = batch_new3[0]
            batch_new31 = batch_new3[1]
            batch_new30 = tuple([i.cuda() if ii < 4 else i for ii,i in enumerate(batch_new30)])
            batch_new3 = (batch_new30, batch_new31)
            batch = ((batch_new0, batch_new1, batch_new2, batch_new3,batch_new4))
            try:
                model.zero_grad()
                loss, kl_div, wacc, tacc, sacc, lfs_loss, scs_loss, denticity_loss= model(batch, beta)
                total_step += 1
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            except Exception as e:
                continue

            if np.isnan(lfs_loss):
                lfs_loss = 0
                nancount += 1
            meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100, lfs_loss, scs_loss, denticity_loss])
        
            if total_step % print_iter == 0:
                meters /= print_iter
                print("[%d] Beta: %.3f, KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f, LFS_loss: %.3f, SCS_loss: %.4f, Denticity_loss: %.4f,Nancount: %d" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model),meters[4],meters[5],meters[6],nancount))
                sys.stdout.flush()
                meters *= 0
                nancount = 0

            if total_step % save_iter == 0:
                if not os.path.exists(save_dir + "/model.iter-" + str(total_step)):
                    torch.save(model.state_dict(), save_dir + "/model.iter-" + str(total_step))

            if total_step % anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_last_lr()[0])

            if total_step % kl_anneal_iter == 0 and total_step >= warmup:
                beta = min(max_beta, beta + step_beta)
        torch.save(model.state_dict(), save_dir + "/model.epoch-" + str(epoch))

    # cleanup
    
    return model

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='../data/vae_training_data')
    parser.add_argument('--vocab', default='../data/data_vocab.txt')
    parser.add_argument('--prop_path', default='../data/prop_ss.json')
    parser.add_argument('--save_dir', default='./model_train')
    parser.add_argument('--load_epoch', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--prop_size', type=int, default=2)

    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--step_beta', type=float, default=0.002)
    parser.add_argument('--max_beta', type=float, default=0.016)
    parser.add_argument('--warmup', type=int, default=700)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=460)
    parser.add_argument('--kl_anneal_iter', type=int, default=460)
    parser.add_argument('--print_iter', type=int, default=10)
    parser.add_argument('--save_iter', type=int, default=150)
    
    args = parser.parse_args()
    print(args)
    
    main_vae_train(args=args)