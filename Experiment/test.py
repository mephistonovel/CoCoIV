import logging
import pandas as pd
import torch
import pyro
from Models.DCIVVAE_gpu import DCIVVAE
from Models.uas_was import Allele
from Models.ours import Ours
from Models.ours_v2 import Oursvt
from Models.ours_vade import Ours_vade
from Models.no_vade_at_all import NoVDAA
from Models.AutoIV import generate_IV
from Dataload.dataloader import data_load_syn_low, data_load_syn_high, data_load_real, data_load_syn_baseline

import numpy as np
import os
from Estimator.estimator import estimate,estimate_report

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


Models = {
    'dcivvae': DCIVVAE,
    'ours': Ours,
    'WAS': Allele,
    'UAS': Allele,
    'novade_atall': NoVDAA,
    'autoiv': generate_IV,
    'oursv2': Oursvt,
    'ours_vade': Ours_vade,
}

dataload= {
    0: data_load_syn_low,
    1: data_load_syn_high,
    2: data_load_syn_baseline
}

#experiment for synthetic data (Sec.5.1) 
def experiment_Syn(args, repetition, sample_size):
    pyro.enable_validation(__debug__)
    torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device(f'cuda:{args.GPU_id}' if torch.cuda.is_available() else 'cpu')
    cuda = True


    result = []
    if args.highdim:
        syn_dim = 'high'
    else:
        syn_dim = 'low'
        

    '''
    dataload:
    t_000: treatment
    x_000: covariates
    y_000: outcome 
    (for detailed relationship between them, see Fig.2 on page 3 of main paper)
    '''
    if not args.baseline:
        data_load_syn = dataload[args.highdim]
    else:
        data_load_syn = dataload[2]
    train, dataloader_train, test, dataloader_test, _= data_load_syn(args=args, n_samples= sample_size,
                                                                batch_size = args.batch_size,cuda = cuda,device =device)
    (x_train, t_train, y_train) = train
    (x_test, t_test, y_test) = test

    if args.model_id in ['ours','ours_vade','novade','novade_atall']:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.GPU_id}"
        # model definition
        Model = Models[args.model_id]
        model = Model(args,device, dataloader_train,dataloader_test)
        
        pth_name = f'best_model_{args.treatment}_syn_{syn_dim}_{args.response}'
        ckpt_path = os.path.join('./weights', f'{pth_name}.pth')
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.VaDEIV.load_state_dict(checkpoint)
        # model.to(device)
        # model.eval()
        ''' (Sec.4.1)
        model output:
            zt_000 : representation of IVs
            zc_000 : representation of non-IVs
        ''' 

        zc_train, _= model.VaDEIV.encode_zc(x_train)
        zt_train, _= model.VaDEIV.encode_zt(x_train)

        zc_test,_= model.VaDEIV.encode_zc(x_test)
        zt_test,_= model.VaDEIV.encode_zt(x_test)
    
        zc_test = zc_test.cpu().detach().numpy().astype(np.float16)
        zt_test = zt_test.cpu().detach().numpy().astype(np.float16)
        
        zc_train = zc_train.cpu().detach().numpy().astype(np.float16)
        zt_train = zt_train.cpu().detach().numpy().astype(np.float16)
        
        t_train = t_train.cpu().detach().numpy().astype(np.float16)
        y_train = y_train.cpu().detach().numpy().astype(np.float16)
        
        t_test = t_test.cpu().detach().numpy().astype(np.float16)
        y_test = y_test.cpu().detach().numpy().astype(np.float16)
        x_test = x_test.cpu().detach().numpy().astype(np.float16)
        

    
    pyro.set_rng_seed(0)
    np.random.seed(0)

    estimates = estimate(args.treatment,args.response,args.true_effect, zt_train,t_train,y_train,zt_test,t_test,y_test)
    print(estimates)
    result.append(estimates)


        
    data_folder = f'./Result'
    
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

  
    filename = f'TEST_{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_hid{args.hidden_dim}.csv'
    result = pd.concat(result, axis=0)
    result.to_csv(os.path.join(data_folder,filename),index=False,float_format='%.5f')


# Experiment for Real datasets (Sec.5.3)
def experiment_Real(args, repetition, target):
    pyro.enable_validation(__debug__)
    torch.set_default_tensor_type('torch.FloatTensor')
    # Generate synthetic data.
    device = torch.device(f'cuda:{args.GPU_id}' if torch.cuda.is_available() else 'cpu')
    cuda = True

    result = []

    pyro.set_rng_seed(0)
    # dataload
    train, dataloader_train, test, dataloader_test, _= data_load_real(args=args,
                                                    batch_size = args.batch_size,cuda = cuda,device =device, target=target)
    (x_train, t_train, y_train) = train
    (x_test, t_test, y_test) = test
    if args.model_id in ['ours','oursv2','ours_vade','ours_vade2']:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.GPU_id}"

        # model definition
        Model = Models[args.model_id]
        model = Model(args,device, dataloader_train,dataloader_test)

        pth_name =f'best_model_{args.treatment}_real'

        ckpt_path = os.path.join('./weights', f'{pth_name}.pth')
        checkpoint = torch.load(ckpt_path, map_location=device)

        model.VaDEIV.load_state_dict(checkpoint)

        zc_train, _= model.VaDEIV.encode_zc(x_train)
        zt_train, _= model.VaDEIV.encode_zt(x_train)
        zc_test,_= model.VaDEIV.encode_zc(x_test)
        zt_test,_= model.VaDEIV.encode_zt(x_test)
        zc_test = zc_test.cpu().detach().numpy().astype(np.float32)
        zt_test = zt_test.cpu().detach().numpy().astype(np.float32)
        zc_train = zc_train.cpu().detach().numpy().astype(np.float32)
        zt_train = zt_train.cpu().detach().numpy().astype(np.float32)
        t_train = t_train.cpu().detach().numpy().astype(np.float32)
        y_train = y_train.cpu().detach().numpy().astype(np.float32)
        t_test = t_test.cpu().detach().numpy().astype(np.float32)
        y_test = y_test.cpu().detach().numpy().astype(np.float32)
        x_test = x_test.cpu().detach().numpy().astype(np.float32)
        x_train = x_train.cpu().detach().numpy().astype(np.float32)
        
        
    estimates = estimate_report(args.treatment,args.response,args.true_effect, zt_train,t_train,y_train,zt_test,t_test,y_test)
    if args.treatment == 'con':
        estimates = estimates.iloc[:,:-4]
    print(estimates)
    result.append(estimates)

    data_folder = f'./Result'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    filename1 = f'TEST_{args.model_id}_{args.treatment}_Real_{target}_miw{args.hyp_mi}.csv'
    result = pd.concat(result, axis=0)
    result.to_csv(os.path.join(data_folder,filename1),index=False,float_format='%.5f')
    
