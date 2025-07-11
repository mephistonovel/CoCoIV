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

#experiment for synthetic data (Sec.V.A) 
def experiment_Syn(args, repetition, sample_size):
    pyro.enable_validation(__debug__)
    torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device(f'cuda:{args.GPU_id}' if torch.cuda.is_available() else 'cpu')
    cuda = True

    if args.model_id == 'dcivvae':
        device = 'cpu'
        cuda = False
    print(device)
    
    result = []
    if args.highdim:
        syn_dim = 'high'
    else:
        syn_dim = 'low'
        
    if args.model_id in ['UAS','WAS']:
        repetition = 1 
    for i in range(repetition):
        ''' (repetition) number of experiments with each random seed: 0,1,...,repetition-1 '''
        pyro.set_rng_seed(i)
        np.random.seed(i)
        print('rep:',i) # experiment number
        
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
            
            #model fit
            if args.pretrain==True and args.use_flex_enc==True:
                model.pretrain()
            model.train()
            
            ''' (Sec.IV.A)
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
            
            rep_folder = f'./Result/Reps/Syn'
            
            if not os.path.exists(rep_folder):
                os.mkdir(rep_folder) 
            csv_filename_z_train = f'rep_z_train_{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_seed{i}.csv'
            csv_filename_z_test = f'rep_z_test_{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_seed{i}.csv'
            csv_filename_c_train = f'rep_c_train_{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_seed{i}.csv'
            csv_filename_c_test = f'rep_c_test_{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_seed{i}.csv'
            
            (pd.DataFrame(zt_train)).to_csv(os.path.join(rep_folder, csv_filename_z_train), index=False)
            (pd.DataFrame(zt_test)).to_csv(os.path.join(rep_folder, csv_filename_z_test), index=False)
            (pd.DataFrame(zc_train)).to_csv(os.path.join(rep_folder, csv_filename_c_train), index=False)
            (pd.DataFrame(zc_test)).to_csv(os.path.join(rep_folder, csv_filename_c_test), index=False)

        elif args.model_id in ['autoiv','modeiv']:
            Model = Models[args.model_id]
            zt_train,zt_test = Model(train,test,i,args)
            t_train = t_train.cpu().detach().numpy().astype(np.float16)
            y_train = y_train.cpu().detach().numpy().astype(np.float16)
            
            t_test = t_test.cpu().detach().numpy().astype(np.float16)
            y_test = y_test.cpu().detach().numpy().astype(np.float16)
            
        elif args.model_id in ['UAS','WAS']:
            Model = Models[args.model_id]
            model = Model(args.model_id)

            zt_train = model.generate_zt(x_train, t_train)
            zt_test = model.generate_zt(x_test,t_test)
            t_train = t_train.cpu().detach().numpy().astype(np.float16)
            y_train = y_train.cpu().detach().numpy().astype(np.float16)
            
            t_test = t_test.cpu().detach().numpy().astype(np.float16)
            y_test = y_test.cpu().detach().numpy().astype(np.float16)
            
        elif args.model_id=='dcivvae':
                    
            ### DVAE.CIV Rep Learning ###  
            # Train.
            os.environ["CUDA_VISIBLE_DEVICES"] =  "-1"
            pyro.clear_param_store()
            args.latent_dim_t = 1
            args.latent_dim = 3
            args.latent_dim_y = 2 
            
            model = DCIVVAE(feature_dim=args.feature_dim, continuous_dim= args.feature_dim, binary_dim = 0,
                    latent_dim=args.latent_dim, latent_dim_t = args.latent_dim_t, latent_dim_y = args.latent_dim_y,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    num_samples=10)                                                                                                                                                                                                                                                                                                                                                           
            model.fit(x_train, t_train, y_train,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    learning_rate_decay=args.lrd, weight_decay=args.weight_decay)
            
            zt_test = model.guide.zt(x_test)
            zc_test = model.guide.zc(x_test)
            zy_test = model.guide.zy(x_test)

            zt_train = model.guide.zt(x_train)
            zc_train = model.guide.zc(x_train)
            zy_train = model.guide.zy(x_train)

            zt_test = zt_test.cpu().detach().numpy().astype(np.float16)
            zc_test = zc_test.cpu().detach().numpy().astype(np.float16)
            zy_test = zy_test.cpu().detach().numpy().astype(np.float16)
            zt_train = zt_train.cpu().detach().numpy().astype(np.float16)
            zc_train = zc_train.cpu().detach().numpy().astype(np.float16)
            zy_train = zy_train.cpu().detach().numpy().astype(np.float16)

            condition_x_test = np.hstack((zc_test, zy_test))
            condition_x_train = np.hstack((zc_train, zy_train))
            
            t_train = t_train.cpu().detach().numpy().astype(np.float16)
            y_train = y_train.cpu().detach().numpy().astype(np.float16)
            
            t_test = t_test.cpu().detach().numpy().astype(np.float16)
            y_test = y_test.cpu().detach().numpy().astype(np.float16)

            condition_x_test = np.hstack((zc_test, zy_test))
            condition_x_train = np.hstack((zc_train, zy_train))


        '''(Causal Effectt Estimation)
        After producing represenation of IVs (zt_000), 
        Usual IV estimators are applied: 2SLS, IVGMM, Ortho, DML, Poly2SLS, KernelIV (Sec.V)
        '''
        pyro.set_rng_seed(0)
        np.random.seed(0)
        if args.model_id in['ours','UAS','WAS','autoiv','modeiv',
                            'detach','ours_vade','novade','novade_atall']:
            estimates = estimate(args.treatment,args.response,args.true_effect, zt_train,t_train,y_train,zt_test,t_test,y_test)
            print(estimates)
            result.append(estimates)
        elif args.model_id in ['dcivvae']:
            estimates = estimate(args.treatment,args.response,args.true_effect, zt_train,t_train,y_train,zt_test,t_test,y_test,x_condition_train = condition_x_train,x_condition_test=condition_x_test)
            result.append(estimates)

        
    data_folder = f'./Result'
    
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    if args.baseline:
        filename = f'{args.model_id}_{args.treatment}_baseline_{args.response}_miw{args.hyp_mi}_{args.sample_size}.csv'
    else:    
        filename = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_hid{args.hidden_dim}.csv'
    result = pd.concat(result, axis=0)
    result.to_csv(os.path.join(data_folder,filename),index=False,float_format='%.5f')


#experiment for ablation (Sec.V.B) 
def experiment_Abl(args, repetition, sample_size):
    pyro.enable_validation(__debug__)
    torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device(f'cuda:{args.GPU_id}' if torch.cuda.is_available() else 'cpu')
    cuda = True

    if args.model_id == 'dcivvae':
        device = 'cpu'
        cuda = False
    print(device)
    
    result = []
    result_d = []

    if args.highdim:
        syn_dim = 'high'
    else:
        syn_dim = 'low'

    for i in range(repetition):
        pyro.set_rng_seed(i)
        
        if not args.baseline:
            data_load_syn = dataload[args.highdim]
        else:
            data_load_syn = dataload[2]
        train, dataloader_train, test, dataloader_test, d1= data_load_syn(args=args, n_samples= sample_size,
                                                        batch_size = args.batch_size,cuda = cuda,device =device)
        if args.model_id in ['ours', 'ours_vade']:
            # dataload
            (x_train, t_train, y_train) = train
            (x_test, t_test, y_test) = test
            # model definition
            Model = Models[args.model_id]
            model = Model(args,device, dataloader_train,dataloader_test)
            
            #model fit
            if args.pretrain==True:
                model.pretrain()
            model.train()
            
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
            
            rep_folder = f'./Result/Reps/Syn'
            if not os.path.exists(rep_folder):
                os.mkdir(rep_folder) 
            csv_filename_z_train = f'rep_z_train_{sample_size}_{args.treatment}_Syn_{syn_dim}_{args.response}_seed{i}.csv'
            csv_filename_z_test = f'rep_z_test_{sample_size}_{args.treatment}_Syn_{syn_dim}_{args.response}_seed{i}.csv'
            csv_filename_c_train = f'rep_c_train_{sample_size}_{args.treatment}_Syn_{syn_dim}_{args.response}_seed{i}.csv'
            csv_filename_c_test = f'rep_c_test_{sample_size}_{args.treatment}_Syn_{syn_dim}_{args.response}_seed{i}.csv'
            
            (pd.DataFrame(zt_train)).to_csv(os.path.join(rep_folder, csv_filename_z_train), index=False)
            (pd.DataFrame(zt_test)).to_csv(os.path.join(rep_folder, csv_filename_z_test), index=False)
            (pd.DataFrame(zc_train)).to_csv(os.path.join(rep_folder, csv_filename_c_train), index=False)
            (pd.DataFrame(zc_test)).to_csv(os.path.join(rep_folder, csv_filename_c_test), index=False)
            
            np.random.seed(0)
            estimates = estimate(args.treatment,args.response,args.true_effect,zt_train,t_train,y_train,zt_test,t_test,y_test)
            print(estimates)
            
            '''
            Sensitivity Analysis for MI regularization (Sec.V.B)
            '''
            tmp = torch.tensor(x_test).to(device)
            tmp2 = model.VaDEIV.analysis(tmp)
            result_d.append(tmp2.mean(axis=0))
            result.append(estimates)
    

    # Save Results
    data_folder = f'./Result/Ablation'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if args.use_dist_net == False and args.use_reg_net == False:
        filename1 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_auxilary.csv'
        filename2 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_auxilary_xai.csv'
    elif args.use_dist_net == False and args.use_reg_net == True:
        filename1 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_prediction.csv'
        filename2 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_prediction_xai.csv'
    elif args.use_reconst_x == False:
        filename1 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_xhat.csv'
        filename2 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_xhat_xai.csv'
    elif args.use_flex_enc == False:
        filename1 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_vade.csv'
        filename2 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_vade_xai.csv'
    else:
        filename1 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_mi.csv'
        filename2 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_mi_xai.csv'
    result = pd.concat(result, axis=0)
    result.to_csv(os.path.join(data_folder,filename1),index=False,float_format='%.5f')
    pd.DataFrame(result_d).to_csv(os.path.join(data_folder,filename2),index=False,float_format='%.5f')

# extended experiment for architecture choice  
def experiment_Abl_vade(args, repetition, sample_size):
    pyro.enable_validation(__debug__)
    torch.set_default_tensor_type('torch.FloatTensor')


    device = torch.device(f'cuda:{args.GPU_id}' if torch.cuda.is_available() else 'cpu')
    cuda = True

    if args.model_id == 'dcivvae':
        device = 'cpu'
        cuda = False
    print(device)
    
    result = []

    if args.highdim:
        syn_dim = 'high'
    else:
        syn_dim = 'low'

    for i in range(repetition):
        pyro.set_rng_seed(i)
        if not args.baseline:
            data_load_syn = dataload[args.highdim]
        else:
            data_load_syn = dataload[2]
        train, dataloader_train, test, dataloader_test, d1= data_load_syn(args=args, n_samples= sample_size,
                                                        batch_size = args.batch_size,cuda = cuda,device =device)
        if args.model_id in ['ours','novade','novade_atall']:


            (x_train, t_train, y_train) = train
            (x_test, t_test, y_test) = test
            # model definition
            Model = Models[args.model_id]
            model = Model(args,device, dataloader_train,dataloader_test)
            
            #model fit
            if args.pretrain==True:
                model.pretrain()
            model.train()
            
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
            
            rep_folder = f'./Result/Reps/Syn'
            if not os.path.exists(rep_folder):
                os.mkdir(rep_folder) 
            csv_filename_z_train = f'rep_z_train_{sample_size}_{args.treatment}_Syn_{syn_dim}_{args.response}_seed{i}.csv'
            csv_filename_z_test = f'rep_z_test_{sample_size}_{args.treatment}_Syn_{syn_dim}_{args.response}_seed{i}.csv'
            csv_filename_c_train = f'rep_c_train_{sample_size}_{args.treatment}_Syn_{syn_dim}_{args.response}_seed{i}.csv'
            csv_filename_c_test = f'rep_c_test_{sample_size}_{args.treatment}_Syn_{syn_dim}_{args.response}_seed{i}.csv'
            
            (pd.DataFrame(zt_train)).to_csv(os.path.join(rep_folder, csv_filename_z_train), index=False)
            (pd.DataFrame(zt_test)).to_csv(os.path.join(rep_folder, csv_filename_z_test), index=False)
            (pd.DataFrame(zc_train)).to_csv(os.path.join(rep_folder, csv_filename_c_train), index=False)
            (pd.DataFrame(zc_test)).to_csv(os.path.join(rep_folder, csv_filename_c_test), index=False)
            
            np.random.seed(0)
            estimates = estimate(args.treatment,args.response,args.true_effect,zt_train,t_train,y_train,zt_test,t_test,y_test)
            print(estimates)
            result.append(estimates)
    

        
    data_folder = f'./Result/Ablation'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if args.use_dist_net == False and args.use_reg_net == False:
        filename1 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_auxilary.csv'
    elif args.use_reconst_x == False:
        filename1 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_xhat.csv'
    elif args.use_flex_enc == False:
        filename1 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_vade.csv'
    else:
        filename1 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_mi.csv'
    result = pd.concat(result, axis=0)
    result.to_csv(os.path.join(data_folder,filename1),index=False,float_format='%.5f')

# Experiment for Real datasets (Sec.V.C)
def experiment_Real(args, repetition, target):
    pyro.enable_validation(__debug__)
    torch.set_default_tensor_type('torch.FloatTensor')
    # Generate synthetic data.
    device = torch.device(f'cuda:{args.GPU_id}' if torch.cuda.is_available() else 'cpu')
    cuda = True
    print(device)
    result = []
    if args.model_id in ['UAS','WAS']:
        repetition = 1 
    if args.model_id == 'dcivvae':
        device = 'cpu'
        cuda = False
    print(device)
    for i in range(repetition):
        print('rep:',i)
        pyro.set_rng_seed(i)
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
            #model fit
            if args.pretrain==True and args.use_flex_enc==True:
                model.pretrain()
            model.train()
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
            rep_folder = f'./Result/Reps/Real'
            
            if not os.path.exists(rep_folder):
                os.mkdir(rep_folder)
            csv_filename_z_train = f'rep_z_train_real_{args.treatment}_seed{i}.csv'
            csv_filename_z_test = f'rep_z_test_real_{args.treatment}_seed{i}.csv'
            csv_filename_c_train = f'rep_c_train_real_{args.treatment}_seed{i}.csv'
            csv_filename_c_test = f'rep_c_test_real_{args.treatment}_seed{i}.csv'
            (pd.DataFrame(zt_train)).to_csv(os.path.join(rep_folder, csv_filename_z_train), index=False)
            (pd.DataFrame(zt_test)).to_csv(os.path.join(rep_folder, csv_filename_z_test), index=False)
            (pd.DataFrame(zc_train)).to_csv(os.path.join(rep_folder, csv_filename_c_train), index=False)
            (pd.DataFrame(zc_test)).to_csv(os.path.join(rep_folder, csv_filename_c_test), index=False)
            # estimates = estimate(args.treatment,zt_train,t_train,y_train,zt_test,t_test,y_test)
        elif args.model_id in ['UAS','WAS']:
            Model = Models[args.model_id]
            model = Model(args.model_id)

            zt_train = model.generate_zt(x_train, t_train)
            zt_test = model.generate_zt(x_test,t_test)
            zt_train = zt_train.astype(np.float64)
            zt_test = zt_test.astype(np.float64)
            t_train = t_train.cpu().detach().numpy().astype(np.float32)
            y_train = y_train.cpu().detach().numpy().astype(np.float32)
            
            t_test = t_test.cpu().detach().numpy().astype(np.float32)
            y_test = y_test.cpu().detach().numpy().astype(np.float32)
        elif args.model_id in ['autoiv','modeiv']:
            Model = Models[args.model_id]
            zt_train,zt_test = Model(train,test,i,args)
            t_train = t_train.cpu().detach().numpy().astype(np.float32)
            y_train = y_train.cpu().detach().numpy().astype(np.float32)
            
            t_test = t_test.cpu().detach().numpy().astype(np.float32)
            y_test = y_test.cpu().detach().numpy().astype(np.float32)
        elif args.model_id=='dcivvae':
                    
            pyro.clear_param_store()
            args.latent_dim_t = 1
            args.latent_dim = 3
            args.latent_dim_y = 2 
            model = DCIVVAE(feature_dim=args.feature_dim, continuous_dim= args.feature_dim, binary_dim = 0,
                    latent_dim=args.latent_dim, latent_dim_t = args.latent_dim_t, latent_dim_y = args.latent_dim_y,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    num_samples=10)                                                                                                                                                                                                                                                                                                                                                           
            model.fit(x_train, t_train, y_train,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    learning_rate_decay=args.lrd, weight_decay=args.weight_decay)
            
            zt_test = model.guide.zt(x_test)
            zc_test = model.guide.zc(x_test)
            zy_test = model.guide.zy(x_test)

            zt_train = model.guide.zt(x_train)
            zc_train = model.guide.zc(x_train)
            zy_train = model.guide.zy(x_train)

            zt_test = zt_test.cpu().detach().numpy().astype(np.float32)
            zc_test = zc_test.cpu().detach().numpy().astype(np.float32)
            zy_test = zy_test.cpu().detach().numpy().astype(np.float32)
            zt_train = zt_train.cpu().detach().numpy().astype(np.float32)
            zc_train = zc_train.cpu().detach().numpy().astype(np.float32)
            zy_train = zy_train.cpu().detach().numpy().astype(np.float32)

            condition_x_test = np.hstack((zc_test, zy_test))
            condition_x_train = np.hstack((zc_train, zy_train))
            
            t_train = t_train.cpu().detach().numpy().astype(np.float32)
            y_train = y_train.cpu().detach().numpy().astype(np.float32)
            
            t_test = t_test.cpu().detach().numpy().astype(np.float32)
            y_test = y_test.cpu().detach().numpy().astype(np.float32)

            condition_x_test = np.hstack((zc_test, zy_test))
            condition_x_train = np.hstack((zc_train, zy_train))
            
        if args.model_id in['ours','UAS','WAS','autoiv','oursv2','ours_vade','ours_vade2','modeiv']:
            estimates = estimate_report(args.treatment,args.response,args.true_effect, zt_train,t_train,y_train,zt_test,t_test,y_test)
            print(estimates)
            result.append(estimates)
        elif args.model_id in ['dcivvae']:
            estimates = estimate_report(args.treatment,args.response,args.true_effect, zt_train,t_train,y_train,zt_test,t_test,y_test,x_condition_train = condition_x_train,x_condition_test=condition_x_test)
            result.append(estimates)
            
    data_folder = f'./Result/Real'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    filename1 = f'{args.model_id}_{args.treatment}_Real_{target}_miw{args.hyp_mi}.csv'
    result = pd.concat(result, axis=0)
    result.to_csv(os.path.join(data_folder,filename1),index=False,float_format='%.5f')
    

# Extended Experiment for assumtion violation / dependence within D1 and D2  
def experiment_Robust(args, repetition, sample_size):
    pyro.enable_validation(__debug__)
    torch.set_default_tensor_type('torch.FloatTensor')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True

    if args.model_id == 'dcivvae':
        device = 'cpu'
        cuda = False
    print(device)
    
    result = []
    result_d = []
    if args.highdim:
        syn_dim = 'high'
    else:
        syn_dim = 'low'
        
    if args.model_id in ['UAS','WAS']:
        repetition = 1 
    for i in range(repetition):

        pyro.set_rng_seed(i)
        np.random.seed(i)
        if not args.baseline:
            data_load_syn = dataload[args.highdim]
        else:
            data_load_syn = dataload[2]
            
        # dataload
        train, dataloader_train, test, dataloader_test, _= data_load_syn(args=args, n_samples= sample_size,
                                                                    batch_size = args.batch_size,cuda = cuda,device =device)
        (x_train, t_train, y_train) = train
        (x_test, t_test, y_test) = test
        if args.model_id in ['ours','ours_vade']:
            print('seed:',i)
            os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.GPU_id}"
            # model definition
            Model = Models[args.model_id]
            model = Model(args,device, dataloader_train,dataloader_test)
            
            #model fit
            if args.pretrain==True and args.use_flex_enc==True:
                model.pretrain()
            model.train()
            
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
            
            rep_folder = f'./Result/Reps/Robust'
            
            if not os.path.exists(rep_folder):
                os.mkdir(rep_folder) 
            csv_filename_z_train = f'rep_z_train_{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_seed{i}.csv'
            csv_filename_z_test = f'rep_z_test_{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_seed{i}.csv'
            csv_filename_c_train = f'rep_c_train_{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_seed{i}.csv'
            csv_filename_c_test = f'rep_c_test_{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_seed{i}.csv'
            
            (pd.DataFrame(zt_train)).to_csv(os.path.join(rep_folder, csv_filename_z_train), index=False)
            (pd.DataFrame(zt_test)).to_csv(os.path.join(rep_folder, csv_filename_z_test), index=False)
            (pd.DataFrame(zc_train)).to_csv(os.path.join(rep_folder, csv_filename_c_train), index=False)
            (pd.DataFrame(zc_test)).to_csv(os.path.join(rep_folder, csv_filename_c_test), index=False)

            tmp = torch.tensor(x_test).to(device)
            tmp2 = model.VaDEIV.analysis(tmp)
            result_d.append(tmp2.mean(axis=0))
        elif args.model_id in ['autoiv','modeiv']:
            Model = Models[args.model_id]
            zt_train,zt_test = Model(train,test,i,args)
            t_train = t_train.cpu().detach().numpy().astype(np.float16)
            y_train = y_train.cpu().detach().numpy().astype(np.float16)
            
            t_test = t_test.cpu().detach().numpy().astype(np.float16)
            y_test = y_test.cpu().detach().numpy().astype(np.float16)
        elif args.model_id in ['UAS','WAS']:
            Model = Models[args.model_id]
            model = Model(args.model_id)

            zt_train = model.generate_zt(x_train, t_train)
            zt_test = model.generate_zt(x_test,t_test)
            t_train = t_train.cpu().detach().numpy().astype(np.float16)
            y_train = y_train.cpu().detach().numpy().astype(np.float16)
            
            t_test = t_test.cpu().detach().numpy().astype(np.float16)
            y_test = y_test.cpu().detach().numpy().astype(np.float16)
            
        elif args.model_id=='dcivvae':
                    
            ### DVAE.CIV Rep Learning ###  
            pyro.clear_param_store()
            args.latent_dim_t = 1
            args.latent_dim = 3
            args.latent_dim_y = 2 
            model = DCIVVAE(feature_dim=args.feature_dim, continuous_dim= args.feature_dim, binary_dim = 0,
                    latent_dim=args.latent_dim, latent_dim_t = args.latent_dim_t, latent_dim_y = args.latent_dim_y,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    num_samples=10)                                                                                                                                                                                                                                                                                                                                                           
            model.fit(x_train, t_train, y_train,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    learning_rate_decay=args.lrd, weight_decay=args.weight_decay)
            
            zt_test = model.guide.zt(x_test)
            zc_test = model.guide.zc(x_test)
            zy_test = model.guide.zy(x_test)

            zt_train = model.guide.zt(x_train)
            zc_train = model.guide.zc(x_train)
            zy_train = model.guide.zy(x_train)

            zt_test = zt_test.cpu().detach().numpy().astype(np.float16)
            zc_test = zc_test.cpu().detach().numpy().astype(np.float16)
            zy_test = zy_test.cpu().detach().numpy().astype(np.float16)
            zt_train = zt_train.cpu().detach().numpy().astype(np.float16)
            zc_train = zc_train.cpu().detach().numpy().astype(np.float16)
            zy_train = zy_train.cpu().detach().numpy().astype(np.float16)

            condition_x_test = np.hstack((zc_test, zy_test))
            condition_x_train = np.hstack((zc_train, zy_train))
            
            t_train = t_train.cpu().detach().numpy().astype(np.float16)
            y_train = y_train.cpu().detach().numpy().astype(np.float16)
            
            t_test = t_test.cpu().detach().numpy().astype(np.float16)
            y_test = y_test.cpu().detach().numpy().astype(np.float16)

            condition_x_test = np.hstack((zc_test, zy_test))
            condition_x_train = np.hstack((zc_train, zy_train))

        pyro.set_rng_seed(0)
        np.random.seed(0)
        if args.model_id in['ours','UAS','WAS','autoiv','modeiv','ours_vade']:
            estimates = estimate(args.treatment,args.response,args.true_effect, zt_train,t_train,y_train,zt_test,t_test,y_test)
            print(estimates)
            result.append(estimates)
        elif args.model_id in ['dcivvae']:
            estimates = estimate(args.treatment,args.response,args.true_effect, zt_train,t_train,y_train,zt_test,t_test,y_test,x_condition_train = condition_x_train,x_condition_test=condition_x_test)
            result.append(estimates)

        
    data_folder = f'./Result/Robust'
    if args.interaction =='interaction' and args.dependency =='within':
        data_folder = f'./Result/Robust/Interaction'
    else:
        data_folder =  f'./Result/Robust/Within'
         
   
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    
    if args.model_id in ['ours','ours_vade']:
        filename2 = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_{args.dependency}_{args.interaction}_mi_xai.csv'
        pd.DataFrame(result_d).to_csv(os.path.join(data_folder,filename2),index=False,float_format='%.5f')

    filename = f'{args.model_id}_{args.treatment}_Syn_{syn_dim}_{args.response}_miw{args.hyp_mi}_{args.sample_size}_{args.dependency}_{args.interaction}.csv'

    result = pd.concat(result, axis=0)
    result.to_csv(os.path.join(data_folder,filename),index=False,float_format='%.5f')

           
