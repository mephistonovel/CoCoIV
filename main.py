import argparse
from Experiment.experiment import experiment_Syn, experiment_Abl,experiment_Real,experiment_Robust, experiment_Abl_vade
import warnings
from numba.core.errors import (NumbaDeprecationWarning, 
                                    NumbaPendingDeprecationWarning,
                                    NumbaPerformanceWarning)
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.filterwarnings('ignore')
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CoCoIV")
    parser.add_argument("--feature-dim", default=784, type=int)
    parser.add_argument("--latent-dim", default=25, type=int)
    parser.add_argument("--latent-dim-t", default=25, type=int)
    parser.add_argument("--latent-dim-y", default=1, type=int)
    parser.add_argument("--hidden-dim", default=150, type=int)
    parser.add_argument("--num-layers", default=3, type=int)
    parser.add_argument("--num-epochs", default=128, type=int)
    parser.add_argument("--sample_size", type=int, default=5000,help="Sample size")
    parser.add_argument("--batch_size", type=int, default=512,help="Batch size")
    parser.add_argument("--lr", default=1e-3, type=float,help='learning rate')
    parser.add_argument("--lrd",  default=0.01, type=float,help="learning-rate-decay")
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--treatment", type=str,default='b',help="Treatment type - b: binary, cat: categorical, con: continuous")
    parser.add_argument("--comp-num-zc", type=int,default=4,help="number of components of iv(zt)")
    parser.add_argument("--model_id",default = 'ours', type=str, help = 'model type')
    parser.add_argument("--hyp_reconst", type=float,default=1,help="hyperparam of weight for reconstruction term p(d|z,c)")
    parser.add_argument("--hyp_ztloss", type=float,default=1,help="hyperparam of weight for loss related to zt(z)")
    parser.add_argument("--hyp_zcloss", type=float,default=1,help="hyperparam of weight for loss related to zc(c)")
    parser.add_argument("--hyp_tdist", type=float,default=5,help="hyperparam of weight for treatment dist p(t|zt)")
    parser.add_argument("--hyp_treg", type=float,default=5,help="hyperparam of weight for treatment regression f(t|zt,zc)")
    parser.add_argument("--hyp_yreg", type=float,default=5,help="hyperparam of weight for outcome regression f(y|t,zc)")
    parser.add_argument("--hyp_mi", type=float,default=5,help="hyperparam of weight for MI loss (zt,zc)")
    parser.add_argument('--pretrain', type=bool, default=True,help='GMM pretrain')
    parser.add_argument('--pretrained_path', type=str, default='./weights/pretrained_parameter.pth',
                        help='Output path')
    parser.add_argument('--GPU_id', type=str, default='0', help='gpu id to execute code when gpu is true')
    parser.add_argument('--exp_id', type=str, default='syn', help='experiment id')
    parser.add_argument('--repetition', type=int, default=20, help='repetition')
    parser.add_argument('--dependency', type=str, default='nodep', help='within d1, d2 dependency')
    parser.add_argument('--interaction', type=str, default='nointer', help='d1 d2 data interaction')
    parser.add_argument('--response', type=str, default='linear', help='form of response function')
    parser.add_argument('--baseline', type=int, default=0, help='dvaeciv data')
    parser.add_argument('--highdim', type=int, default=1, help='highdim')
    parser.add_argument('--true_effect', type=int, default=3, help='true effect')
    parser.add_argument('--use_dist_net', type=bool, default=True,help='construct p(x|z)')
    parser.add_argument('--use_reg_net', type=bool, default=True,help='construct f(y|x^,c),f(x|z,c)')
    parser.add_argument('--use_reconst_x', type=bool, default=True,help='x^ for f(y|x^,c) vs x')    
    parser.add_argument('--use_flex_enc', type=bool, default=True,help='q(c|d) VaDE vs VAE')

    
    args = parser.parse_args()
    if args.exp_id == 'syn':
        # main experiment for synthetic datasets (Sec.V.A)
        # you can choose synthetic datasets according to treatment type, dimension, response function etc. 
        
        experiment_Syn(args, args.repetition, args.sample_size)
        
    
    if args.exp_id == 'abl':
        # extended experiments: ablation  / architecture 
        
        if args.model_id in ['novade','novade_atall']:
            # 1. Expeirment on architecture choice 
            # -> model_id: novade (vae+cvae) / novade_atall (vae+vae)
            args.hyp_mi = 5
            args.use_reconst_x = True
            args.use_flex_enc = False
            experiment_Abl_vade(args, args.repetition, args.sample_size)
        else:
            # 2. ablation 
            # -> \beta for mutual information constraints (hyp_mi)
            # -> dual vs single prediction
            # -> prediction loss vs w/o prediciton loss
            
            for w in [0,1,2,5,8,10,20]:
                args.hyp_mi = w
                experiment_Abl(args, args.repetition, args.sample_size)

            args.hyp_mi = 5
            
            args.use_dist_net = False
            args.use_reg_net = True
            experiment_Abl(args, args.repetition, args.sample_size)
            
            args.use_dist_net = False
            args.use_reg_net = False
            experiment_Abl(args, args.repetition, args.sample_size)


    
    if args.exp_id == 'real':
        # experiment for real data (Sec.V.C)
        experiment_Real(args, args.repetition, 'white')
    
    if args.exp_id =='robust':
        # extended experiment for dependence within D1/D2 
        args.dependency = 'within'
        experiment_Robust(args, args.repetition, args.sample_size)
        
        # extended experiment for assumption violation 
        args.interaction = 'interaction'
        experiment_Robust(args, args.repetition, args.sample_size)
        

