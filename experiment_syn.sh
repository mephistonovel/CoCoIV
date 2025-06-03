model="ours"
size=5000

    # low dimensional synthetic datasets 
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment b --comp-num-zc 4 --hyp_mi 5 --model_id $model --response linear --exp_id syn --true_effect 3 --repetition 20 --highdim 0
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment b --comp-num-zc 4 --hyp_mi 5 --model_id $model --response nonlinear --exp_id syn --true_effect 3 --repetition 20 --highdim 0
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment con --comp-num-zc 4 --hyp_mi 5 --model_id $model --response linear  --exp_id syn --true_effect 3 --repetition 20 --highdim 0
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment con --comp-num-zc 4 --hyp_mi 5 --model_id $model --response nonlinear  --exp_id syn --true_effect 3 --repetition 20 --highdim 0
    
    # high dimensional synthetic datasets 
    python main.py --feature-dim 784 --latent-dim 25 --latent-dim-t 25 --hidden-dim 150 --sample_size $size --batch_size 512 --treatment b --comp-num-zc 4 --hyp_mi 5 --model_id $model --response linear --highdim 1  --exp_id syn --true_effect 3 --repetition 20
    python main.py --feature-dim 784 --latent-dim 25 --latent-dim-t 25 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment b --comp-num-zc 4 --hyp_mi 5 --model_id $model --response nonlinear --highdim 1  --exp_id syn --true_effect 3 --repetition 20
    python main.py --feature-dim 784 --latent-dim 25 --latent-dim-t 25 --hidden-dim 150 --sample_size $size --batch_size 512 --treatment con --comp-num-zc 4 --hyp_mi 5 --model_id $model --response linear --highdim 1  --exp_id syn --true_effect 3 --repetition 20
    python main.py --feature-dim 784 --latent-dim 25 --latent-dim-t 25 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment con --comp-num-zc 4 --hyp_mi 5 --model_id $model --response nonlinear --highdim 1  --exp_id syn --true_effect 3 --repetition 20

    # DVAE.CIV baseline dataset 
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment b --comp-num-zc 3 --hyp_mi 5 --model_id $model --baseline 1 --true_effect 2 --exp_id syn --repetition 20

