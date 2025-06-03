model="ours"
size=5000

    # Ablation studies (Sec.V.B)
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment b --comp-num-zc 4 --hyp_mi 5 --model_id $model --response linear --exp_id abl --true_effect 3 --repetition 20 --highdim 0
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment b --comp-num-zc 4 --hyp_mi 5 --model_id $model --response nonlinear --exp_id abl --true_effect 3 --repetition 20 --highdim 0
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment con --comp-num-zc 4 --hyp_mi 5 --model_id $model --response linear  --exp_id abl --true_effect 3 --repetition 20 --highdim 0
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment con --comp-num-zc 4 --hyp_mi 5 --model_id $model --response nonlinear  --exp_id abl --true_effect 3 --repetition 20 --highdim 0
    
   # Extended Experiments 
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment b --comp-num-zc 4 --hyp_mi 5 --model_id $model --response linear --exp_id robust --true_effect 3 --repetition 20 --highdim 0
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment b --comp-num-zc 4 --hyp_mi 5 --model_id $model --response nonlinear --exp_id robust --true_effect 3 --repetition 20 --highdim 0
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment con --comp-num-zc 4 --hyp_mi 5 --model_id $model --response linear  --exp_id robust --true_effect 3 --repetition 20 --highdim 0
    python main.py --feature-dim 6 --latent-dim 2 --latent-dim-t 3 --hidden-dim 200 --sample_size $size --batch_size 512 --treatment con --comp-num-zc 4 --hyp_mi 5 --model_id $model --response nonlinear  --exp_id robust --true_effect 3 --repetition 20 --highdim 0
    
