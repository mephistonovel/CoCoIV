model="ours"
size=5000

# binary treatment Real-dataset (401k) 
python main.py \
    --feature-dim 6 \
    --latent-dim 2 \
    --latent-dim-t 2 \
    --hidden-dim 20 \
    --treatment b \
    --comp-num-zc 2 \
    --hyp_mi 5 \
    --model_id $model \
    --response linear \
    --repetition 20\
    --exp_id real

# continuous treatment Real-dataset (Police Force)
python main.py \
    --feature-dim 256 \
    --latent-dim 2 \
    --latent-dim-t 3 \
    --hidden-dim 200 \
    --treatment con \
    --comp-num-zc 2 \
    --hyp_mi 5 \
    --model_id $model \
    --response linear \
    --repetition 20 \
    --exp_id real
