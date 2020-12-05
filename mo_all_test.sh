# export CUDA_VISIBLE_DEVICES=5

# t1: CHEMBL262
# t2: CHEMBL267
# t3: CHEMBL3267
# t4: CHEMBL3650
# t5: CHEMBL4005
# t6: CHEMBL4282

gpu=$1              # gpu id
data_ratio=$2       # data ratio: 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0
load_ckpt=131
meta_iter=570
train_epoch=500
inner_iter=$3       # inner epoch: 1, 5, 10, 20, ...
save_ckpt_every=$4  # 5
keep_ckpt=100
batch_size=8
seed=$5             # 42

tasks=("CHEMBL262" "CHEMBL267" "CHEMBL3267" "CHEMBL3650" "CHEMBL4005" "CHEMBL4282")

for task in ${tasks[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu} python -u all_test.py \
        -seed ${seed} \
        -batch_size ${batch_size} \
        -translate_batch_size ${batch_size} \
        -meta_test_task ${task} \
        -data processed_data/meta-test/${task} \
        -output all_test_${task}_data_${data_ratio}_inner_${inner_iter}_seed_${seed}.predict \
        -log_file all_test_${task}.log \
        -data_ratio ${data_ratio} \
        -load_meta_step ${load_ckpt} \
        -save_model experiments/all_train/model \
        -train_epochs ${train_epoch} \
        -inner_iterations ${inner_iter} \
        -save_checkpoint_epochs ${save_ckpt_every} \
        -keep_checkpoint ${keep_ckpt} \
        -report_every 50000 \
        -param_init_glorot \
        -position_encoding \
        -share_embeddings \
        -replace_unk \
        -fast > nohup_all_test_${task}_data_${data_ratio}_inner_${inner_iter}_seed_${seed}.log 2>&1
done

