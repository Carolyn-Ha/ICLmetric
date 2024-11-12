export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
export HF_HOME="/data1/ay0119/hf-cache"

export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"
export HF_TOKEN='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'

DEVICES=0

# seeds=(1 43 666)
seeds=(1)

# tasks=(subj sst2 sst5 cr ag_news mrpc) # ag_news OOM for ice num 8
tasks=(subj sst2 sst5 cr mrpc) 

sentence_models=('sentence-transformers/all-MiniLM-L6-v2')
batch_size=10
ice_nums=(2 4 8 16)
select_time=10
candidate_num=30
ctx_sample_n=2


# models=(
#     meta-llama/Llama-2-7b-hf
#     meta-llama/Llama-2-7b-chat-hf
#     mistralai/Mistral-7B-v0.1
#     meta-llama/Meta-Llama-3-8B
#     lmsys/vicuna-7b-v1.5
# )
models=(
    meta-llama/Llama-2-7b-hf
)

method='iclt'

# Format list
export ICL_FMT='plain'

export SCORE_FUNC='label_logit'


for seed in "${seeds[@]}"
do
    for model in "${models[@]}"
    do
        for task in "${tasks[@]}"
        do
            for ice_num in "${ice_nums[@]}"
            do
                for sentence_model in "${sentence_models[@]}"
                do
                    CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch run.py\
                        --method $method\
                        --task_name $task\
                        --model_name_or_path $model\
                        --seed $seed\
                        --sentence_model_path $sentence_model\
                        --batch_size $batch_size\
                        --ice_num $ice_num\
                        --select_time $select_time\
                        --candidate_num $candidate_num\
                        --ctx_sample_n $ctx_sample_n
                done
            done
        done
    done
done