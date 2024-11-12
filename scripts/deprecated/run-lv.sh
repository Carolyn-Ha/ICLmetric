export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
export HF_HOME="/data1/ay0119/hf-cache"

export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"
export HF_TOKEN='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'

DEVICES=3

# seeds=(1 43 666)
seeds=(1)

# tasks=(subj sst2 sst5 cr ag_news mrpc qnli mnli)
tasks=(subj cr ag_news sst5 sst2 mrpc)
# tasks=(sst5 )

ice_nums=(2 4 8)
# ice_nums=(2 )


models=(
    meta-llama/Llama-2-7b-hf
    # meta-llama/Llama-2-7b-chat-hf
    # mistralai/Mistral-7B-v0.1
    # meta-llama/Meta-Llama-3-8B
    # lmsys/vicuna-7b-v1.5
)

# export RUN_NAME='Funnel' # Before The Last
export CACHE_NAME='BTL' # Before The Last
method='logitVariance'
batch_size=10 # for Inferencer
candidate_ratio=0.1
export RUN_NAME='LogitVarianceWithSamplingFor10PercentWithoutReplacement'

for seed in "${seeds[@]}"
do
    for model in "${models[@]}"
    do
        for task in "${tasks[@]}"
        do
            for ice_num in "${ice_nums[@]}"
            do
                CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch run.py\
                    --method $method\
                    --task_name $task\
                    --model_name_or_path $model\
                    --seed $seed\
                    --batch_size $batch_size\
                    --sentence_model_path 'sentence-transformers/all-MiniLM-L6-v2'\
                    --ice_num $ice_num\
                    --candidate_ratio $candidate_ratio
            done
        done
    done
done