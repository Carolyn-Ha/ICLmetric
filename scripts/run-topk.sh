export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
export HF_HOME="/data1/ay0119/hf-cache"

export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"
export HF_TOKEN='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'

DEVICES=6

# seeds=(1 43 666)
seeds=(1)

tasks=(ag_news commonsense_qa cr hate_speech18 mrpc openbookqa qasc qnli rte sst5 subj)
tasks=(subj)

# ice_nums=(2 4 8)
ice_nums=(2 4 8)
ice_nums=(8)

sentence_models=('sentence-transformers/all-MiniLM-L6-v2')
batch_size=1


models=(
    meta-llama/Llama-2-7b-hf
    # mistralai/Mistral-7B-v0.1
    # meta-llama/Meta-Llama-3-8B
    # meta-llama/Llama-2-7b-chat-hf
    # lmsys/vicuna-7b-v1.5
)

method='topk'
export RUN_NAME='Topk-SanityCheck'
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
                        --ice_num $ice_num
                done
            done
        done
    done
done