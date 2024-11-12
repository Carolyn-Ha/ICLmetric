export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
export HF_HOME="/data1/ay0119/hf-cache"

export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"
export HF_TOKEN='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'

DEVICES=5

# seeds=(1 43 666)
seeds=(1)

tasks=(ag_news commonsense_qa cr hate_speech18 mrpc openbookqa qasc qnli rte ss5 subj)
tasks=(qnli)

batch_size=10

ice_nums=(2 4 8)
ice_nums=(4)


models=(
    meta-llama/Llama-2-7b-hf
    meta-llama/Llama-2-7b-chat-hf
    mistralai/Mistral-7B-v0.1
    meta-llama/Meta-Llama-3-8B
    lmsys/vicuna-7b-v1.5
)

for seed in "${seeds[@]}"
do
    for model in "${models[@]}"
    do
        for task in "${tasks[@]}"
        do
            for ice_num in "${ice_nums[@]}"
            do               
                method='random'
                CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch run.py\
                    --method $method\
                    --task_name $task\
                    --model_name_or_path $model\
                    --seed $seed\
                    --batch_size $batch_size\
                    --ice_num $ice_num                
                
                method='topk'
                sentence_model='sentence-transformers/all-MiniLM-L6-v2'
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

for seed in "${seeds[@]}"
do
    for model in "${models[@]}"
    do
        for task in "${tasks[@]}"
        do
            for ice_num in "${ice_nums[@]}"
            do
                method='bm25'
                CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch run.py\
                    --method $method\
                    --task_name $task\
                    --model_name_or_path $model\
                    --seed $seed\
                    --batch_size $batch_size\
                    --ice_num $ice_num

            done
        done
    done
done


for seed in "${seeds[@]}"
do
    for model in "${models[@]}"
    do
        for task in "${tasks[@]}"
        do
            for ice_num in "${ice_nums[@]}"
            do
                method='SMI'
                export CACHE_NAME='BTL' # Before The Last
                candidate_ratio=2
                export RUN_NAME='SMIWithRetrieverModelAnd2MultipleBasedOnHalf'
                sentence_model='sentence-transformers/all-MiniLM-L6-v2'
                CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch run.py\
                    --method $method\
                    --task_name $task\
                    --model_name_or_path $model\
                    --seed $seed\
                    --batch_size $batch_size\
                    --sentence_model_path $sentence_model\
                    --ice_num $ice_num\
                    --candidate_ratio $candidate_ratio
            done
        done
    done
done


for seed in "${seeds[@]}"
do
    for model in "${models[@]}"
    do
        for task in "${tasks[@]}"
        do
            for ice_num in "${ice_nums[@]}"
            do
                
                # method='topk_cone'
                # sentence_model='sentence-transformers/all-MiniLM-L6-v2'
                # CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch run.py\
                #     --method $method\
                #     --task_name $task\
                #     --model_name_or_path $model\
                #     --seed $seed\
                #     --sentence_model_path $sentence_model\
                #     --batch_size $batch_size\
                #     --ice_num $ice_num\
                #     --select_time $select_time\
                #     --candidate_num $candidate_num

            done
        done
    done
done
