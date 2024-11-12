export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
export HF_HOME="/data1/ay0119/hf-cache"

export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"
export HF_TOKEN='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'

DEVICES=5

# seeds=(1 43 666)
seeds=(1)

tasks=(subj hate_speech18 subj cr qnli rte ag_news commonsense_qa cr  mrpc openbookqa qasc sst5)
tasks=(subj hate_speech18 subj cr  ag_news)

ice_nums=(2 4 8)
ice_nums=(8)


models=(
    # meta-llama/Llama-2-7b-hf
    mistralai/Mistral-7B-v0.1
    # meta-llama/Meta-Llama-3-8B
    # lmsys/vicuna-7b-v1.5
    # meta-llama/Llama-2-7b-chat-hf
)

export CACHE_NAME='PMIFormated' # Before The Last
method='PMI'
batch_size=1 # for Inferencer
candidate_ratio=2
# export RUN_NAME='PMI-MiniLM-30Candidates'
# export RUN_NAME='PMI-MiniLM-30Candidates-onHalf'
# export RUN_NAME='PMI-MiniLM-30Candidates-SanityCheck'
# export RUN_NAME='PMI-MiniLM-30Candidates-Reranked'
# export RUN_NAME='PMI-Inferencer-30Candidates'
export RUN_NAME='PMI-MiniLM-30Candidates-FormatedLabelLogit-kNNReranked'

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
                    --ice_num $ice_num
            done
        done
    done
done