export HF_DATASETS_CACHE="/data1/ay0119/hf-cache"
export HF_HOME="/data1/ay0119/hf-cache"

export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"
export HF_TOKEN='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'

DEVICES=4

# seeds=(1 43 666)
seeds=(1)

tasks=(subj hate_speech18 subj cr qnli rte ag_news commonsense_qa cr  mrpc openbookqa qasc sst5)
# tasks=(hate_speech18)

ice_nums=(2 4 8)
ice_nums=(8)


models=(
    meta-llama/Llama-2-7b-hf
    # mistralai/Mistral-7B-v0.1
    # meta-llama/Meta-Llama-3-8B
    # lmsys/vicuna-7b-v1.5
    # meta-llama/Llama-2-7b-chat-hf
)

# export RUN_NAME='Funnel' # Before The Last
export CACHE_NAME='BTL' # Before The Last
method='SMI'
batch_size=2 # for Inferencer
candidate_ratio=2
# export RUN_NAME='SMIWithRetrieverModelAnd2MultipleBasedOnHalf'
export RUN_NAME='SMIWithRetrieverModelAnd2MultipleBasedOnHalfReproduce'
# export RUN_NAME='SMIWithRetrieverModelAnd2MultipleBasedOnHalf200RandomlyConditioned'
# export RUN_NAME='SMIWithRetrieverModelAnd2MultipleBasedOnMinus2'
# export RUN_NAME='SMIRerankedWithRetrieverModelAnd2MultipleBasedOnHalf'
# export RUN_NAME='SMIWithRetrieverModelAnd30CandidatesRandomlyConditioned'
# export RUN_NAME='SMIWithRetrieverModelAnd30Candidates1000RandomlyConditioned'
# export RUN_NAME='SMIWithRetrieverModelAnd2MultipleCandidates100RandomlyConditioned'
# export RUN_NAME='SMIRerankedWithRetrieverModelAnd100RandomlyConditioned'
# export RUN_NAME='SMIWithRetrieverModelAnd30Candidates100RandomlyConditioned'
# export RUN_NAME='SMIWithRetrieverModelAnd30SharedCandidates100RandomlyConditioned'
# export RUN_NAME='SMIJointWithRetrieverModelAnd30SharedCandidates100RandomlyConditioned'
# export RUN_NAME='SMIWithRetrieverModelAnd30SharedCandidates100RandomlyConditioned10EstimatingSamples'
# export RUN_NAME='SMIRerankedWithRetrieverModelAnd30CandidatesBasedOnHalf'

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