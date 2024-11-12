# method='ReprRetriever-SelfProxy'
method=$1
shots=$2
tasks=(ag_news commonsense_qa cr hate_speech18 mrpc openbookqa qasc qnli rte sst5 subj)

models=(
    meta-llama/Llama-2-7b-hf
    meta-llama/Llama-2-7b-chat-hf
    mistralai/Mistral-7B-v0.1
    meta-llama/Meta-Llama-3-8B
    lmsys/vicuna-7b-v1.5
)

for model in "${models[@]}"
do
    for task in "${tasks[@]}"
    do
        python control.py\
        --func remove_results\
        --model $model\
        --dataset $task\
        --method $method\
        --shots $shots
    done
done
