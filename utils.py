from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
import pandas as pd
from glob import glob

task_names = [
    'rte','qnli',
    'mrpc', 
    'subj', 
    'sst5', 
    'cr',
    'ag_news',
    'hate_speech18',
    # 'dream', 
    'openbookqa', 
    'commonsense_qa','qasc',
    # 'gsm8k', 
    ]
model_names = [
    'meta-llama/Llama-2-7b-chat-hf', 'lmsys/vicuna-7b-v1.5', 
    'meta-llama/Llama-2-7b-hf',  'mistralai/Mistral-7B-v0.1', 
    # 'meta-llama/Meta-Llama-3-8B'
    ]

target_method = [
    'bm25', 'random', 'topk', 'topk+cone', 
    # 'PMI-MiniLM-30Candidates',
    # 'PMI-MiniLM-30Candidates-onHalf',
    # 'PMI-MiniLM-30Candidates-SanityCheck',
    # 'PMI-MiniLM-30Candidates-Reranked',
    # 'PMI-Inferencer-30Candidates',
    'PMI-MiniLM-30Candidates-FormatedLabelLogit-kNNReranked',
    'PMI-MiniLM-30Candidates-LabelLogit',
    'PMI-MiniLM-30Candidates-LabelLogit-kNNReranked',
    # 'SMIWithRetrieverModelAnd2MultipleBasedOnHalf',
    # 'SMIWithRetrieverModelAnd2MultipleBasedOnHalfReproduce',
    # 'SMIWithRetrieverModelAnd2MultipleBasedOnHalf200RandomlyConditioned',

    # 'SMIWithInferenceModelAnd2MultipleBasedOnHalf',
    # 'SMIWithRetrieverModelAnd2MultipleBasedOnMinus2',
    # 'SMIRerankedWithRetrieverModelAnd2MultipleBasedOnHalf',
    # 'SMIWithRetrieverModelAnd30CandidatesRandomlyConditioned',
    # 'SMIWithRetrieverModelAnd30Candidates100RandomlyConditioned',
    # 'SMIWithRetrieverModelAnd2MultipleCandidates100RandomlyConditioned',
    # 'SMIRerankedWithRetrieverModelAnd100RandomlyConditioned',
    # 'SMIWithRetrieverModelAnd30SharedCandidates100RandomlyConditioned',
    # 'SMIJointWithRetrieverModelAnd30SharedCandidates100RandomlyConditioned',
    # 'SMIWithRetrieverModelAnd30SharedCandidates100RandomlyConditioned10EstimatingSamples',
    ]

pretty_model_names={
    'meta-llama/Llama-2-7b-chat-hf':'Llama-2-7b-chat', 
    'meta-llama/Llama-2-7b-hf':'Llama-2-7b', 
    'meta-llama/Meta-Llama-3-8B':'Llama-3-8B', 
    'mistralai/Mistral-7B-v0.1':'Mistral-7B', 
    'lmsys/vicuna-7b-v1.5':'Vicuna-7b',
}


input_columns={}

input_columns_names = input_columns

output_columns={}

test_split={}

task_name_to_path = {}


def enroll_dataset(task_name, dataset_path, test_split_str, output_columns_str, input_columns_list):
    task_name_to_path[task_name] = dataset_path
    test_split[task_name] = test_split_str
    output_columns[task_name] = output_columns_str
    input_columns[task_name] = input_columns_list


# Natural Language Inference
enroll_dataset('qnli', 'nyu-mll/glue', 'validation', 'label', ['question', 'sentence'])
enroll_dataset('mnli', 'nyu-mll/glue', 'validation', 'label', ['premise', 'hypothesis'])
enroll_dataset('rte', 'nyu-mll/glue', 'validation', 'label', ['sentence1', 'sentence2'])
enroll_dataset('mrpc', 'nyu-mll/glue', 'test', 'label', ['sentence1', 'sentence2'])
enroll_dataset('sst5', 'SetFit/sst5', 'test', 'label', ['text'])
enroll_dataset('subj', 'SetFit/subj', 'test', 'label', ['text'])
enroll_dataset('sst2', 'stanfordnlp/sst2', 'test', 'label', ['sentence'])

enroll_dataset('ag_news', 'SetFit/ag_news', 'test', 'label', ['text'])
enroll_dataset('cr', 'SetFit/CR', 'test', 'label', ['text'])

# Sentiment classification
enroll_dataset('poem_sentiment', 'google-research-datasets/poem_sentiment', 'test', 'label', ['verse_text'])

# Sentence Completion
enroll_dataset('dream', None, 'test', 'label', ['text'])

# Hate speech detection
enroll_dataset('hate_speech18', None, 'test', 'label', ['text'])

# Question Anwering
enroll_dataset('openbookqa', None, 'test', 'label', ['text'])
enroll_dataset('commonsense_qa', None, 'validation', 'label', ['text'])
enroll_dataset('qasc', None, 'validation', 'label', ['text'])

# Math problem
enroll_dataset('gsm8k', 'openai/gsm8k', 'test', 'label', ['question', 'label'])

def load_icl_dataset(task_name):
    if task_name in ['mrpc', 'mnli', 'qnli', 'rte']:
        combined_dataset = load_dataset('glue', task_name)
    elif task_name == 'hate_speech18':
        HATE_SPEECH_DATA_DIR = '/data1/ay0119/icl/hate-speech-dataset'
        metadata = pd.read_csv(f'{HATE_SPEECH_DATA_DIR}/annotations_metadata.csv').set_index('file_id')
        label_ids = {
            'noHate': 0,
            'hate': 1,
            'relation' : 2,
            'idk/skip': 3
        }
        dataset = []
        for fpath in glob(f'{HATE_SPEECH_DATA_DIR}/all_files/*.txt'):
            fname = fpath.split('/')[-1].split('.')[0]
            with open(fpath) as f:
                text = f.readline()
            f.close()
            dataset.append({'text': text, 'label': label_ids[metadata.loc[fname, 'label']]})
        dataset = pd.DataFrame(dataset)
        combined_dataset = Dataset.from_pandas(dataset).train_test_split(0.2, seed=1)        
    elif task_name == 'openbookqa':
        dataset = load_dataset('allenai/openbookqa', 'main', trust_remote_code=True)
        alphabet2id = {'A':0, 'B':1, 'C':2, 'D':3}
        def process_text(examples):
            choices = examples['choices']
            option_list = [f'({option_label}) {option_text}' for option_label, option_text in zip(choices['label'], choices['text'])]

            question = examples['question_stem']
            options = '\n'.join(option_list)
            return {'text': f"{question}\n{options}", 'label': alphabet2id[examples['answerKey']]}
        combined_dataset = dataset.map(process_text)
    elif task_name == 'commonsense_qa':
        dataset = load_dataset('tau/commonsense_qa', trust_remote_code=True)
        dataset.pop('test')
        alphabet2id = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
        def process_text(examples):
            choices = examples['choices']
            option_list = [f'({option_label}) {option_text}' for option_label, option_text in zip(choices['label'], choices['text'])]

            question = examples['question']
            options = '\n'.join(option_list)
            return {'text': f"{question}\n{options}", 'label': alphabet2id[examples['answerKey']]}
        combined_dataset = dataset.map(process_text)
    elif task_name == 'qasc':
        dataset = load_dataset('allenai/qasc', trust_remote_code=True)
        dataset.pop('test')
        alphabet2id = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}
        def process_text(examples):
            choices = examples['choices']
            option_list = [f'({option_label}) {option_text}' for option_label, option_text in zip(choices['label'], choices['text'])]

            question = examples['question']
            options = '\n'.join(option_list)
            return {'text': f"{question}\n{options}", 'label': alphabet2id[examples['answerKey']]}
        combined_dataset = dataset.map(process_text)
    elif task_name == 'dream':
        dataset = load_dataset('dataset-org/dream', trust_remote_code=True)
        id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

        def process_text(examples):
            choices = examples['choice']
            option_list = [f'{id2alphabet[idx]} {option_text}' for idx, option_text in enumerate(examples['choice'])]

            dialogue = '\n'.join(examples['dialogue'])
            question = examples['question']
            answers = '\n'.join(option_list)
            return {
                'text': f"{dialogue}\n{question}\n{answers}", 
                'label': examples['choice'].index(examples['answer'])
                }
        combined_dataset = dataset.map(process_text)            
    elif task_name == 'gsm8k':
        dataset = load_dataset('openai/gsm8k', 'main', trust_remote_code=True)
        def process_text(examples):
            return {
                'question': examples['question'], 
                'label': examples['answer'].split('####')[-1].strip(' ')
                }
        combined_dataset = dataset.map(process_text)             
    else:
        combined_dataset = load_dataset(task_name_to_path[task_name]) 

    return combined_dataset  