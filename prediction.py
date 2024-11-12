from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import PPLInferencer, AccEvaluator
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
import os

task_name_to_path = {
    'sst2': 'stanfordnlp/sst2',
    'subj': 'SetFit/subj',
    "sst5": 'SetFit/sst5',
    'cr': 'SetFit/CR',
    "ag_news": 'SetFit/ag_news',
    #  'mnli': 'label',
    #  "qnli": 'label'
    }


input_columns_names={'sst2': ["sentence"],
            'subj': ['text'],
            "sst5": ["text"],
            "cr": ["text"],
            "ag_news": ["text"],
            'mnli': ['text1', 'text2'],
            "qnli": ["text1", "text2"]
            }

output_columns={'sst2': 'label',
             'subj': 'label',
             "sst5": 'label',
             'cr': 'label',
             "ag_news": 'label',
             'mnli': 'label',
             "qnli": 'label'
            }

test_split={
            'sst2': 'test',
            "subj": 'test',
            "sst5": 'test',
            "cr": 'test',
            "ag_news": 'test',
            'mnli': 'validation', # cannot get gold labels for the test split
            "qnli": 'validation',
    }

model_names = ['meta-llama/Llama-2-7b-chat-hf']

# set the model and dataset path
result_dir = 'results/'
dataset_path = ''

# task_names = ['sst2']
task_names = ['subj', 'sst2', 'sst5', 'cr']


for model_name in model_names:
    print('model_name:', model_name)
    for task_name in task_names:
        # load dataset
        # train_path = dataset_path + task_name + '/train.jsonl'
        # test_name = test_split[task_name]
        # test_path = dataset_path + task_name + '/' + test_name + '.jsonl'

        # combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
        combined_dataset = load_dataset(task_name_to_path[task_name])

        train_dataset = combined_dataset["train"]
        test_dataset = combined_dataset["test"]

        # Print some information about the datasets
        print(train_dataset)
        print(test_dataset)

        # Construct the DatasetReader
        data = DatasetReader(combined_dataset, input_columns=input_columns_names[task_name], output_column=output_columns[task_name])
    
        prediction_dir = result_dir + model_name + '/' + task_name
        print('task_name: {}\n'.format(task_name))

        for file_name in os.listdir(prediction_dir):
            if 'process' in file_name:
                continue

            prediction_path = os.path.join(prediction_dir, file_name)
    
            import json
            with open(prediction_path, 'r') as f:
                data1 = json.load(f)

            predictions = []
            for i in range(len(data1)):
                predictions.append(data1[str(i)]['prediction'])

            num = 0
            for i in range(len(data.references)):
                if data.references[i] == predictions[i]:
                    num += 1

            print('{} score: {}'.format(file_name, num / len(data.references)))
