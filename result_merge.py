from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import PPLInferencer, AccEvaluator
from datasets import load_dataset, concatenate_datasets, Dataset
from accelerate import Accelerator
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from glob import glob
from utils import input_columns, output_columns, test_split, task_name_to_path, load_icl_dataset, task_names, model_names

    
# set the model and dataset path
result_dir = '/data1/ay0119/icl/results/'

df = []
for model_name in model_names:
    print('model_name:', model_name)
    for task_name in task_names:
        combined_dataset = load_icl_dataset(task_name)

        train_dataset = combined_dataset["train"]
        test_dataset = combined_dataset[test_split[task_name]]

        # Print some information about the datasets
        print(train_dataset['label'][:10])
        print(test_dataset['label'][:10])

        # Construct the DatasetReader
        data = DatasetReader(combined_dataset, input_columns=input_columns[task_name], output_column=output_columns[task_name], test_split=test_split[task_name])
        if data.references is None:
            raise KeyError(f'{task_name} is None references')

        prediction_dir = result_dir + model_name + '/' + task_name
        print('task_name: {}\n'.format(task_name))
        if not os.path.isdir(prediction_dir):
            continue
        for file_name in os.listdir(prediction_dir):
            skip_iter = False
            if 'process' in file_name:
                continue
            try:
                if file_name.startswith('topk_seed_'):
                    method = 'topk'
                    _, _, seed, ice_num, _ = file_name.split('.')[0].split('_')
                elif file_name.startswith('PMI'):
                    method, _, seed, ice_num, _ = file_name.split('.')[0].split('_')
                elif file_name.startswith('SMI'):
                    method, _, seed, ice_num, _ = file_name.split('.')[0].split('_')
                elif file_name.startswith('topk_cone_seed'):
                    method = 'topk+cone'
                    _, _, _, seed, candidate_num, ice_num, _ = file_name.split('.')[0].split('_')
                elif file_name.startswith('bm25_seed_'):
                    method = 'bm25'
                    _, _, seed, ice_num, _ = file_name.split('.')[0].split('_')
                elif file_name.startswith('random_seed_'):
                    method = 'random'
                    _, _, seed, ice_num, _ = file_name.split('.')[0].split('_')
                elif file_name.startswith('ReprRetriever'):
                    method, _, seed, ice_num, _ = file_name.split('.')[0].split('_')
            except AssertionError as err:
                print(f'Except : {file_name} of {task_name}')
                skip_iter = True
            except KeyError as err:
                print(f'Except : {file_name} of {task_name}')
                skip_iter = True
            except ValueError as err:
                print(f'Except : {file_name} of {task_name}')
                skip_iter = True

            if skip_iter:
                continue
            prediction_path = os.path.join(prediction_dir, file_name)
    
            with open(prediction_path, 'r') as f:
                data_json = json.load(f)
            if len(data_json) == 0:
                continue
            predictions = []
            for i in range(len(data_json)):
                predictions.append(data_json[str(i)]['prediction'])

            num = 0
            for i in range(len(data.references)):
                if task_name == 'sst2':
                    predictions[i] = int(predictions[i])
                if data.references[i] == predictions[i]:
                    num += 1
            
            performance = num / len(data.references)

            result = {
                'Model' : model_name,
                'Task' : task_name,
                'Method': method,
                'ICE_Num' : int(ice_num),
                'Seed' : seed,
                'Performance' : performance
            }
            df.append(result)

df = pd.DataFrame(df)
print(df)
df = df.sort_values(by=['Model', 'Task', 'Method', 'ICE_Num']).reset_index(drop=True)
print(df.head())
df.to_csv('./outputs/result_merged.csv')


