import os
import sys
import torch

sys.path.append('/home/ay0119/icl')

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(f'./logging/{__name__}.log', 'w'))
logging.basicConfig(level=logging.DEBUG)

from openicl import DatasetReader
from openicl import RandomRetriever, BM25Retriever, ConERetriever, TopkRetriever, PPLInferencer, GenInferencer, AccEvaluator, SlicedMutualInformationeRetriever, PMIRetriever
from datasets import load_dataset, concatenate_datasets, Dataset
from accelerate import Accelerator

import argparse

import pandas as pd
from glob import glob

from utils import input_columns, output_columns, test_split, task_name_to_path, load_icl_dataset
from template_library import templates
parser = argparse.ArgumentParser()

# NOTE ag news : candidates 30, shot 8에서 OOM
parser.add_argument('--method') 
parser.add_argument('--task_name')
parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--seed', type=int, choices=[1, 43, 666, 9781, 124])
parser.add_argument('--sentence_model_path', choices=['sentence-transformers/all-MiniLM-L6-v2'])

parser.add_argument('--batch_size', type=int)

# Retrieve
parser.add_argument('--ice_num', type=int, choices=[2, 4, 8, 16, 24])
    # in-context learning의 예제 개수 선택

# ConE
parser.add_argument('--select_time', type=int, choices=[2, 10]) # 몇 번 샘플링해서 평가할지
parser.add_argument('--candidate_num', type=int, choices=[30, 100])

# Ours
parser.add_argument('--ctx_sample_n', type=int) # 샘플링하는 예제 개수
parser.add_argument('--candidate_ratio', type=float) # 샘플링한 예제 중 최종 후보 예제의 비율

parser.add_argument('--debug', default='FALSE')

def main():
    torch.set_default_device('cuda')
    
    args = parser.parse_args()

    FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'   # 로깅 포맷 설정
    if args.debug.upper() == 'TRUE':
        logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FORMAT, level=logging.INFO)

    output_json_filepath = '/data1/ay0119/icl/results/' + args.model_name_or_path + '/' + args.task_name
    os.makedirs(output_json_filepath, exist_ok=True)
    
    # load dataset
    combined_dataset = load_icl_dataset(args.task_name)
    print(combined_dataset)

    train_dataset = combined_dataset["train"]
    test_dataset = combined_dataset[test_split[args.task_name]]

    # Print some information about the datasets
    print(train_dataset[0])
    print(test_dataset[0])
    accelerator = Accelerator() # Accelerator 객체 생성: 모델 병렬화 지원

    # Construct the DatasetReader
    data = DatasetReader(combined_dataset, input_columns=input_columns[args.task_name], output_column=output_columns[args.task_name])

    # different retrival stratigies
    if args.method == 'topk':
        retriever = TopkRetriever(
            data, 
            sentence_transformers_model_name=args.sentence_model_path, # top K: embedding model로 충분
            ice_num=args.ice_num, 
            test_split=test_split[args.task_name],
            tokenizer_name=args.sentence_model_path, 
            batch_size=args.batch_size, 
            accelerator=accelerator)
    elif args.method == 'PMI':
        retriever = PMIRetriever(
            data, 
            retriever_model_name=args.sentence_model_path, 
            ice_num=args.ice_num, 
            test_split=test_split[args.task_name],
            batch_size=args.batch_size, 
            accelerator=accelerator,
            inferencer_model_name=args.model_name_or_path, # PMI: inference model 따로 지정 => PMI 반영
            seed=args.seed, 
            dataset_name=args.task_name,
            )
    elif args.method == 'SMI':
        retriever = SlicedMutualInformationeRetriever(
            data, 
            sentence_transformers_model_name=args.sentence_model_path, 
            ice_num=args.ice_num, 
            test_split=test_split[args.task_name],
            tokenizer_name=args.sentence_model_path, 
            batch_size=args.batch_size, 
            accelerator=accelerator,
            inference_model_name=args.model_name_or_path, 
            inference_model_tokenizer_name=args.model_name_or_path, 
            seed=args.seed, 
            dataset_name=args.task_name,
            candidate_ratio=args.candidate_ratio,
            )
    elif args.method == 'topk_cone':
        retriever = ConERetriever(
            data, 
            sentence_transformers_model_name=args.sentence_model_path, 
            ice_num=args.ice_num, 
            candidate_num=args.candidate_num, 
            test_split=test_split[args.task_name],
            tokenizer_name=args.sentence_model_path, 
            model_tokenizer_name=args.model_name_or_path, 
            ce_model_name=args.model_name_or_path,  # topK+conE: inference model 따로 지정 => conE 반영
            ice_template=templates[args.task_name], 
            select_time=args.candidate_num, 
            seed=args.seed, 
            batch_size=args.batch_size, 
            accelerator=accelerator)
    elif args.method == 'bm25': # embedding을 별도로 계산할 필요 X => text간 빈도 기반 통계
        retriever = BM25Retriever(
            data, 
            ice_num=args.ice_num, 
            test_split=test_split[args.task_name],
            accelerator=accelerator)
    elif args.method == 'random':
        retriever = RandomRetriever(
            data, 
            ice_num=args.ice_num, 
            test_split=test_split[args.task_name],
            seed=args.seed, 
            accelerator=accelerator)
    logger.info("Start inference....")

    if args.task_name in ['gsm8k']:
        inferencer = GenInferencer( # 생성형 inferencer
            model_name=args.model_name_or_path, 
            tokenizer=args.model_name_or_path,
            output_json_filepath=output_json_filepath, 
            batch_size=args.batch_size, 
            accelerator=accelerator
        )
    else:
        inferencer = PPLInferencer( # 확률 기반 접근법
            model_name=args.model_name_or_path, 
            tokenizer=args.model_name_or_path,
            output_json_filepath=output_json_filepath, 
            batch_size=args.batch_size, 
            accelerator=accelerator)

    # inference
    if args.method == 'topk':
        topk_predictions = inferencer.inference(
            retriever, 
            ice_template=templates[args.task_name], 
            output_json_filename=f'topk_seed_{args.seed}_{args.ice_num}_shot')
    elif args.method == 'PMI':
        topk_predictions = inferencer.inference(
            retriever, 
            ice_template=templates[args.task_name], 
            output_json_filename=f'{os.getenv("RUN_NAME")}_seed_{args.seed}_{args.ice_num}_shot')
    elif args.method == 'SMI':
        topk_predictions = inferencer.inference(
            retriever, 
            ice_template=templates[args.task_name], 
            output_json_filename=f'{os.getenv("RUN_NAME")}_seed_{args.seed}_{args.ice_num}_shot')
    elif args.method == 'logitVariance':
        topk_predictions = inferencer.inference(
            retriever, 
            ice_template=templates[args.task_name], 
            output_json_filename=f'{os.getenv("RUN_NAME")}_seed_{args.seed}_{args.ice_num}_shot')
    elif args.method == 'topk_cone':
        cone_predictions = inferencer.inference(
            retriever, 
            ice_template=templates[args.task_name], 
            output_json_filename=f'topk_cone_seed_{args.seed}_{args.candidate_num}_{args.ice_num}_shot')
    elif args.method == 'bm25':
        bm25_predictions = inferencer.inference(
            retriever, 
            output_json_filename=f'bm25_seed_{args.seed}_{args.ice_num}_shot')
    elif args.method =='random':
        random_predictions = inferencer.inference(
            retriever, 
            output_json_filename=f'random_seed_{args.seed}_{args.ice_num}_shot')
    elif args.method == 'repr':
        prefix = 'ReprRetriever'
        if os.getenv('RUN_NAME'):
            prefix += f'-{os.getenv("RUN_NAME")}'
        repr_predictions = inferencer.inference(
            retriever, 
            output_json_filename=f'{prefix}_seed_{args.seed}_{args.ice_num}_shot')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)
        raise Exception(e)
