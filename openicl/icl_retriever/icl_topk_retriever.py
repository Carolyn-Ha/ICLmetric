"""Topk Retriever"""

from openicl import DatasetReader
from openicl.icl_dataset_reader import DatasetEncoder
from openicl.icl_retriever import BaseRetriever
from openicl.utils.collators import DataCollatorWithPaddingAndCuda
from openicl.utils.logging import get_logger
import torch
from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import tqdm
import faiss
import copy
import numpy as np
from accelerate import Accelerator
import logging
import os


logger = get_logger(__name__)
logger.addHandler(logging.FileHandler(f'./logging/{os.getenv("RUN_NAME")}_{__name__}.log', 'w'))


class TopkRetriever(BaseRetriever):
    """Topk In-context Learning Retriever Class
        Class of Topk Retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
            # 예시들 사이에서 예시와 예시를 구분
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
            # 개별 예시의 종료를 표시
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
    """
    model = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        gen_datalist = self.dataset_reader.generate_input_field_corpus(self.test_ds)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        self.encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        self.dataloader = DataLoader(self.encode_dataset, batch_size=self.batch_size, collate_fn=co)

        self.model = SentenceTransformer(sentence_transformers_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.index = self.create_index()

    def create_index(self):
        self.select_datalist = self.dataset_reader.generate_input_field_corpus(self.index_ds)
        encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer) # tokenizer를 활용해 선택된 데이터셋 encode
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co) # dataset을 배치 크기에 맞춰서 로드 
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))  # FAISS를 활용해 인덱스 객체를 생성
        res_list = self.forward(dataloader, process_bar=True, information="Creating index for index set...")
        id_list = np.array([res['metadata']['id'] for res in res_list]) # forward method를 호출해 각 데이터를 임베딩한 결과를 얻음
        self.embed_list = np.stack([res['embed'] for res in res_list])  # 각 embedding의 ID를 추출하고 임베딩 리스트를 생성
        index.add_with_ids(self.embed_list, id_list)
        return index

    def knn_search(self, ice_num):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...") # test dataset에 대한 embedding을 생성
        rtr_idx_list = [[] for _ in range(len(res_list))]
        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, ice_num)[1][0].tolist() 
                # index.search: FAISS library에 있는 function => 특정 query vector에 대해 가장 가까운 이웃 벡터를 검색하는 역할
            rtr_idx_list[idx] = near_ids
            # logger.info(f"entry['metadata'] : {entry['metadata']}")
            # logger.info(f"embed : {embed.shape}")
            logger.debug(f'idx : {idx}')
            logger.info(f'near_ids:{near_ids}')
            raise KeyError()
        return rtr_idx_list

    def forward(self, dataloader, process_bar=False, information=''):
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        if process_bar:
            logger.info(information)
            _dataloader = tqdm.tqdm(_dataloader, disable=not self.is_main_process)
        for _, entry in enumerate(_dataloader):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                raw_text = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
                res = self.model.encode(raw_text, show_progress_bar=False)
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def retrieve(self):
        return self.knn_search(self.ice_num)
