"""Sliced Mutual Information Retriever

Reference
- https://github.com/kentridgeai/SMI-DNN
"""

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

import tqdm
import pickle
from glob import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pprint import pformat

from sklearn.metrics import mutual_info_score
from openicl.utils.entropy_estimators import micd


logger = get_logger(__name__)
logger.addHandler(logging.FileHandler(f'./logging/{os.getenv("RUN_NAME")}_{__name__}.log', 'w'))

quantization = {
    'load_in_4bit': True,
    'load_in_8bit': False,
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_compute_dtype': 'float16',
    'bnb_4bit_quant_type': 'nf4',
}

CACHE_DIR = '/data1/ay0119/icl/cache'
CAHCE_LIST = glob(f'{CACHE_DIR}/*.pkl')
CAHCE_LIST = list(map(lambda x: x.split('/')[-1], CAHCE_LIST))


def normalize_np(arr):
    return arr / (np.linalg.norm(arr)+1e-6)

def compute_smi(xn, yn, estimate_sample_size=1):
    """
    xn : (n, d)
    yn : (n, )
    """
    def _compute_smi_sample(_xn, _yn):
        random_vector = np.random.normal(size = _xn.shape[-1])[:, np.newaxis]
        random_vector /= np.linalg.norm(random_vector, axis=0)
        _xn_sliced = np.dot(_xn, random_vector)
        # _yn_sliced = np.dot(_yn, random_vector)
        _yn_sliced = _yn

        if len(_xn_sliced.shape) < 2:
            _xn_sliced = _xn_sliced[:, np.newaxis]
        if len(_yn_sliced.shape) < 2:
            _yn_sliced = _yn_sliced[:, np.newaxis]
        logger.debug(f'_xn_sliced : {_xn_sliced.shape}')
        logger.debug(f'_yn_sliced : {_yn_sliced.shape}')
        # return mutual_info_score(_xn_sliced, _yn_sliced)
        return micd(_xn_sliced, _yn_sliced, k=len(_xn_sliced)-1, warning=False)
    # return np.mean([_compute_smi_sample(xn, yn) for _ in range(estimate_sample_size)])
    return np.mean(list(map(lambda _ : _compute_smi_sample(xn, yn), range(estimate_sample_size))))


def load_from_cache(fname):
    with open(f'{CACHE_DIR}/{fname}', 'rb') as f:
        result = pickle.load(f)
    f.close()
    return result


def keysort_dict(_dictionary):
    # Get the keys and values as lists
    keys = list(_dictionary.keys())
    values = list(_dictionary.values())

    # Get the indices that would sort the values
    sorted_indices = sorted(range(len(values)), key=lambda x: values[x])
    # Map those indices back to the corresponding keys
    sorted_keys = [keys[i] for i in sorted_indices]
    return sorted_keys


class SlicedMutualInformationeRetriever(BaseRetriever):
    """Topk In-context Learning Retriever Class
        Class of Topk Retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
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

    def __init__(
        self,
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
        accelerator: Optional[Accelerator] = None,

        inference_model_name=None,
        inference_model_tokenizer_name=None,
        seed: Optional[int] = 1,
        dataset_name:Optional[str] = '',
        candidate_ratio=None
        ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        logger.info(f'Dataset : {dataset_name}')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.retriever_tokenizer_name = tokenizer_name
        self.dataset_name = dataset_name
        self.seed = seed
        self.candidate_ratio = candidate_ratio

        gen_datalist = self.dataset_reader.generate_input_field_corpus(self.test_ds)

        logger.info(f'self.retriever_tokenizer_name : {self.retriever_tokenizer_name}')
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(self.retriever_tokenizer_name)
        self.retriever_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.retriever_tokenizer.pad_token = self.retriever_tokenizer.eos_token
        # self.retriever_tokenizer.pad_token_id = self.retriever_tokenizer.eos_token_id
        self.retriever_tokenizer.padding_side = "right"

        self.encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.retriever_tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.retriever_tokenizer, device=self.device)
        self.dataloader = DataLoader(self.encode_dataset, batch_size=self.batch_size, collate_fn=co, shuffle=False)

        self.retriever_model_name = sentence_transformers_model_name
        self.retriever_model = SentenceTransformer(sentence_transformers_model_name)
        self.retriever_model = self.retriever_model.to(self.device)
        self.retriever_model.eval()

        bnb_config = BitsAndBytesConfig(**quantization)    
        self.inference_model = AutoModelForCausalLM.from_pretrained(inference_model_name, quantization_config=bnb_config)    
        self.inference_model_tokenizer = AutoTokenizer.from_pretrained(inference_model_tokenizer_name, token=os.getenv('HF_TOKEN'))
        self.inference_model_tokenizer.pad_token = self.inference_model_tokenizer.eos_token
        self.inference_model_tokenizer.pad_token_id = self.inference_model_tokenizer.eos_token_id
        self.inference_model_tokenizer.padding_side = "right"

        self.inference_model_encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.inference_model_tokenizer)
        self.inference_model_dataloader = DataLoader(
            self.inference_model_encode_dataset, 
            batch_size=self.batch_size, 
            collate_fn=DataCollatorWithPaddingAndCuda(tokenizer=self.inference_model_tokenizer, device=self.device))

        self.sample_set = self.dataset_reader.generate_input_field_corpus(self.index_ds)
        self.sample_labels_integer_indexed = np.array(self.index_ds[self.dataset_reader.output_column])

        self.sample_labels = self.index_ds[self.dataset_reader.output_column]
        if dataset_name == 'qnli':
            label_int_to_str = {
                0 : 'Yes',
                1 : 'No'
            }
            self.sample_labels = list(map(lambda x: label_int_to_str[x], self.sample_labels))
        elif dataset_name == 'mrpc':
            label_int_to_str = {
                0 : 'No',
                1 : 'Yes'
            }
            self.sample_labels = list(map(lambda x: label_int_to_str[x], self.sample_labels))
        elif dataset_name == 'ag_news':
            label_int_to_str = {
                0 : 'world',
                1 : 'sports',
                2 : 'business',
                3 : 'technology',
            }
            self.sample_labels = list(map(lambda x: label_int_to_str[x], self.sample_labels))
        elif dataset_name == 'cr':
            label_int_to_str = {
                0 : 'negative',
                1 : 'positive',
            }
            self.sample_labels = list(map(lambda x: label_int_to_str[x], self.sample_labels))
        elif dataset_name == 'sst5':
            label_int_to_str = {
                0 : 'terrible',
                1 : 'bad',
                2 : 'okay',
                3 : 'good',
                4 : 'great',
            }
            self.sample_labels = list(map(lambda x: label_int_to_str[x], self.sample_labels))
        elif dataset_name == 'sst2':
            label_int_to_str = {
                0 : 'negative',
                1 : 'positive',
            }
            self.sample_labels = list(map(lambda x: label_int_to_str[x], self.sample_labels))
        elif dataset_name == 'subj':
            label_int_to_str = {
                0 : 'objective',
                1 : 'subjective',
            }
            self.sample_labels = list(map(lambda x: label_int_to_str[x], self.sample_labels))

        self.label_ids = sorted(list(set(self.sample_labels)))
        self.label_ids = list(map(str, self.label_ids))
        logger.info(f'label ids : {self.label_ids}')

        self.labels_information = {
            label: {'token_id': self.inference_model_tokenizer.convert_tokens_to_ids(label)} for label in self.label_ids
            }
        for label in self.label_ids:
            label_classifier = self.inference_model.lm_head.weight[self.inference_model_tokenizer.convert_tokens_to_ids(label), :].detach().cpu()
            self.labels_information[label]['classifier'] = label_classifier

        logger.debug(f'labels_information :')
        for label, label_info in self.labels_information.items():
            logger.debug(f'label:{label}')
            for k, v in label_info.items():
                if isinstance(v, torch.Tensor):
                    logger.debug(f'{k}: {v.size()}')
                else:
                    logger.debug(f'{k}: {v}')

        fname = f'Repr{os.getenv("CACHE_NAME")}-{inference_model_name.replace("/", "_")}-{self.dataset_name}.pkl'
        if fname in CAHCE_LIST:
            res_list = load_from_cache(fname)
            logger.debug(f'Cache file loaded')
        else:
            encode_datalist = DatasetEncoder(self.sample_set, tokenizer=self.inference_model_tokenizer)
            co = DataCollatorWithPaddingAndCuda(tokenizer=self.inference_model_tokenizer, device=self.device)
            dataloader = DataLoader(encode_datalist, batch_size=1, collate_fn=co, shuffle=False)            
            
            res_list = []
            for entry_idx, entry in tqdm.tqdm(enumerate(dataloader), desc='Encoding representations'):

                with torch.no_grad():
                    metadata = entry.pop("metadata")[0]
                    input_ids = entry['input_ids']
                    outputs = self.inference_model(input_ids, labels=input_ids, return_dict=True, output_hidden_states=True)

                result = {
                    'loss': outputs['loss'].item(),
                    'emb' : outputs['hidden_states'][-1][:, -1, :].contiguous().detach().cpu().numpy(),
                    'emb_btl' : outputs['hidden_states'][-2][:, -1, :].contiguous().detach().cpu().numpy(), # btl : before the last
                    'logits': outputs['logits'][:, -1].contiguous().detach().cpu().numpy(),
                    "metadata": metadata
                }
                logger.debug(f'emb : {result["emb"].shape}')
                logger.debug(f'logits : {result["logits"].shape}')
                res_list.append(result)

            with open(f'{CACHE_DIR}/{fname}', 'wb') as f:
                pickle.dump(res_list, f)
        torch.cuda.empty_cache()
        logger.debug(f'res_list : {pformat(res_list[0])}')

        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.inference_model.lm_head.weight.size(1)))
        embed_list = np.stack([res['emb'] for res in res_list]).astype('float32')
        embed_list = np.squeeze(embed_list)
        logger.info(f'embed_list : {embed_list.shape}')
        id_list = np.array([res['metadata']['id'] for res in res_list])
        index.add_with_ids(embed_list, id_list)
        self.inference_model_embed_list = embed_list
        self.inference_model_id_list = id_list
        self.inference_model_index = index
        
        self.retriever_model_index = self.create_index()

        # logger.info(f'Candidate size : {max(self.ice_num, int(np.floor(len(self.index_ds)*self.candidate_ratio)))}')

    def create_index(self, subset=None):
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.retriever_model.get_sentence_embedding_dimension()))

        fname = f'Repr{os.getenv("CACHE_NAME")}-{self.retriever_model_name.replace("/", "_")}-{self.dataset_name}.pkl'
        if fname in CAHCE_LIST:
            retriever_outputs = load_from_cache(fname)
            self.retriever_model_embed_list = retriever_outputs['embed_list']
            self.retriever_model_id_list = retriever_outputs['id_list']
        else:
            self.select_datalist = self.dataset_reader.generate_input_field_corpus(self.index_ds)
            encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.retriever_tokenizer)
            co = DataCollatorWithPaddingAndCuda(tokenizer=self.retriever_tokenizer, device=self.device)
            dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co)
            
            res_list = self.forward(dataloader, process_bar=True, information="Creating index for index set...")
            self.retriever_model_id_list = np.array([res['metadata']['id'] for res in res_list])
            self.retriever_model_embed_list = np.stack([res['embed'] for res in res_list])
        
        if subset is None:
            index.add_with_ids(self.retriever_model_embed_list, self.retriever_model_id_list)
        else:
            logger.info(f'self.retriever_model_embed_list : {self.retriever_model_embed_list.shape}')
            logger.info(f'self.retriever_model_id_list : {self.retriever_model_id_list.shape}')            
            target_ids = np.isin(self.retriever_model_id_list, subset)
            self.retriever_model_id_list = self.retriever_model_id_list[target_ids]
            self.retriever_model_embed_list = self.retriever_model_embed_list[target_ids]
            logger.info(f'self.retriever_model_embed_list : {self.retriever_model_embed_list.shape}')
            logger.info(f'self.retriever_model_id_list : {self.retriever_model_id_list.shape}')
            index.add_with_ids(self.retriever_model_embed_list, self.retriever_model_id_list)            

        return index

    def knn_search(self, ice_num):
        np.random.seed(self.seed)
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]

        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            results_ids = []
            results_labels = []

            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0)

            candidate_ids = self.retriever_model_index.search(embed, int(self.ice_num*self.candidate_ratio))[1][0].tolist()
            # candidate_ids = self.retriever_model_index.search(embed, 30)[1][0].tolist()
            # candidate_ids = self.retriever_model_index.search(embed, self.ice_num+1)[1][0].tolist()

            # candidate_embs = self.retriever_model_embed_list[candidate_ids]
            # similarity = np.matmul(normalize_np(candidate_embs), normalize_np(embed).T)
            
            results_ids.extend(candidate_ids[:int(np.floor(self.ice_num/2))])
            # results_ids.extend(candidate_ids[:self.ice_num-2])
            
            candidate_ids = candidate_ids[len(results_ids):]
            # candidate_ids.pop(candidate_ids.index(near_idx))

            # logger.info(f'results_ids : {results_ids}')
            candidate_smi = {}
            # conditioned_ids = np.random.choice(len(self.index_ds), 200, replace=False).tolist()
            conditioned_ids = results_ids
            for candidate_sample_idx, candidate_idx in enumerate(candidate_ids):
                
                candidate_xn = self.retriever_model_embed_list[conditioned_ids + [candidate_idx]]
                candidate_yn = self.sample_labels_integer_indexed[conditioned_ids + [candidate_idx]]
                
                logger.debug(f'candidate_xn : {candidate_xn}')
                logger.debug(f'candidate_yn : {candidate_yn}')
                logger.debug(f'candidate_xn : {candidate_xn.shape}')
                logger.debug(f'candidate_yn : {candidate_yn.shape}')
                
                candidate_idx_smi = compute_smi(candidate_xn, candidate_yn, estimate_sample_size=3)
                candidate_smi[candidate_idx] = candidate_idx_smi
            selected_ids = keysort_dict(candidate_smi)[::-1][:self.ice_num-len(results_ids)] # Maximum SMI
            # selected_ids = keysort_dict(candidate_smi)[::-1][:self.ice_num] # Maximum SMI
            results_ids.extend(selected_ids)

            # ## Reranking
            # results_smi = {candidate_idx:candidate_smi[candidate_idx] for candidate_idx in selected_ids}
            # for result_idx in results_ids:
            #     # TODO : Can we use inference model representation for SMI computation?
            #     # candidate_xn = self.inference_model_embed_list[results_ids + [candidate_idx]]
            #     candidate_xn = self.retriever_model_embed_list[selected_ids + [result_idx]]
            #     candidate_yn = self.sample_labels_integer_indexed[selected_ids + [result_idx]]
                
            #     logger.debug(f'candidate_xn : {candidate_xn}')
            #     logger.debug(f'candidate_yn : {candidate_yn}')
            #     logger.debug(f'candidate_xn : {candidate_xn.shape}')
            #     logger.debug(f'candidate_yn : {candidate_yn.shape}')
                
            #     result_idx_smi = compute_smi(candidate_xn, candidate_yn, estimate_sample_size=3)
            #     results_smi[result_idx] = result_idx_smi
            
            # reranked_ids = keysort_dict(results_smi)[::-1]
            # results_ids = reranked_ids
            
            assert len(results_ids) == self.ice_num, f'Selected {len(results_ids)} examples! But we need {self.ice_num} examples only'
            logger.debug(f'idx : {idx}')
            logger.debug(f'results_ids:{results_ids}')

            rtr_idx_list[idx] = results_ids
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
                raw_text = self.retriever_tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
                res = self.retriever_model.encode(raw_text, show_progress_bar=False)
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def retrieve(self):
        return self.knn_search(self.ice_num)
