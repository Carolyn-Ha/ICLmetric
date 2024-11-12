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

import pickle
from glob import glob
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from template_library import dictionary_templates, templates

logger = get_logger(__name__)
logger.addHandler(logging.FileHandler(f'./logging/{os.getenv("RUN_NAME")}_{__name__}.log', 'w'))

bnb_config = {
    'load_in_4bit': True,
    'load_in_8bit': False,
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_compute_dtype': 'float16',
    'bnb_4bit_quant_type': 'nf4',
}

CACHE_DIR = '/data1/ay0119/icl/cache'
CAHCE_LIST = glob(f'{CACHE_DIR}/*.pkl')
CAHCE_LIST = list(map(lambda x: x.split('/')[-1], CAHCE_LIST))


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


def compute_prob(logits, input_ids):
    # logger.info(f'logits : {logits.size()}')
    # logger.info(f'input_ids : {input_ids.size()}')
    
    logits = torch.exp(logits) # (Seq len, Vocab size)
    logits_factor = logits[torch.arange(logits.size(0)), input_ids[0, 1:]] / logits.sum(dim=-1) # (Seq len, )
    # logger.info(f'logits_factor : {logits_factor.size()}')
    
    # prob = torch.exp(torch.log(logits_factor / logits_norm).sum())
    prob = torch.prod(logits_factor)
    # logger.info(f'prob : {prob}')    
    return prob


class PMIRetriever(BaseRetriever):
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
        retriever_model_name: Optional[str] = 'all-mpnet-base-v2',
        ice_num: Optional[int] = 1,
        index_split: Optional[str] = 'train',
        test_split: Optional[str] = 'test',
        batch_size: Optional[int] = 1,
        accelerator: Optional[Accelerator] = None,
        inferencer_model_name=None,
        seed: Optional[int] = 1,
        dataset_name:Optional[str] = '') -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.retriever_model_name = retriever_model_name
        self.inferencer_model_name = inferencer_model_name

        # self.tokenizer_name = tokenizer_name
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
        self.retriever_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.retriever_tokenizer.pad_token = self.retriever_tokenizer.eos_token
        # self.retriever_tokenizer.pad_token_id = self.retriever_tokenizer.eos_token_id
        self.retriever_tokenizer.padding_side = "right"
        self.retriever_model = SentenceTransformer(retriever_model_name)
        self.retriever_model = self.retriever_model.to(self.device)
        self.retriever_model.eval()

        self.inferencer_tokenizer = AutoTokenizer.from_pretrained(inferencer_model_name, token=os.getenv('HF_TOKEN'))
        self.inferencer_tokenizer.pad_token = self.inferencer_tokenizer.eos_token
        self.inferencer_tokenizer.pad_token_id = self.inferencer_tokenizer.eos_token_id
        self.inferencer_tokenizer.padding_side = "right"        
        self.inferencer_model = AutoModelForCausalLM.from_pretrained(inferencer_model_name, quantization_config=bnb_config) 

        self.index_label_list = self.index_ds[self.dataset_reader.output_column]
        self.label_set_int = list(set(self.index_ds[self.dataset_reader.output_column]))
        if dataset_name == 'qnli':
            self.label_int_to_str = {
                0 : 'Yes',
                1 : 'No'
            }
        elif dataset_name == 'rte':
            self.label_int_to_str = {
                0 : 'Yes',
                1 : 'No'
            }
        elif dataset_name == 'mrpc':
            self.label_int_to_str = {
                0 : 'No',
                1 : 'Yes'
            }
        elif dataset_name == 'ag_news':
            self.label_int_to_str = {
                0 : 'world',
                1 : 'sports',
                2 : 'business',
                3 : 'technology',
            }
        elif dataset_name == 'cr':
            self.label_int_to_str = {
                0 : 'negative',
                1 : 'positive',
            }
        elif dataset_name == 'sst5':
            self.label_int_to_str = {
                0 : 'terrible',
                1 : 'bad',
                2 : 'okay',
                3 : 'good',
                4 : 'great',
            }
        elif dataset_name == 'sst2':
            self.label_int_to_str = {
                0 : 'negative',
                1 : 'positive',
            }
        elif dataset_name == 'subj':
            self.label_int_to_str = {
                0 : 'objective',
                1 : 'subjective',
            }
        elif dataset_name == 'openbookqa':
            self.label_int_to_str = {
                0 : 'A',
                1 : 'B',
                2 : 'C',
                3 : 'D',
            }
        elif dataset_name == 'hate_speech18':
            self.label_int_to_str = {
                0 : 'No',
                1 : 'Yes',
                2 : 'relation',
                3 : 'skip',
            }
        elif dataset_name == 'poem_sentiment':
            self.label_int_to_str = {
                0 : 'negative',
                1 : 'positive',
                2 : 'neutral',
                3 : 'mixed',
            }
        elif dataset_name == 'openbookqa':
            self.label_int_to_str = {
                0 : 'A',
                1 : 'B',
                2 : 'C',
                3 : 'D',
            }
        elif dataset_name == 'qasc':
            self.label_int_to_str = {
                0 : 'A',
                1 : 'B',
                2 : 'C',
                3 : 'D',
                4 : 'E',
                5 : 'F',
                6 : 'G',
                7 : 'H',
            }            
        else:
            raise KeyError(f'{dataset_name} is not Found!!')
        self.label_set_str = list(map(lambda x: self.label_int_to_str[x], self.label_set_int))        
        self.labels_information = {
            label_int: {
                'label_str' : self.label_int_to_str[label_int],
                'token_id': self.inferencer_tokenizer.convert_tokens_to_ids(self.label_int_to_str[label_int]),
                'label_token' : self.inferencer_tokenizer.encode_plus(self.label_int_to_str[label_int], truncation=True, return_tensors='pt', verbose=False)['input_ids'],
                'template_str': dictionary_templates[dataset_name][label_int],
                'template_tokenized' : self.inferencer_tokenizer.encode_plus(dictionary_templates[dataset_name][label_int], truncation=True, return_tensors='pt', verbose=False)['input_ids'],
                } for label_int in self.label_set_int
            }

        # test_datalist = self.dataset_reader.generate_input_field_corpus(self.test_ds)
        # self.test_encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer)
        # co = DataCollatorWithPaddingAndCuda(tokenizer=self.retriever_tokenizer, device=self.device)
        # self.dataloader = DataLoader(self.test_encode_dataset, batch_size=self.batch_size, collate_fn=co)
        self.retriever_index, self.retriever_embed_list, self.retriever_id_list = self.setup_index(self.retriever_model, self.retriever_tokenizer, self.retriever_model.get_sentence_embedding_dimension())

        self.inferencer_index, self.inferencer_embed_list, self.inferencer_id_list = self.setup_index(self.inferencer_model, self.inferencer_tokenizer, self.inferencer_model.lm_head.weight.size(1))

        # self.testset_dataloader = self.create_dataloader(self.test_ds, self.retriever_tokenizer)
        # self.testset_dataloader = self.create_dataloader(self.test_ds, self.inferencer_tokenizer)

    def create_dataloader(self, dataset, tokenizer):
        target_datalist = self.dataset_reader.generate_input_field_corpus(dataset)
        encode_dataset = DatasetEncoder(target_datalist, tokenizer=tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=tokenizer, device=self.device)
        dataloader = DataLoader(encode_dataset, batch_size=self.batch_size, collate_fn=co)
        return dataloader

    def setup_index(self, model, tokenizer, dimension):
        # self.select_datalist = self.dataset_reader.generate_input_field_corpus(self.index_ds)
        # encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer)
        # co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        # dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co)
        dataloader = self.create_dataloader(self.index_ds, tokenizer)

        index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

        if isinstance(model, SentenceTransformer):
            logger.info(f'Current model is sentence transformer : {model}')
            fname = f'Repr{os.getenv("CACHE_NAME")}-{self.retriever_model_name.replace("/", "_")}-{self.dataset_name}.pkl'
            if fname in CAHCE_LIST:
                res_list = load_from_cache(fname)
            else:
                res_list = self.retriever_forward(dataloader, process_bar=True, information="Creating index for index set...")
                    # 별도의 embedding model을 사용할 경우: retrieve_forward를 통해 생성된 임베딩 사용
                with open(f'{CACHE_DIR}/{fname}', 'wb') as f:
                    pickle.dump(res_list, f)
            id_list = np.array([res['metadata']['id'] for res in res_list])
            embed_list = np.stack([res['embed'] for res in res_list])  
            self.retriever_res_list = res_list          
        else:            
            logger.info(f'Current model is LLM: {model}')
            fname = f'Repr{os.getenv("CACHE_NAME")}-{self.inferencer_model_name.replace("/", "_")}-{self.dataset_name}.pkl'
            if fname in CAHCE_LIST:
                logger.debug(f'Cache file loaded')            
                res_list = load_from_cache(fname)
            else:            
                res_list = self.inferencer_forward(dataloader, process_bar=True, information="Creating index for index set...")
                    # LLM을 사용할 경우: inferencer_forward를 통해 생성된 임베딩 사용
                with open(f'{CACHE_DIR}/{fname}', 'wb') as f:
                    pickle.dump(res_list, f)
            embed_list = np.stack([res['embed'] for res in res_list]).astype('float32')
            embed_list = np.squeeze(embed_list)
            id_list = np.array([res['metadata']['id'] for res in res_list])                
            self.inferencer_res_list = res_list
        logger.info(f'embed_list : {embed_list.shape}')
        index.add_with_ids(embed_list, id_list)
        return index, embed_list, id_list

    def retriever_forward(self, dataloader, process_bar=False, information=''):
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        if process_bar:
            logger.info(information)
            _dataloader = tqdm.tqdm(_dataloader, disable=not self.is_main_process)
        for _, entry in enumerate(_dataloader):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                raw_text = self.retriever_tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
                    # retriever tokenizer 활용
                res = self.retriever_model.encode(raw_text, show_progress_bar=False)
                    # Sentence Transformer를 활용해 텍스트 임베딩 생성
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def inferencer_forward(self, dataloader, process_bar=False, information=''):
        def format_labeled_input(example):
            res = dictionary_templates[self.dataset_name][example[self.dataset_reader.output_column]]
            for column, key in template_col_to_key.items():
                res = res.replace(key, example[column])
            return {'labeled_input' : res}
        res_list = []
        template_col_to_key = templates[self.dataset_name].column_token_map

        target_datalist = self.index_ds.map(lambda x: format_labeled_input(x))['labeled_input']

        encode_dataset = DatasetEncoder(target_datalist, tokenizer=self.inferencer_tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.inferencer_tokenizer, device=self.device)
        _dataloader = DataLoader(encode_dataset, batch_size=self.batch_size, collate_fn=co)        

        # _dataloader = copy.deepcopy(dataloader)
        if process_bar:
            logger.info(information)
            _dataloader = tqdm.tqdm(_dataloader, disable=not self.is_main_process)

        for entry_idx, entry in enumerate(_dataloader):
            entry_label = self.index_label_list[entry_idx]
            entry_label_token = self.labels_information[entry_label]['label_token']
            # entry_label_token_id = self.labels_information[self.label_int_to_str[entry_label]]['token_id']
            with torch.no_grad():
                metadata = entry.pop("metadata")[0]
                # input_ids = torch.concat((entry['input_ids'], entry_label_token), dim=-1)                
                input_ids = entry['input_ids']
                outputs = self.inferencer_model(input_ids, labels=input_ids, return_dict=True, output_hidden_states=True)

                # #P(Y|X)
                logits = outputs['logits'][0] # (Seq len, vocabulary size)
                logits_factor = logits[-entry_label_token.size(0):, entry_label_token].mean()

                # #Method : PMI computation
                p_yx_logits = outputs['logits'][0, :-1] # (Seq len, vocabulary size)
                p_yx = compute_prob(p_yx_logits, input_ids)
                
                p_y_input_ids = self.labels_information[entry_label]['template_tokenized']
                p_y_logits = self.inferencer_model(p_y_input_ids, return_dict=True)['logits'][0, :-1] # (Seq len, vocabulary size)
                p_y = compute_prob(p_y_logits, p_y_input_ids)

            #     outputs = self.inferencer_model(input_ids, labels=input_ids, return_dict=True, output_hidden_states=True)
            result = {
                'loss': outputs['loss'].item(),
                'embed' : outputs['hidden_states'][-1][:, -1, :].contiguous().detach().cpu().numpy(),
                # 'emb_btl' : outputs['hidden_states'][-2][:, -1, :].contiguous().detach().cpu().numpy(), # btl : before the last
                # 'last_token_logits': outputs['logits'][:, -1].contiguous().detach().cpu().numpy(),
                # 'label_logits': outputs['logits'][:, :, entry_label_token_id].contiguous().detach().cpu().numpy(),
                "metadata": metadata}        
            # logger.info(f'last_token_logits : {result["last_token_logits"].shape}')
            # logger.info(f'label_logits : {result["label_logits"].shape}')
            result['pmi'] = torch.log(p_yx / p_y).item()
            result['label_logit'] = logits_factor.item()
            # logger.info(f"pmi : {result['pmi']}")
            
            res_list.append(result)
        return res_list

    def retrieve(self):
        # test_dataloader = self.create_dataloader(self.test_ds, self.inferencer_tokenizer)
        test_dataloader = self.create_dataloader(self.test_ds, self.retriever_tokenizer)
        _test_dataloader = copy.deepcopy(test_dataloader)

        # rtr_idx_list = [[] for _ in range(len(test_emb_list))]

        logger.info("Retrieving data for test set...")        
        rtr_idx_list = []
        rtr_idx_list = [[] for _ in range(len(self.test_ds))]
        for entry_idx, entry in enumerate(tqdm.tqdm(_test_dataloader, disable=not self.is_main_process)):
            metadata = entry.pop("metadata")[0]
            test_sample_idx = metadata['id']        

            # test_sample_embed = self.inferencer_model(entry['input_ids'], return_dict=True, output_hidden_states=True)['hidden_states'][-1][:, -1, :].contiguous().detach().cpu().numpy().astype('float32')
            # candidates = self.inferencer_index.search(test_sample_embed, 30)[1][0].tolist()
            
            raw_text = self.retriever_tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
            test_sample_embed = self.retriever_model.encode(raw_text, show_progress_bar=False)
            candidates = self.retriever_index.search(test_sample_embed, 30)[1][0].tolist()

            # #Method :SanityCheck
            # result_ids = candidates[:self.ice_num]

            # #Method : PMI-MiniLM-30Candidates-LabelLogit
            # candidates_pmi = {candidate_idx :self.inferencer_res_list[candidate_idx]['label_logit'] for candidate_idx in candidates}
            # selected_ids = keysort_dict(candidates_pmi)[:self.ice_num] 
            # result_ids = selected_ids

            #Method : PMI-MiniLM-30Candidates-LabelLogit-kNNReranked
            candidates_pmi = {candidate_idx :self.inferencer_res_list[candidate_idx]['label_logit'] for candidate_idx in candidates}
            selected_ids_logit = keysort_dict(candidates_pmi)[:self.ice_num] 
            selected_ids_logit_to_distance = {selected_idx: candidates.index(selected_idx) for selected_idx in selected_ids_logit}
            result_ids = keysort_dict(selected_ids_logit_to_distance)


            # #Method : PMI-MiniLM-30Candidates
            # candidates_pmi = { candidate_idx :self.inferencer_res_list[candidate_idx]['pmi'] for candidate_idx in candidates}
            # selected_ids = keysort_dict(candidates_pmi)[::-1][:self.ice_num] 
            # result_ids = selected_ids

            # #Method : PMI-MiniLM-30Candidates-onHalf
            # result_ids = candidates[:int(np.floor(self.ice_num/2))]
            # candidates = candidates[int(np.floor(self.ice_num/2)):]
            # candidates_pmi = { candidate_idx :self.inferencer_res_list[candidate_idx]['pmi'] for candidate_idx in candidates}
            # selected_ids = keysort_dict(candidates_pmi)[::-1][:self.ice_num - len(result_ids)] 
            # result_ids.extend(selected_ids)

            # #Method : PMI-MiniLM-30Candidates-Reranked
            # candidates = candidates[:self.ice_num]
            # candidates_pmi = {candidate_idx :self.inferencer_res_list[candidate_idx]['pmi'] for candidate_idx in candidates}
            # selected_ids = keysort_dict(candidates_pmi)[::-1]
            # result_ids = selected_ids

            ########################################
            # logger.info(f'entry_idx : {entry_idx}')
            # logger.info(f'raw_text : {raw_text}')
            # logger.info(f'test_sample_idx : {test_sample_idx}')
            # logger.info(f'test_sample_embed  : {test_sample_embed.shape}')
            # logger.info(f'candidates ({len(candidates)}): {candidates}')
            # logger.info(f'ICL Num : {self.ice_num}')
            # logger.info(f'result_ids ({len(result_ids)}) : {result_ids}')
            # raise KeyError()

            rtr_idx_list[test_sample_idx] = result_ids
            assert len(result_ids) == self.ice_num, f'Selected {len(result_ids)} examples! But we need {self.ice_num} examples only'

            logger.debug(f'test_sample_idx : {test_sample_idx} = {rtr_idx_list[test_sample_idx]}')

        for idx_sample in rtr_idx_list:
            assert len(idx_sample) ==  self.ice_num
        return rtr_idx_list
