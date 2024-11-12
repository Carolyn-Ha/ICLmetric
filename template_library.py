from openicl import PromptTemplate


subj_tp_dict = {
    0: "</E>Input: </text> Type: objective",
    1: "</E>Input: </text> Type: subjective"
}
subj_template = PromptTemplate(subj_tp_dict, {'text': '</text>'}, ice_token='</E>')

sst2_tp_dict = {
    0: "</E>Review: </text> Sentiment: negative",
    1: "</E>Review: </text> Sentiment: positive"
}
sst2_template = PromptTemplate(sst2_tp_dict, {'sentence': '</text>'}, ice_token='</E>')

sst5_tp_dict = {
    0: "</E>Review: </text> Sentiment: terrible",
    1: "</E>Review: </text> Sentiment: bad",
    2: "</E>Review: </text> Sentiment: okay",
    3: "</E>Review: </text> Sentiment: good",
    4: "</E>Review: </text> Sentiment: great",
}
sst5_template = PromptTemplate(sst5_tp_dict, {'text': '</text>'}, ice_token='</E>')

poem_sentiment_tp_dict = {
    0: "</E>Review: </text> Sentiment: negative",
    1: "</E>Review: </text> Sentiment: positive",
    2: "</E>Review: </text> Sentiment: no impact",
    3: "</E>Review: </text> Sentiment: mixed",
}
poem_sentiment_template = PromptTemplate(poem_sentiment_tp_dict, {'verse_text': '</text>'}, ice_token='</E>')

cr_tp_dict = {
    0: "</E>Review: </text> Sentiment: negative",
    1: "</E>Review: </text> Sentiment: positive"
}
cr_template = PromptTemplate(cr_tp_dict, {'text': '</text>'}, ice_token='</E>')

ag_news_tp_dict = {
    0: "</E>Input: </text> Type: world",
    1: "</E>Input: </text> Type: sports",
    2: "</E>Input: </text> Type: business",
    3: "</E>Input: </text> Type: technology",
}
ag_news_template = PromptTemplate(ag_news_tp_dict, {'text': '</text>'}, ice_token='</E>')

mnli_tp_dict = {
    0: "</E></text1> Can we know </text>? Yes.",
    1: "</E></text1> Can we know </text>? Maybe.",
    2: "</E></text1> Can we know </text>? No."
    }
mnli_template = PromptTemplate(mnli_tp_dict, {'premise': '</text1>', 'hypothesis': '</text>'}, ice_token='</E>')

qnli_tp_dict = {
    0: "</E></text1> Can we know </text>? Yes.",
    1: "</E></text1> Can we know </text>? No."
    }
qnli_template = PromptTemplate(qnli_tp_dict, {'sentence': '</text1>', 'question': '</text>'}, ice_token='</E>')

mrpc_tp_dict = {
    0: "</E>Is </text1> equivalent to </text>? No.",
    1: "</E>Is </text1> equivalent to </text>? Yes.",
    }
mrpc_template = PromptTemplate(mrpc_tp_dict, {'sentence1': '</text1>', 'sentence2': '</text>'}, ice_token='</E>')

mrpc_tp_dict = {
    0: "</E>Is </text1> equivalent to </text>? No.",
    1: "</E>Is </text1> equivalent to </text>? Yes.",
    }
mrpc_template = PromptTemplate(mrpc_tp_dict, {'sentence1': '</text1>', 'sentence2': '</text>'}, ice_token='</E>')

# ("sentence 1: " + datapoint["sentence1"] + " [SEP] sentence 2: " + datapoint["sentence2"], self.label[datapoint["label"]]))
rte_tp_dict = {
    0: "</E>Is </text1> entailed with </text>? Yes.",
    1: "</E>Is </text1> entailed with </text>? No.",
    }
rte_template = PromptTemplate(rte_tp_dict, {'sentence1': '</text1>', 'sentence2': '</text>'}, ice_token='</E>')


hate_speech18_tp_dict = {
    0: "</E>Input: </text> Type: noHate",
    1: "</E>Input: </text> Type: hate",
    2: "</E>Input: </text> Type: relation",
    3: "</E>Input: </text> Type: idk/skip",
}
hate_speech18_template = PromptTemplate(hate_speech18_tp_dict, {'text': '</text>'}, ice_token='</E>')

openbookqa_tp_dict = {
    0: "</E>Input: </text> Answer : A",
    1: "</E>Input: </text> Answer : B",
    2: "</E>Input: </text> Answer : C",
    3: "</E>Input: </text> Answer : D",
}
openbookqa_template = PromptTemplate(openbookqa_tp_dict, {'text': '</text>'}, ice_token='</E>')

dream_tp_dict = {
    0: "</E>Input: </text> Answer : A",
    1: "</E>Input: </text> Answer : B",
    2: "</E>Input: </text> Answer : C",
    3: "</E>Input: </text> Answer : D",
    4: "</E>Input: </text> Answer : E",
}
dream_template = PromptTemplate(dream_tp_dict, {'text': '</text>'}, ice_token='</E>')

gsm8k_template_str = """</E>Input: </text> Answer : </label>"""
gsm8k_template = PromptTemplate(gsm8k_template_str, {'question': '</text>', 'label': '<\label>'}, ice_token='</E>')

commonsense_qa_tp_dict = {
    0: "</E>Input: </text> Answer : A",
    1: "</E>Input: </text> Answer : B",
    2: "</E>Input: </text> Answer : C",
    3: "</E>Input: </text> Answer : D",
    4: "</E>Input: </text> Answer : E",
}
commonsense_qa_template = PromptTemplate(commonsense_qa_tp_dict, {'text': '</text>'}, ice_token='</E>')

qasc_tp_dict = {
    0: "</E>Input: </text> Answer : A",
    1: "</E>Input: </text> Answer : B",
    2: "</E>Input: </text> Answer : C",
    3: "</E>Input: </text> Answer : D",
    4: "</E>Input: </text> Answer : E",
    5: "</E>Input: </text> Answer : F",
    6: "</E>Input: </text> Answer : G",
    7: "</E>Input: </text> Answer : H",
}
qasc_template = PromptTemplate(qasc_tp_dict, {'text': '</text>'}, ice_token='</E>')


templates = {
    'sst2': sst2_template,
    'subj': subj_template,
    "sst5": sst5_template,
    'cr': cr_template,
    "ag_news": ag_news_template,
    'mrpc': mrpc_template,
    "mnli": mnli_template,
    "qnli": qnli_template,
    'poem_sentiment': poem_sentiment_template,
    'hate_speech18': hate_speech18_template,
    'openbookqa' : openbookqa_template,
    'dream': dream_template,
    'gsm8k': gsm8k_template,
    'commonsense_qa': commonsense_qa_template,
    'qasc': qasc_template,
    'rte': rte_template
    }

dictionary_templates = {
    'sst2': sst2_tp_dict,
    'subj': subj_tp_dict,
    "sst5": sst5_tp_dict,
    'cr': cr_tp_dict,
    "ag_news": ag_news_tp_dict,
    'mrpc': mrpc_tp_dict,
    "mnli": mnli_tp_dict,
    "qnli": qnli_tp_dict,
    'poem_sentiment': poem_sentiment_tp_dict,
    'hate_speech18': hate_speech18_tp_dict,
    'openbookqa' : openbookqa_tp_dict,
    'dream': dream_tp_dict,
    # 'gsm8k': gsm8k_tp_dict,
    'commonsense_qa': commonsense_qa_tp_dict,
    'qasc': qasc_tp_dict,
    'rte': rte_tp_dict
    }