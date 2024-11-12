import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from utils import model_names, target_method, pretty_model_names, task_names

results_df = pd.read_csv('./outputs/result_merged.csv', index_col=0)
results_df = results_df.loc[(results_df['Method'].isin(target_method)) & (results_df['Task'].isin(task_names))]


def gen_color_map(_categories):
    _color_map = {}
    _palette = sns.hls_palette(len(_categories))
    for i, _category in enumerate(_categories):
        _color_map[_category] = _palette[i]
    return _color_map


methods = results_df['Method'].unique().tolist()
palette = gen_color_map(methods)


# BIGFONT=10
# fig, axes = plt.subplots(len(model_names), len(task_names), figsize=(10,12))

# for model_idx, model_name in enumerate(model_names):
#     for task_idx, task_name in enumerate(task_names):
#         if task_name == 'sst2':
#             continue
#         results_df_tmp = results_df.loc[(results_df['Model']==model_name) & (results_df['Task']==task_name)].sort_values(by='ICE_Num')
#         ax = axes[model_idx][task_idx]
#         plot_tmp = sns.lineplot(results_df_tmp, x='ICE_Num', y='Performance', hue='Method', ax=ax, palette=palette, hue_order=methods)
#         if plot_tmp.get_legend() is not None:
#             plot_tmp.get_legend().remove()
#         if model_idx == 0:
#             ax.set_title(task_name, fontsize=BIGFONT)
        
#         if model_idx == len(model_names) -1:
#             ax.set_xlabel(' ')
#         else:
#             ax.set_xlabel('')

#         if task_idx == 0:
#             ax.set_ylabel(pretty_model_names[model_name], fontsize=BIGFONT)
#         else:
#             ax.set_ylabel('')
#         ax.set_xticks(sorted(results_df_tmp['ICE_Num'].unique()))
# handles, labels = ax.get_legend_handles_labels()
# # fig.legend(handles, labels, loc='center left', ncol=len(labels), fontsize=BIGFONT, bbox_to_anchor=(1, 0.5))
# fig.legend(handles, labels, loc='lower center', fontsize=BIGFONT, ncol=len(labels))
# plt.subplots_adjust(wspace=0.3, hspace=0.4)
# # fig.legend(handles, labels, loc='center right', fontsize=BIGFONT)
# # fig.tight_layout()
# fig.savefig(f'./outputs/view.png')
   

task_category={
    'rte': 'NLI',
    'mrpc': 'Paraphrase detection',
    'sst5': 'Sentiment analsys',
    'qnli': 'NLI',
    'subj': 'NLI',
    'cr': 'Sentiment analsys',
    'ag_news': 'Topic classification',
    'hate_speech18': 'Hate speech detection',
    'openbookqa': 'Question answering',
    'commonsense_qa': 'Question answering',
    'qasc': 'Question answering',
}

results_df = results_df.groupby(['Model', 'Task', 'Method', 'ICE_Num']).mean()['Performance'].reset_index()
results_df.to_csv('./outputs/results_exported.csv')

for target_ice_num in [2, 4, 8]:
    with pd.ExcelWriter(f"./outputs/results_{target_ice_num}shots_summarized.xlsx") as writer:

    # for target_ice_num in [4, 8]:
        tmp_df = results_df.loc[results_df['ICE_Num']==target_ice_num]

        for model in tmp_df['Model'].unique():
            print('='*50)
            print(f'Model : {model}')
            df_model = tmp_df.loc[tmp_df['Model'] == model].groupby(['Method', 'Task']).sum()['Performance']
            df_model = df_model.unstack(level=1)
            df_model *= 100
            # df_model = df_model[[i for i in task_names if i in df_model.columns]]
            df_model = pd.concat((df_model, df_model.mean(axis=1).to_frame(name='Avg')), axis=1)
            print(df_model)
            df_model.to_excel(writer, sheet_name=f"{pretty_model_names[model]}-{target_ice_num}Shots")

            if target_ice_num == 8:
                print(df_model)
                print(df_model.to_latex(float_format="{:.2f}".format))
            # latex_code 
            # for row_idx in range(df_model.shape[0]):
            #     row_name = df_model.index[row_idx]
            #     for col_idx in range(df_model.shape[1]):
            #         col_name = df_model.columns[col_idx]
