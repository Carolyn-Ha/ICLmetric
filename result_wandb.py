import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import wandb
from utils import modelnames, target_method, pretty_model_names, task_names

results_df = pd.read_csv('./outputs/result_merged.csv', index_col=0)

results_df = results_df.loc[(results_df['Method'].isin(target_method)) & (results_df['Task'].isin(task_names))]
print(results_df)


# Wandb logging for 4 shots
for model in model_names:
    for method in target_method:
        _tmp = results_df.loc[(results_df['Method']==method) & (results_df['Model']==model) & (results_df['ICE_Num']== 4)]
        for seed in _tmp['Seed'].unique():
            wandb.init(
                project="In-Context-Learning",
                config={
                    "Model": model,
                    "Method": method,
                    'Seed': seed
                },
            )
            wandb.run.name = f'{model}-{method}-{seed}'
            _tmp_seed = _tmp.loc[_tmp['Seed']==seed]
            for task in _tmp['Task'].unique():
                _tmp_seed_task = _tmp_seed.loc[_tmp_seed['Task']==task]
                wandb.log({
                    f'{task}/Performance': _tmp_seed_task.loc[_tmp_seed_task['ICE_Num']==ice_num, 'Performance'].item(),
                    },)
            
            wandb.finish()

# Wandb logging for ICE num changes
# for model in model_names:
#     for method in target_method:
#         _tmp = results_df.loc[(results_df['Method']==method) & (results_df['Model']==model)].sort_values(by='ICE_Num')
#         for seed in _tmp['Seed'].unique():
#             wandb.init(
#                 project="In-Context-Learning",
#                 config={
#                     "Model": model,
#                     "Method": method,
#                     'Seed': seed
#                 },
#             )
#             wandb.run.name = f'{model}-{method}-{seed}'
#             _tmp_seed = _tmp.loc[_tmp['Seed']==seed]
#             for task in _tmp['Task'].unique():
#                 _tmp_seed_task = _tmp_seed.loc[_tmp_seed['Task']==task]
#                 for ice_num in sorted(_tmp_seed_task['ICE_Num'].unique()):
#                     # print({f'{task}/Performance': _tmp_seed_task.loc[_tmp_seed_task['ICE_Num']==ice_num, 'Performance'].item()})
#                     wandb.log(
#                         {f'{task}/Performance': _tmp_seed_task.loc[_tmp_seed_task['ICE_Num']==ice_num, 'Performance'].item(),},
#                         step=ice_num )
            
#             wandb.finish()