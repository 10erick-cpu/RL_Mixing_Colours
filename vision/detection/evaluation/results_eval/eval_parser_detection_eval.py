import pandas as pd

from utils.models.folder import Folder
from vision.detection.evaluation.results_eval.eval_parser import CombinedCocoResults, ThesisResults

target_folder_bf = "/Users/Dennis/Desktop/thesis/thesis_writing/ia_analysis_results/bf/results_bf_detection_sigmoid_instance_l1_adam_lr0002"
#target_folder_bf = "/Users/Dennis/Desktop/thesis/thesis_writing/ia_analysis_results/fl/results_fl_detection_adam"

target_folder = Folder(target_folder_bf)
files = list(target_folder.make_file_provider(extensions=['csv']))
print(files)
files = [pd.read_csv(f) for f in files]

ccr = CombinedCocoResults(files)

r = ThesisResults(ccr)
df = r.segm().ccr.df

df = df[df['th_type'].isin(["50_95", "50", "75"])]
df['th_type'] = df['th_type'].replace({'50': '0.50', '75': '0.75', '50_95': '0.50:0.95'})
df = df[df['area'] == "all"]
# df = df.drop("area", 1)


df = df.drop("det", 1)

df = df[df['ds_type'].isin(['test', 'val'])]

for idx, g in df.groupby(['dataset', 'th_type', 'metric']):
    print(g)
df = df.groupby(['dataset', 'th_type', 'metric']).agg({'score': 'mean'}).reset_index()


def rank_fn(d):
    if d.th_type == "0.50":
        return 0
    if d.th_type == "0.75":
        return 1
    return -1


df['rank'] = df.apply(lambda row: rank_fn(row), axis=1)

df = df.sort_values(['dataset', 'rank','metric',  'th_type'])


df = df.rename(columns={"th_type": "@IoU", "iou_type": "Output", "metric": "Metric", 'dataset': "Dataset", 'score':'Score'})
df['Metric'] = df['Metric'].map({'ap': 'Precision', 'recall': 'Recall'})
print(df[['Dataset', 'Metric','@IoU',  'Score']].to_latex(index=False, float_format="%.3f"))

#
#
#
# data = data[data['area'] == 'all']
#
# metrics = ['ap', 'recall']
#
# th_types = ['50', '75', '50_95']
#
#
#
# sort = ['dataset', 'ds_type', 'th_type', 'metric', 'score', 'det']
#
# target_cols = ['dataset', 'ds_type', 'th_type', 'metric', 'score']
#
# filter_ths = ['50_95_2', '50_95_3',]
#
# data = data[~data['th_type'].isin(filter_ths)]
#
# data['th_type'] = data['th_type'].replace({'50': '0.50', '75':'0.75', '50_95':'0.50:0.95'})
#
#
#
# data = data.drop('area', 1)
# data['th_type'].sort_values()
# #print(data)
#
# #print(pd.DataFrame(d))
# d = data.groupby(['dataset', 'ds_type'])
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# data = data[data['th_type']=="0.50:0.95"]
# metric = data['ds_type'].unique()
# print(metric)
