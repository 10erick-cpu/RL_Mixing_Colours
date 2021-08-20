import pandas as pd

from evaluation.custom_coco_eval import CustomCOCOeval, DETECTIONS


class CocoResultsHandler():
    recall_offset = 6
    order = {
        "ap": {
            'all': {'50_95': 0,
                    '50': 1,
                    '75': 2},
            'small': {'50_95': 3},
            'medium': {'50_95': 4},
            'large': {'50_95': 5}
        }, "recall": {
            'all': {'50_95': 0 + recall_offset,
                    '50_95_2': 1 + recall_offset,
                    '50_95_3': 2 + recall_offset},
            'small': {'50_95': 3 + recall_offset},
            'medium': {'50_95': 4 + recall_offset},
            'large': {'50_95': 5 + recall_offset}
        }
    }

    def __init__(self, eval_dict, dataset, ds_type, detections=DETECTIONS):
        self.eval_dict = eval_dict
        self.detections = detections
        self.dataset = dataset
        self.ds_type = ds_type
        self.results = self.parse(self.eval_dict, self.detections)

    def parse(self, eval_dict, detections):
        data = []
        for iou_type, results in eval_dict.items():
            data.append(self.metric_to_data_dict(results, iou_type, detections))

        result = pd.concat(data)
        result['dataset'] = self.dataset
        result['ds_type'] = self.ds_type
        result.groupby(['iou_type', 'metric', 'area'])
        return result

    def name(self):
        return f"coco_results_{self.dataset}_{self.ds_type}"

    def save(self, folder):
        self.results.to_csv(folder.get_file_path(self.name() + ".csv"))

    def metric_to_data_dict(self, results, iou_type, detection):
        df = []
        for metric, areas in self.order.items():
            for area, ths in areas.items():

                for th, idx in ths.items():
                    if th == "50_95_2":
                        dets = detection[1]
                    elif th == "50_95" and metric == "recall" and area == "all":
                        dets = detection[0]
                    else:
                        dets = detection[-1]
                    data = {'iou_type': iou_type, 'metric': metric, 'area': area, 'th_type': th, 'score': results[idx],
                            'det': dets}
                    df.append(data)

        return pd.DataFrame(df)
