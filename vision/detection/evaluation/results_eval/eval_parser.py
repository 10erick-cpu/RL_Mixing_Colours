import pandas as pd

from utils.models.folder import Folder

pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr',
              False)  # more options can be specified also


class CocoResults(object):
    def __init__(self, base_folder):
        self.base = base_folder
        self.df = self.parse()
        self.df = self.clean()

    def parse(self):
        files = list(self.base.get_files(filter_extensions=['csv']))
        files = [pd.read_csv(f) for f in files]
        df = pd.concat(files)

        folder_name = self.base.name
        try:
            if "fl" in folder_name:
                det_type, activation, norm_type, loss = folder_name.split("_")
            else:
                det_type, activation, norm_type, loss = "bf", "sigmoid", "instance", "l1"

            df['det_type'] = det_type
            df['activation'] = activation
            df['norm_type'] = norm_type
            df['loss'] = loss
        except Exception as e:
            pass
        return df

    def clean(self):
        df = self.df
        df = df[df['area'] == "all"]
        df = df[df['th_type'].isin(["50_95", "50", "75"])]

        filter_ths = ['50_95_2', '50_95_3']

        df = df[~df['th_type'].isin(filter_ths)]
        df['ds_type'] = df['ds_type'].replace({'test_dense': 'dense', 'test_sparse': 'sparse'})
        df['th_type'] = df['th_type'].replace({'50': '0.50', '75': '0.75', '50_95': '0.50:0.95'})

        df = df.drop("area", 1)
        df = df.drop("det", 1)

        self.df = df
        return df


class CombinedCocoResults(object):
    def __init__(self, coco_results, is_df=False):
        if is_df:
            self.df = coco_results
        else:
            self.df = pd.concat(coco_results, sort=True)

    def _filter(self, df, property, target_val):
        return df[df[property] == target_val]

    def get_iou(self, df, type):
        return self._filter(df, 'iou_type', type)

    def get_bbox(self, df):
        return self._filter(df, 'iou_type', "bbox")

    def get_by_th(self, df, th):
        return self._filter(df, 'th_type', th)

    def get_by_det_type(self, df, det):
        return self._filter(df, 'det_type', det)

    def ds_type(self, df, type):
        return self._filter(df, "ds_type", type)

    def norm(self, df, type):
        return self._filter(df, "norm_type", type)

    def loss(self, df, l):
        return self._filter(df, "loss", l)

    def _as_result(self, df):
        return df.sort_values("score", ascending=False)

    def get_old(self, iou, ds, th, norm=None, loss=None):
        df = self.get_by_th(self.ds_type(self.get_iou(self.df, iou), ds), th)
        if norm is not None:
            df = self.norm(df, norm)
        if loss is not None:
            df = self.loss(df, loss)
        return self._as_result(df)

    def get(self, **kwargs):
        df = self.df.copy()
        for k, v in kwargs.items():
            df = self._filter(df, str(k), v)
        return self._as_result(df)

    def segmentation_50(self):
        df = self.df
        df = self.get_iou(df, "segm")
        df = self.get_by_th(df, "0.50")
        return self._as_result(df)


class ThesisResults(object):
    def __init__(self, ccr: CombinedCocoResults):
        self.ccr = ccr

    def __repr__(self):
        return self.ccr.df.to_string()

    def brightfield(self):
        return self._to_tr_coco(self.ccr.get(det_type="bf"))

    def fluorescence(self):
        return self._to_tr_coco(self.ccr.get(det_type="fl"))

    def segm(self):
        return self._to_tr_coco(self.ccr.get(iou_type="segm"))

    def bbox(self):
        return self._to_tr_coco(self.ccr.get(iou_type="bbox"))

    def sparse(self):
        return self._to_tr_coco(self.ccr.get(ds_type="sparse"))

    def dense(self):
        return self._to_tr_coco(self.ccr.get(ds_type="dense"))

    def _to_tr_coco(self, df):
        return ThesisResults(CombinedCocoResults(df, is_df=True))

    def ap_50(self):
        return self._to_tr_coco(self.ccr.get(th_type="0.50", metric="ap"))

    def ap_75(self):
        return self._to_tr_coco(self.ccr.get(th_type="0.75", metric="ap"))

    def ap(self):
        return self._to_tr_coco(self.ccr.get(th_type="0.50:0.95", metric="ap"))

    def recall(self):
        return self._to_tr_coco(self.ccr.get(th_type="0.50:0.95", metric="recall"))

    @staticmethod
    def from_folder(input_folder):
        data = []
        for folder in input_folder.get_folders():
            results = CocoResults(input_folder.make_sub_folder(folder, create=False))
            data.append(results.df)

        results = ThesisResults(CombinedCocoResults(data))
        return results

if __name__ == '__main__':

    target_folder_bf = "/Users/Dennis/Desktop/thesis/thesis_writing/ia_analysis_results/bf/results_bf_detection_sigmoid_instance_l1_adam_lr0002"
    base_folder = Folder("/Users/Dennis/Desktop/thesis/thesis_writing/ia_analysis_results/gt_evaluations")

    results = ThesisResults.from_folder(base_folder)

    results = results

    bbox = results.fluorescence().bbox()
    segm = results.brightfield().segm()
    if False:
        print("BBox")
        print("50", bbox.ap_50())
        print("75", bbox.ap_75())
        print(".5:.95", bbox.ap())
        print("Recall")
        print(bbox.recall())

        print("Segm")
        print("50", segm.ap_50())
        print("75", segm.ap_75())
        print(".5:.95", segm.ap())
        print("recall", segm.recall())

    # print(pd.DataFrame(d))


    data = [bbox.ap_50(), bbox.ap_75(), bbox.ap(), bbox.recall(), segm.ap_50(), segm.ap_75(), segm.ap(), segm.recall()]

    data = [d.ccr.df for d in data]
    data = pd.concat(data)
    data = segm.ccr.df
    print(data)
    data = data[data['loss'].isin(["bce", "l1"])]
    data = data[data['activation'].isin(["Sigmoid", "sigmoid"])]

    data['model'] = data['norm_type'] + "_" + data['activation'] + "_" + data['loss']
    data = data.drop(['activation', 'norm_type', 'loss'], 1)
    data = data[data['th_type'].isin(["0.50:0.95", "0.50"])]
    data = data.sort_values("score", ascending=False)
    data = data.rename(columns={"th_type": "@IoU", "iou_type": "Output", "metric": "Metric", 'ds_type': "Dataset"})
    data['Metric'] = data['Metric'].map({'ap': 'Precision', 'recall': 'Recall'})

    pivot = data.pivot_table(["score"], ['model', 'Metric', '@IoU' ]).reset_index()
    pivot = pivot.sort_values(['model', 'Metric', '@IoU', 'score'], ascending=False)
    # pivot = pivot.sort_values("score", ascending=False)
    # pivot = pivot.reindex(pivot.sort_values('score', ascending=False).index)
    print(pivot.columns)
    print(pivot.to_latex(index=False, float_format="%.3f"))

    # print(d)
    # print(data.pivot_table("score", ['det_type', 'activation', 'th_type','metric', 'norm_type', 'loss', 'ds_type', 'iou_type']))

    # data = data[data['th_type'] == "0.50:0.95"]
    # metric = data['ds_type'].unique()
    # print(metric)
    # sns.pairplot(data=data, hue="")
    # g = sns.FacetGrid(data=data, col="metric", row="dataset", hue="ds_type")
    #
    # g.map(sns.barplot, "ds_type", "score", order=metric)
    #
    # plt.show()

    vals_50_95 = []

    vals_50 = []
    vals_75 = []
