import pandas as pd

from utils.models.dot_dict import DotDict
from utils.models.folder import Folder

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)


class ModelValidation(object):
    def __init__(self, root: Folder):
        self.root = root
        self.validation_files = self.get_validation_files()
        self.name, self.attr_dict = self.get_model_name()

    def short_name(self):
        return f"{self.attr_dict.exp_name}: {self.attr_dict.activation} - {self.attr_dict.norm_type} - {self.attr_dict.loss}"

    def unique_name(self):
        return f"{self.attr_dict.activation}|{self.attr_dict.norm_type}|{self.attr_dict.loss}|{self.attr_dict.batch_size}"

    def __repr__(self):
        return f"{self.short_name()} | {len(self.validation_files)}"

    def get_model_name(self):
        name = self.root.name
        attrs = name.split("-")
        name_dict = DotDict()

        name_dict.exp_name = attrs[0]
        attrs = attrs[1:]
        for attr in attrs:
            key, value = attr.split("=")
            name_dict[key] = value

        return name, name_dict

    def _add_model_params(self, data):
        data['model_id'] = self.unique_name()
        data['activation'] = self.attr_dict.activation
        data['batch_size'] = self.attr_dict.batch_size
        data['loss'] = self.attr_dict.loss
        data['norm'] = self.attr_dict.norm_type
        return data

    def get_validation_files(self):
        return list(self.root.make_file_provider(extensions=["csv"], contains="validation_results"))

    def load_validation(self):

        validations = []
        for file_path in self.validation_files:
            data = pd.read_csv(file_path)
            ds_name = data['dataset_id'].iloc[0]

            ds_type = 'none'
            if 'train' in ds_name.lower():
                ds_type = 'train'
            elif 'test' in ds_name.lower():
                ds_type = 'test'
            elif 'val' in ds_name.lower():
                ds_type = 'val'
            else:
                raise ValueError("Unknown dataset type", ds_name)

            data['ds_type'] = ds_type
            validations.append(data)

        if len(validations) == 0:
            print("No validations:", self.name)
            return None
        return pd.concat(validations)

    def process_validation(self):
        vals = self.load_validation()
        if vals is None:
            return vals
        vals = vals.groupby(['dataset_id', 'ds_type']).agg(
            {'l1': ['mean', 'std', 'min', 'max'], 'l2': ['mean', 'std', 'min', 'max'], 'ssim': ['mean', 'std', 'min', 'max']}).reset_index()

        # vals.columns = vals.columns.to_flat_index()
        def join_cols(inp):
            if len(inp[1]) == 0:
                return inp[0]
            return "_".join(inp)

        vals.columns = vals.columns.map(join_cols)
        vals = self._add_model_params(vals)

        # print(vals[['ssim', 'max']])
        return vals

    def get_min_max(self, mode="ssim"):
        data = self.load_validation()
        grouped = data.groupby(['dataset_id'])
        mins = []
        maxes = []

        # print(grouped['l1'].describe())
        # print(grouped['l2'].describe())
        for idx, g in grouped:
            mins.append(g.iloc[g[mode].idxmin()])
            maxes.append(g.iloc[g[mode].idxmax()])

        mins = pd.DataFrame(mins)

        maxes = pd.DataFrame(maxes)
        return mins, maxes

    def get_min_max_images(self, mode="ssim"):
        mins, maxes = self.get_min_max(mode)

        mins = mins[['dataset_id', 'img', mode]]
        maxes = maxes[['dataset_id', 'img', mode]]
        return mins, maxes


class ValidationResultViewer(object):
    def __init__(self, cp_root):
        self.models = self.parse(cp_root)

    def parse(self, cp_root: Folder):
        models = cp_root.get_folders(abs_path=False)

        data_list = []

        for model in models:
            data_list.append(ModelValidation(cp_root.make_sub_folder(model)))
        return data_list

    def get_full_results(self):
        model_df = [model.process_validation() for model in self.models]
        model_df = list(filter(lambda x: x is not None, model_df))

        return pd.concat(model_df)

    def results_table(self):
        return to_latex_table(self.get_full_results())

    def print_ds_stats(self):
        model_df = self.get_full_results()


def to_latex_table(df):
    by_model_dstype = df.groupby(['activation', 'norm', 'loss', 'ds_type']).agg(
        {
            'l1_mean': 'mean',
            'l2_mean': 'mean',
            'ssim_mean': 'mean'
        }
    ).reset_index()

    df = df[df['ds_type'] != 'train']

    by_model = df.groupby(['activation', 'loss', 'norm']).agg(
        {
            'l1_mean': 'mean',
            'l2_mean': 'mean',
            'ssim_mean': 'mean'
        }
    ).reset_index()

    print(by_model)

    return by_model.to_latex(index=False, float_format="{:0.4f}".format)


if __name__ == '__main__':
    cps = Folder("/Users/Dennis/Desktop/checkpoints_final_runs")
    vv = ValidationResultViewer(cps)
    model = vv.models[0]
    print(model)

    vals = model.process_validation()
    vals2 = vv.models[1].process_validation()
    vals = pd.concat([vals, vals2])
    print(vals)

    # vals.plot(kind="bar", x="dataset_id", y="ssim_mean", yerr="ssim_std")
    import matplotlib.pyplot as plt
    import seaborn as sns

    vals = vv.get_full_results()

    valid = vals[vals['ssim_mean'] > 0.7]['model_id']
    vals = vals[vals['model_id'].isin(valid)]

    order = sorted(vals['model_id'].unique())
    print("total", len(order))
    sns.lineplot(data=vals, x="dataset_id", y="ssim_mean", hue="model_id", hue_order=order)
    plt.xticks(rotation=30)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=0, borderaxespad=0.)
    plt.tight_layout()
    plt.show()


    vals = vv.get_full_results()


    valid = vals[vals['ssim_mean'] > 0.7]['model_id']
    vals = vals[vals['model_id'].isin(valid)]

    order = sorted(vals['model_id'].unique())
    print("total", len(order))
    sns.barplot(data=vals, x="norm", y="ssim_mean", hue="dataset_id")
    plt.xticks(rotation=30)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=0, borderaxespad=0.)
    plt.tight_layout()
    plt.show()




    #
    # vals = vv.get_full_results()
    # print(to_latex_table(vals))

    #
    # import matplotlib.style as style
    #
    # print(style.available)
    # # style.use('seaborn-paper')  # sets the size of the charts
    # # style.use('ggplot')
    # # sns.barplot(data=vals, x='dataset_id', y="l1_mean", yerr="l1_std")
    # import seaborn as sns
    #
    # # sns.barplot(data=vals, x='dataset_id', y="ssim_mean", hue="model_id")
    # print(vals)
    #
    # # vals.plot(kind="bar", x='dataset_id', y="l1_mean", yerr="l1_std", table=True)
    # # vals.plot(table)
    # # print(vals['l1_mean'].map('{:.4f}'.format))
    # # test = vals[['l1_mean']].map('{0:.2f}'.format)
    # data = vals[['dataset_id', 'model_id', 'l1_mean', 'l2_mean']]
    #
    # plt.table(cellText=data[['l1_mean', 'l2_mean']].to_numpy(), colWidths=[0.25] * len(vals.columns),
    #           rowLabels=data['dataset_id'],
    #           colLabels=['l1_mean', 'l2_mean'],
    #           cellLoc='center', rowLoc='center',
    #           loc='center')
    #
    # sns.despine()
    #
    # plt.show()
