from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter


# DEFAULT_LOG_DIR = get_root_folder().make_sub_folder("tb_logs")
# print("Tensorboard log dir", DEFAULT_LOG_DIR)


def get_tb_writer(run_id):
    print(run_id)
    # target_dir = DEFAULT_LOG_DIR.make_sub_folder(run_id)

    return SummaryWriter(comment=run_id, flush_secs=30)


def persist_model_state(sw: SummaryWriter, model: Module, epoch):
    print(epoch)
    for name, weight in model.named_parameters():
        sw.add_histogram(name, weight, epoch)
        sw.add_histogram('f{name}.grad', weight.grad, epoch)
    sw.flush()


def add_model_graph(sw: SummaryWriter, model: Module, example_data):
    sw.add_graph(model, example_data, verbose=False)
    sw.flush()
