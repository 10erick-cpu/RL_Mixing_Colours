import numpy as np
from models.dot_dict import DotDict

target_path = "/home/mrd/Desktop/pytorch-caffe/2d_cell_net_v0.caffemodel.h5"
target_path_modeldef = "/home/mrd/Desktop/pytorch-caffe/2d_cell_net_v0.modeldef.h5"

import h5py


class Layer(object):

    def __init__(self, data):
        self.data = data
        name = data['name']
        self.type = name.split("_")[0]
        if self.type == "conv":
            level_target = name.split("-")
            self.level = level_target[0][-1:]
            self.id = level_target[0][-3:-1]
            self.target = level_target[1]

        elif self.type == "upconv":
            split = name.split("_")
            self.id, self.level = split[1][:2], split[1][2]
            self.target = split[2]
        else:
            split = name.split("_")
            if len(split) == 1:
                self.id, self.level, self.target = "", "", ""
                return
            self.id, self.level = split[1][:2], split[1][2]
            self.target = ""
            pass
            # warnings.("Unknown type " + self.type)

    def build_id(self):
        return self.type + "_" + self.id + self.level + "-" + self.target

    def __repr__(self):
        return self.build_id() + " " + str(self.data.weights.shape) + " b: " + str(self.data.bias.shape)


def export_model_definition():
    f = h5py.File(target_path_modeldef, 'r')

    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())

    model = f.get('model_prototxt')

    with open('model.prototxt', 'w') as f:
        f.write(str(model[()].decode("UTF-8")))
        f.close()


def extract_weights_biases(data):
    datadict = DotDict()
    unused = []
    for key in sorted(data.keys()):
        # print(key)
        elem = data[key]
        num_keys = len(elem.keys())
        if num_keys > 0:
            if num_keys == 2:
                weights = np.array(elem["0"])
                bias = np.array(elem["1"])

                # print("\tWeight: ", weights.shape)
                # print("\tBias: ", bias.shape)
                assert key not in datadict
                datadict[key] = DotDict()
                datadict[key].weights = weights
                datadict[key].bias = bias
                datadict[key].name = key

            else:
                print("Unkown element with num_keys", num_keys)
                print(key)


        else:
            unused.append({'name': key})
    return datadict, unused


def export_model_weights(file_path):
    f = h5py.File(file_path, 'r')

    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]
    print()
    data = f[a_group_key]

    datadict, unused = extract_weights_biases(data)
    # for u in unused:
    #    print(u.name.replace("/data/", ""))
    # print("num layers: ", len(datadict))
    # print(sorted(datadict.keys()))

    size_sorted = dict()

    for key in datadict.keys():
        elem = datadict[key]
        in_channel = elem.weights.shape[0]
        if in_channel not in size_sorted:
            size_sorted[in_channel] = dict()
            size_sorted[in_channel]['up'] = []
            size_sorted[in_channel]['down'] = []

        if 'upconv' in key:
            size_sorted[in_channel]['up'].append(elem)
        else:
            size_sorted[in_channel]['down'].append(elem)

    total = 0
    for key in size_sorted:
        len_up = len(size_sorted[key]['up'])
        len_down = len(size_sorted[key]['down'])
        total += len_up
        total += len_down

    assert len(datadict) == total

    down = []
    up = []
    for size_bucket in size_sorted.keys():
        bucket = size_sorted[size_bucket]

        down += bucket['down']
        up += bucket['up']

    down = sorted(down, key=lambda x: x.name)
    up = sorted(up, key=lambda x: x.name)

    layers = []
    for elem in down + up:
        layers.append(Layer(elem))

    layers = sorted(layers, key=lambda x: (x.id, x.level, x.target))
    layer_dict = dict()
    for layer in layers:
        assert layer.build_id() not in layer_dict
        layer_dict[layer.build_id()] = layer

    return layer_dict


def ordered_architecture_from_layer_dict(id_arr, layer_dict):
    result = []
    for id in id_arr:
        result.append(layer_dict[id])
    return result


def get_architecture():
    id_arr = [
        'conv_d0a-b',
        'conv_d0b-c',
        'conv_d1a-b',
        'conv_d1b-c',
        'conv_d2a-b',
        'conv_d2b-c',
        'conv_d3a-b',
        'conv_d3b-c',
        'conv_d4a-b',
        'conv_d4b-c',
        'upconv_d4c-u3a',
        'conv_u3b-c',
        'conv_u3c-d',
        'upconv_u3d-u2a',
        'conv_u2b-c',
        'conv_u2c-d',
        'upconv_u2d-u1a',
        'conv_u1b-c',
        'conv_u1c-d',
        'upconv_u1d-u0a',
        'conv_u0b-c',
        'conv_u0c-d',
        'conv_u0d-score'
    ]

    return ordered_architecture_from_layer_dict(id_arr, export_model_weights(target_path))


print()
print("#" * 10 + "ARCHITECTURE" + "#" * 10)
print()
for e in get_architecture():
    print("#" * int(50 * (e.data.weights.shape[0] / 1024)))
    print(e)
print()

# store = HDFStore(target_path)
# print(list(store.keys()))
