import os
import sys

import pandas as pd
from utils.file_management.image_set import ImageSet
from utils.models.dot_dict import DotDict
from utils.helper_functions.filesystem_utils import split_filename_extension


class MetaData(DotDict):
    def __init__(self, extraction_strategy):
        super().__init__()
        self.src_strategy = extraction_strategy

    def add(self, key, val):
        self[key] = val


class MetaDataMismatchException(ValueError):
    def __init__(self, cause):
        super(MetaDataMismatchException, self).__init__(cause)


class MetadataParser(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def extract_from(self, file_provider):
        results = []
        for file in file_provider:
            try:
                results.append(self.strategy.get_meta_data(file))
            except MetaDataMismatchException as e:
                print("Strategy", self.strategy, "not applicable to", file, e)

        return results


class FileNamingStrategy(object):
    REQUIRED_KEYS = ['patch_id']
    INTERNAL_KEYS = {'src_strategy', 'base.', 'o-id'}
    KEY_CHANNEL = "channel"

    def find_image_sets(self, base_folder, file_extensions=['tif', 'tiff', 'png'], subdirs=False):
        parser = MetadataParser(self)
        fp = base_folder.make_file_provider(extensions=file_extensions, include_subdirs=subdirs)
        metadata = parser.extract_from(fp)
        
        image_sets = self.identify_image_sets(metadata)

        return image_sets

    def __init__(self, strategy_name, channel_identifier):
        self.strategy_name = strategy_name
        self.base_group_by = ['base.folder_name', 'base.folder_path']
        self.chan_identifier = channel_identifier

    def split_path_fn(self, abs_path):
        tail, head = os.path.split(abs_path)
        return tail, head

    def remove_keys(self, removes, keys):
        for key in removes:
            self.remove_key_if_present(key, keys)

    def remove_key_if_present(self, key, keys):
        if key in keys:
            keys.remove(key)

    def _get_grouping_keys(self, pd_data_batch_frame):
        raise NotImplementedError("base")

    def _parse_attributes(self, fn, meta_data):
        raise NotImplementedError("base")

    def _key_value_to_string(self, key, value):
        raise NotImplementedError("base")

    def _add_base_attributes_to_file_name(self, name, meta):
        ext = meta['base.extension'].item()
        if not ext.startswith("."):
            ext = "." + ext
        return name + ext

    def get_file_names(self, image_set):
        return self._image_set_to_file_names(image_set)

    def parse_base_attributes(self, base_path, fn, meta):
        name_no_ext, extension = split_filename_extension(fn)
        path_parts = base_path.rstrip().split("/")

        meta["base.folder_name"] = path_parts[-1]
        meta["base.folder_path"] = os.path.join(base_path)
        meta["base.file_path"] = os.path.join(base_path, fn)
        meta["base.extension"] = extension
        meta["base.fn"] = fn
        return name_no_ext

    def _group_frame(self, pd_frame, group_by_arr):
        if not isinstance(group_by_arr, list):
            group_by_arr = [group_by_arr]
        return pd_frame.groupby(self.base_group_by + group_by_arr)

    def get_meta_data(self, abs_path):
        data = MetaData(self.strategy_name)
        tail, head = self.split_path_fn(abs_path)
        try:
            name_no_ext = self.parse_base_attributes(tail, head, data)
            self._parse_attributes(name_no_ext, data)
        except Exception:
            cause = "Unexpected error: " + str(sys.exc_info())
            raise MetaDataMismatchException(cause)
        return data

    def identify_image_sets(self, metadata_list):
        frame = pd.DataFrame(metadata_list)
        available_keys = frame.columns.values.tolist()
        available_keys = [k for k in available_keys if
                          not k.startswith("base.") and k != self.chan_identifier and k not in self.INTERNAL_KEYS]
        available_keys = self._get_grouping_keys(available_keys)
        grouped = self._group_frame(frame, available_keys)
        imagesets = []
        invalids = []
        for name, image_group in grouped:

            if any(image_group[self.chan_identifier].isna()):
                print("Unable to assign channels for grouped set", image_group['base.file_path'])
                invalids.append(image_group)
                continue

            im_set = ImageSet(self)
            for key in image_group[self.chan_identifier]:
                channel = image_group.loc[image_group[self.chan_identifier] == key]
                im_set.add_channel(key, channel)
            imagesets.append(im_set)

        return imagesets

    def _channel_dict_to_file_name(self, channel_name, channel_dict):
        raise NotImplementedError("base")

    def _image_set_to_file_names(self, image_set):
        channel_dict = image_set.channels
        names = DotDict()
        for channel_name in channel_dict:
            chan = channel_dict[channel_name]
            fn = self._channel_dict_to_file_name(channel_name, chan)
            names[channel_name] = self._add_base_attributes_to_file_name(fn, chan)

        channel_dict = image_set.masks

        for channel_name in channel_dict:
            chan = channel_dict[channel_name]
            fn = self._channel_dict_to_file_name(channel_name, chan)
            names[channel_name] = self._add_base_attributes_to_file_name(fn, chan)

        return names


class CsvNamingStrategy(FileNamingStrategy):
    CHAN_KEY = "type"

    def __init__(self):
        super().__init__("CsvNamingStrategy", self.CHAN_KEY)

        self.group_by = self.CHAN_KEY
        self.k_v_delimiter = "="
        self.attr_delimiter = "_"

    def _get_grouping_keys(self, frame_columns):
        return frame_columns

    def _parse_attributes(self, fn, meta_data):

        k_v_arr = fn.split(self.attr_delimiter)
        for k_v in k_v_arr:
            kv_split = k_v.split(self.k_v_delimiter)
            if len(kv_split) == 1:
                raise MetaDataMismatchException("Split went wrong for node: " + str(kv_split) + " @ " + fn)

            key, value = kv_split
            meta_data[key] = value

    def _key_value_to_string(self, key, value):
        return key + self.k_v_delimiter + value

    def _channel_dict_to_file_name(self, channel_name, channel_dict):
        attrs_keys = [k for k in channel_dict.keys() if "base." not in k and k not in self.INTERNAL_KEYS]
        attrs = []
        for key in sorted(attrs_keys, reverse=False):
            val = channel_dict[key].item()
            attrs.append(self.k_v_delimiter.join([key, val]))

        return self.attr_delimiter.join(attrs)


class ZtzCsvNaming(CsvNamingStrategy):
    CHAN_KEY = "channel"

    def __init__(self):
        super().__init__()
        self.attr_delimiter = ","
        self.k_v_delimiter = "_"
        self.attr_mapping = {'w': "w", 't': "ex", "s": "cp"}
        self.group_keys = ['cp', 'dte', 'w', 'z']

    def _get_grouping_keys(self, frame_columns):
        return self.group_keys

    def _parse_attributes(self, fn, meta_data):
        if "s_2_1" in fn:
            fn = fn.replace("s_2_1", "s_2.1")
        if "s_2_b" in fn:
            fn = fn.replace("s_2_b", "s_2,iinfo_b")
        if "s_2.1_b" in fn:
            fn = fn.replace("s_2.1_b", "s_2.1,iinfo_b")

        attrs = fn.split(self.attr_delimiter)
        for att in attrs:
            kv_split = att.split(self.k_v_delimiter)
            if len(kv_split) != 2:
                raise MetaDataMismatchException("Split went wrong for node: " + str(kv_split) + " @ " + fn)
            k, v = kv_split
            if k in self.attr_mapping:
                k = self.attr_mapping[k]
            meta_data[k] = v
        if "year" in meta_data and "month" in meta_data and "day" in meta_data:
            meta_data['dte'] = meta_data['year'][:2] + meta_data['month'].zfill(2) + meta_data['day'].zfill(2)
            del meta_data['year']
            del meta_data['month']
            del meta_data['day']
        if "iinfo" in meta_data:
            img_info = meta_data['iinfo']
            if "c1" in img_info:
                meta_data[self.CHAN_KEY] = "dic"
            elif "c0" in img_info:
                meta_data[self.CHAN_KEY] = "outline"
            else:
                raise MetaDataMismatchException("Unknown image type: " + img_info + " @ " + fn)
            meta_data['pl'] = meta_data["iinfo"].split("m")[-1]
            del meta_data['iinfo']


class ZtzRawDataStrategy(FileNamingStrategy):
    CHANNEL_IDENTIFIER = "type"

    def _channel_dict_to_file_name(self, channel_name, channel_dict):
        vals = []
        for v_key in self._meta_keys + self.REQUIRED_KEYS:
            if v_key in channel_dict:
                val = channel_dict[v_key].item()
                if v_key in {'well', 'plate', 'z-idx', 'time'}:
                    val = self._add_init_letter_and_zeros(v_key, val)
                vals.append(val)

        fn = "--".join(vals)
        return fn

    def _key_value_to_string(self, key, value):
        return value

    def __init__(self):
        super().__init__("ZtzRawDataStrategy", self.CHANNEL_IDENTIFIER)
        self._meta_keys = ['lane', 'well', 'plate', 'z-idx', 'time', self.CHANNEL_IDENTIFIER]
        self.group_by = self._meta_keys[:-1]

    def _get_grouping_keys(self, pd_data_batch_frame):
        return self.group_by

    def _add_init_letter_and_zeros(self, key, val):
        val = key[0].upper() + val.zfill(5)
        return val

    def _remove_init_letter_and_zeros(self, val):
        if val[0].isalpha():
            val = val[1:]

        while val[0] == "0" and len(val) > 1:
            val = val[1:]

        return val

    def _parse_attributes(self, fn, meta_data):

        split = fn.split("--")

        if len(self._meta_keys) != len(split):
            raise MetaDataMismatchException(
                "Expected {} keys but split had {}".format(len(self._meta_keys), len(split)))

        for idx in range(len(split)):
            key = self._meta_keys[idx]

            val = split[idx]
            if key not in {'lane', self.CHANNEL_IDENTIFIER}:
                val = self._remove_init_letter_and_zeros(val)
            meta_data.add(key, val)
