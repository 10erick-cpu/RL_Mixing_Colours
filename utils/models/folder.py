import os
from os.path import join, isfile

from utils.helper_functions.filesystem_utils import has_file_allowed_extension, is_folder


class Folder(object):
    """docstring for Folder"""

    def __init__(self, absolute_path, parent=None, create=False):
        super(Folder, self).__init__()
        self.path_abs = os.path.abspath(absolute_path)
        self.parent = parent
        self.parent_abs_path, self.name = os.path.split(self.path_abs)
        if not self.exists() and create:
            self.make_dir()

    @staticmethod
    def from_python_file(file):
        return Folder(os.path.dirname(os.path.abspath(file)))

    def parent_folder(self):
        return Folder(self.parent_abs_path)

    def exists(self):
        return os.path.exists(self.path_abs)

    def exists_sub_folder(self, sub_f_name):
        return os.path.exists(self.get_file_path(sub_f_name))

    def make_dir(self):
        if self.exists() is False:
            os.makedirs(self.path_abs)

    def get_file_path(self, file_name):
        return self.build_path(self.path_abs, file_name)

    def make_sub_folder(self, subf_name, create=True):
        folder = Folder(self.build_path(self.path_abs, subf_name))
        if create:
            folder.make_dir()
        return folder

    def __repr__(self):
        return self.path_abs

    def build_path(self, p1, p2):
        return os.path.join(p1, p2)

    def create(self, path, name):
        path_abs = os.path.join(path, name)
        return Folder(path_abs)

    def path(self):
        return self.path_abs

    def exists_path_in_folder(self, path):
        return os.path.exists(self.build_path(self.path_abs, path))

    def get_files(self, filter_extensions=None, abs_path=True):
        result = []
        for f in os.listdir(self.path_abs):
            if isfile(join(self.path_abs, f)):
                if filter_extensions is not None and not has_file_allowed_extension(f, filter_extensions):
                    continue
                else:
                    result.append(join(self.path_abs, f) if abs_path else f)

        return result

    def get_folders(self, abs_path=True):
        if not abs_path:
            return [f for f in os.listdir(self.path_abs) if not isfile(join(self.path_abs, f))]
        return [join(self.path_abs, f) for f in os.listdir(self.path_abs) if not isfile(join(self.path_abs, f))]

    def make_file_provider(self, extensions=None, contains=None, include_subdirs=False):
        return FilteredFileProvider(self.path_abs, extensions, contains, include_subdirs)

class DataProvider(object):

    def __len__(self):
        raise NotImplementedError("Base class")

    def __getitem__(self, item):
        raise NotImplementedError("Base class")


class FilteredFileProvider(DataProvider, Folder):
    DEFAULT_FILTERED = ['.DS_STORE']

    def matches_filters(self, fn):
        if any([fn.lower().endswith(f.lower()) for f in self.DEFAULT_FILTERED]):
            return False
        contain_filter_matched = True if not self.contain_filter else all(f in fn for f in self.contain_filter)

        extensions_matched = True if not self.extension_filter else has_file_allowed_extension(fn,
                                                                                               self.extension_filter)

        match = contain_filter_matched and extensions_matched
        if not match and self.log:
            print("File rejected: ", fn)
        return match

    def filter_files(self, root_dir, filter_subdirs):
        if not is_folder(root_dir):
            raise ValueError("root dir is not a folder", root_dir)

        data = []
        for root, dirs, files in os.walk(root_dir):

            for dir in dirs:
                os.path.join(root, dir)
            for file in files:
                data.append(os.path.join(root, file))
            if not filter_subdirs:
                break

        result = list(filter(lambda fn: self.matches_filters(fn), data))
        print(root_dir, "{} of {} files accepted".format(len(result), len(data)))
        return sorted(result)

    def __init__(self, abs_path, f_extension, contains=None, filter_subdirs=False, log=False):
        super().__init__(abs_path)
        self.log = log
        self.contain_filter = contains if isinstance(contains, list) else [contains] if contains is not None else None
        self.extension_filter = f_extension if isinstance(f_extension, list) else [
            f_extension] if f_extension is not None else None
        self.filtered = self.filter_files(self.path_abs, filter_subdirs)

    def __len__(self):
        return len(self.filtered)

    def __getitem__(self, item):
        return self.filtered[item]


class DicFileProvider(FilteredFileProvider):
    def __init__(self, abs_path, f_extension, filter_subdirs=False):
        super().__init__(abs_path, f_extension, contains=['DIC'], filter_subdirs=filter_subdirs)


class MultiChannelFileProvider(DicFileProvider):
    DIC_IDENTIFIER = "DIC"

    def __init__(self, abs_path, f_extension, filter_subdirs=False, required_extensions=None):
        self.required_extensions = required_extensions
        super().__init__(abs_path, f_extension, filter_subdirs)

    def make_required_files(self, dic_file):
        return [dic_file.replace(MultiChannelFileProvider.DIC_IDENTIFIER, ext) for ext in self.required_extensions]

    def matches_filters(self, fn):
        matches_super = super(MultiChannelFileProvider, self).matches_filters(fn)
        if not matches_super:
            return matches_super

        exist_all_extension_files = True
        for fname in self.make_required_files(fn):
            if not os.path.exists(fname):
                exist_all_extension_files = False
                print("Missing ext file {}, ignoring element")
                break
        return matches_super and exist_all_extension_files

    def __getitem__(self, item):
        dic = self.filtered[item]
        result = self.make_required_files(self.filtered[item])
        assert dic in result
        return result

