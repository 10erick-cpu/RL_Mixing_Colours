class DotDict(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    def get_or_default(self, key, value):
        if key not in self:
            self[key] = value
        return self[key]



@staticmethod
def from_json(json):
    result = DotDict()
    for k in json:
        result[k] = json[k]
    return result
