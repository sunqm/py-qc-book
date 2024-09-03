import json
import numpy as np

class NpArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'python/ndarray': obj.tolist()}
        return super().default(obj)

def np_decoder(obj):
    if 'python/ndarray' in obj:
        return np.array(obj['python/ndarray'])
    return obj

if __name__ == '__main__':
    arr = np.arange(6.).reshape(2, 3)
    dat_encoded = json.dumps({'a': arr, 'b': [2, 3, 4]}, cls=NpArrayEncoder)
    print(dat_encoded)

    dat_decoded = json.loads(dat_encoded, object_hook=np_decoder)
    print(dat_decoded)
