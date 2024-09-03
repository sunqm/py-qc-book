import json
from base64 import b64encode, b64decode
import numpy as np

class NpArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'python/ndarray': b64encode(obj.tobytes()).decode(),
                    'shape': obj.shape}
        return super().default(obj)

def np_decoder(obj):
    if 'python/ndarray' in obj:
        return np.frombuffer(b64decode(obj['python/ndarray'])).reshape(obj['shape'])
    return obj

if __name__ == '__main__':
    arr = np.arange(6.).reshape(2, 3)
    dat_encoded = json.dumps({'a': arr, 'b': [2, 3, 4]}, cls=NpArrayEncoder)
    print(dat_encoded)

    dat_decoded = json.loads(dat_encoded, object_hook=np_decoder)
    print(dat_decoded)
