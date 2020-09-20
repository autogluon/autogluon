import mxnet as mx
import logging
import time

from .utils import param_to_pretty_string

logger = logging.getLogger(__name__)


class DebugGPRegression(object):
    """
    Supports finding errors in GaussianProcessRegression.fit. For each
    criterion evaluation, we store the input arguments before and the
    criterion value afterwards. Storage is done to a rolling sequence
    of local files.
    """
    def __init__(self, fname_msk='debug_gpr_{}', rolling_size=3):
        self.fname_msk = fname_msk
        self.rolling_size = rolling_size
        self.global_counter = 0
        self.optim_counter = -1
        self.local_counter = 0

    def start_optimization(self):
        self.optim_counter += 1
        self.local_counter = 0

    def store_args(self, params, X, Y, param_encoding_pairs):
        arg_dict = {
            'features': X,
            'targets': Y}
        for param in params:
            arg_dict[param.name] = param.data(ctx=mx.cpu())
        fname = self._filename()
        mx.nd.save(fname + '_args.nd', arg_dict)
        with open(fname + '_args.txt', 'w') as f:
            self._write_meta(f)
            for param, encoding in param_encoding_pairs:
                f.write(param_to_pretty_string(param, encoding) + '\n')

    def store_value(self, value):
        fname = self._filename()
        with open(fname + '_value.txt', 'w') as f:
            f.write('value = {}\n'.format(value))
            self._write_meta(f)
        # Advance counters
        self.global_counter += 1
        self.local_counter += 1

    def _filename(self):
        return self.fname_msk.format(self.global_counter % self.rolling_size)

    def _write_meta(self, f):
        f.write('optim_counter = {}\n'.format(self.optim_counter))
        f.write('local_counter = {}\n'.format(self.local_counter))
        f.write('global_counter = {}\n'.format(self.global_counter))
        f.write('time = {}\n'.format(time.time()))
