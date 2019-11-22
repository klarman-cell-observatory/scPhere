# ==============================================================================
"""The exponentially scaled modified Bessel function of the first kind,
from the hyperspherical_vae package """

import numpy as np
import scipy.special

from tensorflow.python.ops import script_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.custom_gradient import custom_gradient


@custom_gradient
def ive(v, z):
    """Exponentially scaled modified Bessel function of the first kind."""
    output = array_ops.reshape(script_ops.py_func(
        lambda v, z: np.select(condlist=[v == 0, v == 1],
                               choicelist=[scipy.special.i0e(z, dtype=z.dtype),
                                           scipy.special.i1e(z, dtype=z.dtype)],
                               default=scipy.special.ive(v, z, dtype=z.dtype)),
        [v, z], z.dtype),
        ops.convert_to_tensor(array_ops.shape(z), dtype=dtypes.int32))

    def grad(dy):
        return None, dy * (ive(v - 1, z) - ive(v, z) * (v + z) / z)

    return output, grad
