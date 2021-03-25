# ==============================================================================
"""The wrapped norm distribution on hyperbolic space."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow_probability.python import distributions

import tensorflow_probability as tfp
from tensorflow.python.ops import linalg_ops

__all__ = [
    "HyperbolicWrappedNorm",
]


class HyperbolicWrappedNorm(distributions.Distribution):
    """The hyperbolic wrapped normal distribution with location `loc`
    and scale `scale`.
    #### Mathematical details
    """

    def __init__(self, loc, scale, validate_args=False, allow_nan_stats=True,
                 name="Hyperbolic-wrapped-norm"):
        """Construct hyperbolic wrapped normal distributions with mean of 'loc'
        and scale of `scale`.

        Args:
          loc: Floating point tensor; the mean of the distribution(s).
          scale: Floating point tensor; the concentration of the distribution(s).
            Must contain only non-negative values.
          validate_args: Python `bool`, default `False`. When `True` distribution
            parameters are checked for validity despite possibly degrading runtime
            performance. When `False` invalid inputs may silently render incorrect
            outputs.
          allow_nan_stats: Python `bool`, default `True`. When `True`,
            statistics (e.g., mean, mode, variance) use the value "`NaN`" to
            indicate the result is undefined. When `False`, an exception is raised
            if one or more of the statistic's batch members are undefined.
          name: Python `str` name prefixed to Ops created by this class.

        Raises:
          TypeError: if `loc` and `scale` have different `dtype`.
        """
        parameters = locals()
        with ops.name_scope(name, values=[loc, scale]):
            with ops.control_dependencies(
                    [check_ops.assert_positive(scale),
                     check_ops.assert_near(linalg_ops.norm(loc, axis=-1),
                                           1, atol=1e-5)] if validate_args else []):
                self._loc = array_ops.identity(loc, name="loc")
                self._scale = array_ops.identity(scale, name="scale")
                check_ops.assert_same_float_dtype([self._loc, self._scale])

        super(HyperbolicWrappedNorm, self).__init__(
            dtype=self._scale.dtype,
            reparameterization_type=distributions.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            graph_parents=[self._loc, self._scale],
            name=name)

        self._base_dist = tfp.distributions.Normal(loc=tf.zeros_like(self._scale),
                                                   scale=self._scale)
        self._dim = tf.shape(self._loc)[1] - 1

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(zip(("loc", "scale"),
                        ([ops.convert_to_tensor(sample_shape, dtype=dtypes.int32),
                          ops.convert_to_tensor(sample_shape[:-1].concatenate([1]),
                                                dtype=dtypes.int32)])))

    @property
    def loc(self):
        """Distribution parameter for the mean."""
        return self._loc

    @property
    def scale(self):
        """Distribution parameter for concentration."""
        return self._scale

    def _batch_shape_tensor(self):
        return array_ops.broadcast_dynamic_shape(
            array_ops.shape(self._loc),
            array_ops.shape(self._loc))

    def _batch_shape(self):
        return array_ops.broadcast_static_shape(
            self._loc.get_shape(),
            self._loc.get_shape())

    def _event_shape_tensor(self):
        return constant_op.constant([], dtype=dtypes.int32)

    def _event_shape(self):
        return tensor_shape.scalar()

    def _sample_n(self, n, seed=0):
        shape = array_ops.concat([[n], array_ops.shape(self._scale)], 0)
        zn = random_ops.random_normal(shape, mean=0., stddev=1.,
                                      dtype=self.dtype, seed=seed)
        zn *= self._scale

        shape1 = [n, self.batch_shape_tensor()[0], 1]
        z0 = tf.concat([tf.zeros(shape1), zn], axis=-1)

        loc0 = self._lorentzian_orig(shape1, shape)
        tmp = tf.expand_dims(self._loc, 0)

        shape2 = [n, 1, 1]
        zt = self._parallel_transport(z0, loc0, tf.tile(tmp, shape2))
        z = self._exp_map(zt, tf.tile(tmp, shape2))

        return z

    @staticmethod
    def _lorentzian_orig(s1, s0):
        x1 = tf.ones(s1)
        x0 = tf.zeros(s0)

        x_orig = tf.concat([x1, x0], axis=-1)

        return x_orig

    @staticmethod
    def _clip_min_value(x, eps=1e-6):
        return tf.nn.relu(x - eps) + eps

    def _exp_map(self, x, mu):
        res = self._lorentzian_product(x, x)
        res = tf.math.sqrt(self._clip_min_value(res))

        res = tf.clip_by_value(res, 0, 32)

        return tf.math.cosh(res) * mu + tf.math.sinh(res) * x / res

    @staticmethod
    def _lorentzian_product(x, y):
        y0, y1 = tf.split(y, [1, y.shape.as_list()[-1] - 1], axis=-1)
        y_neg_first = tf.concat([-y0, y1], axis=-1)

        return tf.reduce_sum(tf.multiply(x, y_neg_first), -1, keepdims=True)

    def _parallel_transport(self, x, m1, m2):
        alpha = -self._lorentzian_product(m1, m2)
        coef = self._lorentzian_product(m2, x) / (alpha + 1.0)

        return x + coef * (m1 + m2)

    def _lorentz_distance(self, x, y):
        res = -self._lorentzian_product(x, y)
        res = self._clip_min_value(res, 1+1e-6)

        z = tf.sqrt(res+1) * tf.sqrt(res-1)

        return tf.math.log(res + z)

    def _inv_exp_map(self, x, mu):
        alpha = -self._lorentzian_product(x, mu)
        alpha = self._clip_min_value(alpha, 1+1e-6)

        tmp = self._lorentz_distance(x, mu) / \
            tf.sqrt(alpha+1) / tf.sqrt(alpha-1)

        return tmp * (x - alpha * mu)

    def _log_prob(self, x):
        v = self._inv_exp_map(x, self._loc)
        tmp = self._lorentzian_product(v, v)
        x_norm = tf.math.sqrt(self._clip_min_value(tmp))

        x_norm = tf.clip_by_value(x_norm, 0, 32)
        res = (tf.cast(self._dim, dtype=tf.float32) - 1.0) * \
            tf.math.log(tf.math.sinh(x_norm) / x_norm)

        shape = array_ops.shape(self._scale)
        shape1 = [self.batch_shape_tensor()[0], 1]

        loc0 = self._lorentzian_orig(shape1, shape)
        v1 = self._parallel_transport(v, self._loc, loc0)
        xx = v1[..., 1:]
        log_base_prob = tf.reduce_sum(self._base_dist.log_prob(xx), -1)
        log_base_prob = array_ops.reshape(log_base_prob, shape1)

        return log_base_prob - res

    def _prob(self, x):
        return math_ops.exp(self._log_prob(x))
