from abc import ABC, abstractmethod
from typing import Optional, Tuple, Sequence

import numpy as np
import tensorflow as tf


class MaskGenerator(ABC):
    def __init__(
        self,
        seed=None,
        dtype=np.float32,
    ):
        self._rng = np.random.RandomState(seed=seed)
        self._dtype = dtype

    def __call__(self, shape):
        return self.call(np.asarray(shape)).astype(self._dtype)

    @abstractmethod
    def call(self, shape):
        pass


class BernoulliMaskGenerator(MaskGenerator):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)

        self.p = p

    def call(self, shape):
        return self._rng.binomial(1, self.p, size=shape)


class UniformMaskGenerator(MaskGenerator):
    def __init__(self, bounds: Optional[Tuple[float, float]] = None, **kwargs):
        super().__init__(**kwargs)
        self._bounds = bounds

    def call(self, shape: Sequence[int]) -> np.ndarray:
        orig_shape = None
        if len(shape) != 2:
            orig_shape = shape
            shape = (shape[0], np.prod(shape[1:]))

        b, d = shape

        result = []
        for _ in range(b):
            if self._bounds is None:
                q = self._rng.choice(d)
            else:
                l = int(d * self._bounds[0])
                h = int(d * self._bounds[1])
                q = l + self._rng.choice(h)
                import pdb; pdb.set_trace()
            inds = self._rng.choice(d, q, replace=False)
            mask = np.zeros(d)
            mask[inds] = 1
            result.append(mask)

        result = np.vstack(result)

        if orig_shape is not None:
            result = np.reshape(result, orig_shape)

        return result

def get_add_mask_fn(mask_generator):
    def fn(x):
        [mask] = tf.py_function(mask_generator, [tf.shape(x)], [x.dtype])
        # This reshape ensures that mask shape is not unknown.
        mask = np.reshape(mask, tf.shape(x))
        return x, mask

    return fn


def get_add_marginal_masks_fn(marginal_dims):
    def fn(x):
        missing = tf.greater_equal(tf.range(tf.shape(x)[-1]), marginal_dims)
        missing = tf.cast(missing, x.dtype)
        return x, tf.zeros_like(x), missing

    return fn
