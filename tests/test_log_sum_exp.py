from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from edward.util import log_sum_exp

sess = tf.InteractiveSession()


def test_1d():
    x = tf.constant([-1.0, -2.0, -3.0, -4.0])
    val_ed = log_sum_exp(x)
    val_true = -0.5598103014388045
    assert np.allclose(val_ed.eval(), val_true)


def test_2d():
    x = tf.constant([[-1.0], [-2.0], [-3.0], [-4.0]])
    val_ed = log_sum_exp(x)
    val_true = -0.5598103014388045
    assert np.allclose(val_ed.eval(), val_true)
