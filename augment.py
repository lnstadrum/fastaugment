import tensorflow as tf
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
lib = tf.load_op_library(os.path.join(script_dir, 'build', 'libaugment.so'))

empty = tf.convert_to_tensor(())

DEFAULT_PARAMS = {
    'translation'      : [0.1],
    'scale'            : [0.1],
    'rotation'         : 15,
    'perspective'      : [20],
    'cutout_prob'      : 0.5,
    'cutout_size'      : [0.3, 0.5],
    'mixup_prob'       : 0.5,
    'mixup_alpha'      : 0.4,
    'hue'              : 10,
    'saturation'       : 0.4,
    'value'            : 0.1,
    'flip_horizontally': True,
    'flip_vertically'  : False
}

def augment(x, y=None, **kwargs):
    x_, y_ = lib.Augment(input=x,
                         input_labels=empty if y is None else y,
                         **kwargs)
    if y is None:
        return x_
    return x_, y_
