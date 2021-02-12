import tensorflow as tf
import os


# load library
script_dir = os.path.dirname(os.path.realpath(__file__))
try:
    lib = tf.load_op_library(os.path.join(script_dir, 'libaugment.so'))
except:
    lib = tf.load_op_library(os.path.join(script_dir, 'build', 'libaugment.so'))


# make empty tensor placeholder
empty_tensor = tf.convert_to_tensor(())


# parameters to bypass any augmentation (for debuging purposes)
BYPASS_PARAMS = {
    'translation'      : 0,
    'scale'            : 0,
    'prescale'         : 1,
    'rotation'         : 0,
    'perspective'      : 0,
    'cutout_prob'      : 0,
    'mixup_prob'       : 0,
    'saturation'       : 0,
    'value'            : 0,
    'hue'              : 0,
    'flip_horizontally': False,
    'flip_vertically'  : False
}


@tf.autograph.experimental.do_not_convert
def augment(x, y=None,
            output_size=None,
            translation=0.1,
            scale=0.1,
            prescale=1,
            rotation=10,
            perspective=20,
            cutout_prob=0.5,
            cutout_size=(0.3, 0.5),
            mixup_prob=0.5,
            mixup_alpha=0.4,
            hue=10,
            saturation=0.4,
            value=0.1,
            flip_horizontally=True,
            flip_vertically=False):
    listify = lambda it : it if isinstance(it, list) or isinstance(it, tuple) else [it]
    x_, y_ = lib.Augment(input=x,
                         input_labels=empty_tensor if y is None else y,
                         output_size=output_size or [],
                         translation=listify(translation),
                         scale=listify(scale),
                         prescale=prescale,
                         rotation=rotation,
                         perspective=listify(perspective),
                         cutout_prob=cutout_prob,
                         cutout_size=listify(cutout_size),
                         mixup_prob=mixup_prob,
                         mixup_alpha=mixup_alpha,
                         hue=hue,
                         saturation=saturation,
                         value=value,
                         flip_horizontally=flip_horizontally,
                         flip_vertically=flip_vertically)
    if y is None:
        return x_
    return x_, y_


def set_seed(seed):
    lib.set_seed(seed)