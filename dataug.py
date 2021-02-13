import tensorflow as tf
import os


# loading the library
script_dir = os.path.dirname(os.path.realpath(__file__))
try:
    lib = tf.load_op_library(os.path.join(script_dir, 'libdataug.so'))
except:
    lib = tf.load_op_library(os.path.join(script_dir, 'build', 'libdataug.so'))


# empty tensor placeholder
empty_tensor = tf.convert_to_tensor(())


# parameters disabling all augmentation tranformations (for debuging purposes)
BYPASS_PARAMS = {
    'translation'      : 0,
    'scale'            : 0,
    'prescale'         : 1,
    'rotation'         : 0,
    'perspective'      : 0,
    'flip_horizontally': False,
    'flip_vertically'  : False,
    'cutout_prob'      : 0,
    'mixup_prob'       : 0,
    'saturation'       : 0,
    'value'            : 0,
    'hue'              : 0,
    'gamma_corr'       : 0
}


@tf.autograph.experimental.do_not_convert
def center_crop(x, size, translation=0):
    """ Helper function sampling the center crop of maximum size from a given image.
    This can be useful when constructing batches from a set of images of different sizes trying to maximize the sampled area.
    The output is sampled by taking the maximum area fitting into the input image bounds, but keeping a given output aspect ratio.
    The output size in pixels is fixed and does not depend on the input image size.
    By default, i.e. with `translation=0`, the output image center matches the input image center. Otherwise, the output image sampling area is randomly shifted
    according to the given `translation` value and may fall out of the input image area. The corresponding output pixels are filled with gray.
    Bilinear interpolation is used to compute the output pixels values.

    Args:
        x:              Input image tensor  of `uint8` type in channels-last (HWC) layout. The color input images are supported only (C=3).
        size:           A list or tuple `(W, H)` specifying the output image size in pixels.
        translation:    Normalized image translation range along X and Y axis. `0.1` corresponds to a random shift by at most 10% of the image size in both directions.
    """
    x, _ = lib.Augment(input=x,
                       input_labels=empty_tensor,
                       output_size=size,
                       output_type=tf.uint8,
                       translation=[translation])
    return x


@tf.autograph.experimental.do_not_convert
def augment(x, y=None,
            output_size=None,
            output_type=tf.float32,
            translation=0.1,
            scale=0.1,
            prescale=1,
            rotation=15,
            perspective=15,
            cutout_prob=0.5,
            cutout_size=(0.3, 0.5),
            mixup_prob=0.5,
            mixup_alpha=0.4,
            hue=10,
            saturation=0.4,
            value=0.1,
            gamma_corr=0.2,
            flip_horizontally=True,
            flip_vertically=False):
    listify = lambda it : it if isinstance(it, list) or isinstance(it, tuple) else [it]
    x_, y_ = lib.Augment(input=x,
                         input_labels=empty_tensor if y is None else y,
                         output_size=output_size or [],
                         output_type=output_type,
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
                         gamma_corr=gamma_corr,
                         flip_horizontally=flip_horizontally,
                         flip_vertically=flip_vertically)
    if y is None:
        return x_
    return x_, y_


def set_seed(seed):
    lib.set_seed(seed)