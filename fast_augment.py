import tensorflow as tf
import os


# loading the library
script_dir = os.path.dirname(os.path.realpath(__file__))
try:
    lib = tf.load_op_library(os.path.join(script_dir, 'libfastaugment.so'))
except:
    lib = tf.load_op_library(os.path.join(script_dir, 'build', 'libfastaugment.so'))


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
    'brightness'       : 0,
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
            brightness=0.1,
            gamma_corr=0.2,
            flip_horizontally=True,
            flip_vertically=False,
            seed=0):
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
                         brightness=brightness,
                         gamma_corr=gamma_corr,
                         flip_horizontally=flip_horizontally,
                         flip_vertically=flip_vertically,
                         seed=seed)
    if y is None:
        return x_
    return x_, y_


class Augment(tf.keras.layers.Layer):
    """ Data augmentation layer.
    Wraps `augment()` function.

    Args:
        training_only:      If set to `True`, the data augmentation is only applied during training and is bypassed otherwise.
                            Useful when the layer makes part of the actual model to train.
        **kwargs:           Data augmentation parameters. Same as `augment()` function parameters.
    """

    def __init__(self, training_only, **kwargs):
        self.training_only = training_only

        # keep kwargs to call augment()
        self.args = kwargs

        # remove name from it, if any
        name = self.args.pop('name', None)

        # set default output_type
        self.output_type = self.args.pop('output_type', tf.float32)

        # call superclass constructor
        super().__init__(name=name)


    def call(self, inputs, training=None):
        if isinstance(inputs, list):
            assert len(inputs) == 2, "Augment layer expects at most 2 input tensors"
            images, prob = inputs[0], inputs[1]
        else:
            images, prob = inputs, None

        # run augment if training
        if training or not self.training_only:
            return augment(x=images, y=prob, output_type=self.output_type, **self.args)

        # bypass otherwise, but cast to the same output_type to avoid type mismatch
        images = tf.cast(images, self.output_type)
        if prob is None:
            return images
        else:
            return images, prob


    def get_config(self):
        config = super().get_config()
        config.update(self.args)
        config['output_type'] = str(self.output_type)
        return config