import tensorflow as tf
from os import path


# loading the library
base_dir = path.dirname(path.realpath(__file__))
try:
    lib = tf.load_op_library(
        path.join(base_dir, "build", "libfastaugment_tensorflow.so")
    )
except:
    raise ImportError(
        "Cannot load the extension library 'libfastaugment_tensorflow.so'. Make sure it is compiled."
    )


# empty tensor placeholder
empty_tensor = tf.convert_to_tensor(())


# converts a value into a list if it is not yet a list or a tuple
def listify(it):
    return it if isinstance(it, list) or isinstance(it, tuple) else [it]


# parameters disabling all augmentation tranformations (for debuging purposes)
BYPASS_PARAMS = {
    "translation": 0,
    "scale": 0,
    "prescale": 1,
    "rotation": 0,
    "perspective": 0,
    "flip_horizontally": False,
    "flip_vertically": False,
    "cutout": 0,
    "mixup": 0,
    "saturation": 0,
    "brightness": 0,
    "hue": 0,
    "gamma_corr": 0,
    "color_inversion": False,
}


@tf.autograph.experimental.do_not_convert
def center_crop(x, size, translation=0):
    """Helper function to sample a center crop of maximum size from a given image.
    This can be useful when constructing batches from a set of images of different
    sizes trying to maximize the sampled area.
    The output is sampled by taking the maximum area fitting into the input image
    bounds, but keeping a given output aspect ratio.
    The output size in pixels is fixed and does not depend on the input image size.

    By default, i.e. with `translation=0`, the output image center matches the
    input image center. Otherwise, the output image sampling area is randomly
    shifted according to the given `translation` value and may fall out of the
    input image area. The corresponding output pixels are filled with gray.
    Bilinear interpolation is used to compute the output pixels values.

    Args:
        x:              Input image tensor  of `uint8` type in channels-last (HWC)
                        layout. The color input images are supported only (C=3).
        size:           A list or tuple `(W, H)` specifying the output image size
                        in pixels.
        translation:    Normalized image translation range along X and Y axis.
                        `0.1` corresponds to a random shift by at most 10% of the
                        image size in both directions.
    """
    x, _ = lib.Augment(
        input=x,
        input_labels=empty_tensor,
        output_size=size,
        output_type=tf.uint8,
        translation=[translation],
    )
    return x


@tf.autograph.experimental.do_not_convert
def augment(
    x,
    y=None,
    output_size=None,
    output_type=tf.float32,
    translation=0.1,
    scale=0.1,
    prescale=1,
    rotation=15,
    perspective=15,
    cutout=0.5,
    cutout_size=(0.3, 0.5),
    mixup=0,
    mixup_alpha=0.4,
    hue=10,
    saturation=0.4,
    brightness=0.1,
    gamma_corr=0.2,
    color_inversion=False,
    flip_horizontally=True,
    flip_vertically=False,
    seed=0,
):
    """Applies a set of random geometry and color transformations to batches of images.
    The applied transformation differs from one image to another one in the same batch.
    Transformations parameters are sampled from uniform distributions of given ranges.
    Their default values enable some moderate amount of augmentations.
    Every image is sampled only once through a bilinear interpolator.

    Transformation application order:
        - horizontal and/or vertical flipping,
        - perspective distortion,
        - in-plane image rotation and scaling,
        - translation,
        - gamma correction,
        - hue, saturation and brightness correction,
        - color inversion,
        - mixup,
        - CutOut.

        Args:
            translation:        Normalized image translation range along X and Y axis.
                                `0.1` corresponds to a random shift by at most 10% of
                                the image size in both directions (default).
                                If one value given, the same range applies for X and Y
                                axes.
            scale:              Scaling factor range along X and Y axes. `0.1`
                                corresponds to stretching the images by a random factor
                                of at most 10% (default).
                                If one value given, the applied scaling keeps the
                                aspect ratio: the same factor is used along X and Y
                                axes.
            prescale:           A constant scaling factor applied to all images. Can be
                                used to shift the random scaling distribution from its
                                default average equal to 1 and crop out image borders.
                                The default value is 1.
            rotation:           Rotation angle range in degrees. The images are rotated
                                in both clockwise and counter-clockwise direction by a
                                random angle less than `rotation`. Default: 10 degrees.
            perspective:        Perspective distortion range setting the maximum
                                tilting and panning angles in degrees.
                                The image plane is rotated in 3D around X and Y axes
                                (tilt and pan respectively) by random angles smaller
                                than the given value(s).
                                If one number is given, the same range applies for both
                                axes. The default value is 15 degrees.
            flip_horizontally:  A boolean. If `True`, the images are flipped
                                horizontally with 50% chance. Default: True.
            flip_vertically:    A boolean. If `True`, the images are flipped vertically
                                with 50% chance. Default: False.
            hue:                Hue shift range in degrees. The image pixels color hues
                                are shifted by a random angle smaller than `hue`.
                                A hue shift of +/-120 degrees transforms green in
                                red/blue and vice versa. The default value is 10 deg.
            saturation:         Color saturation factor range. For every input image,
                                the color saturation is scaled by a random factor
                                sampled in range `[1 - saturation, 1 + saturation]`.
                                Applying zero saturation scale produces a grayscale
                                image. The default value is 0.4.
            brightness:         Brightness factor range. For every input image, the
                                intensity is scaled by a random factor sampled in range
                                `[1 - brightness, 1 + brightness]`.
                                The default value is 0.1
            gamma_corr:         Gamma correction factor range. For every input image,
                                the factor value is randomly sampled in range
                                `[1 - gamma_corr, 1 + gamma_corr]`.
                                Gamma correction boosts (for factors below 1) or
                                reduces (for factors above 1) dark image areas
                                intensity, while bright areas are less affected.
                                The default value is 0.2.
            color_inversion:    A boolean. If `True`, colors of all pixels in every
                                image are inverted (negated) with 50% chance.
                                Default: False.
            cutout:             Probability of CutOut being applied to a given input
                                image. The default value is 0.5.
                                CutOut erases a randomly placed rectangular area of an
                                image. See the original paper for more details:
                                https://arxiv.org/pdf/1708.04552.pdf
            cutout_size:        A list specifying the normalized size range CutOut area
                                width and height are sampled from.
                                `[0.3, 0.5]` range produces a rectangle of 30% to 50%
                                of image size on every side (default).
                                If an empty list is passed, CutOut application is
                                disabled.
            mixup:              Probability of mixup being applied to a given input
                                image. Mixup is disabled by default (`mixup` is set to
                                zero).
                                Mixup is applied across the batch. Every two mixed
                                images undergo the same set of other transformations
                                except flipping which can be different.
                                Requires the input labels `y`. If not provided, an
                                exception is thrown.
            mixup_alpha:        Mixup `alpha` parameter (default: 0.4). See the
                                original paper for more details:
                                https://arxiv.org/pdf/1710.09412.pdf
            seed:               Random seed. If different from 0, reproduces the same
                                sequence of transformations for a given set of
                                parameters and input size.

        Returns:
            A `Tensor` with a set of transformations applied to the input image or
            batch, and another `Tensor` containing the image labels in one-hot format.
    """
    x_, y_ = lib.Augment(
        input=x,
        input_labels=empty_tensor if y is None else y,
        output_size=output_size or [],
        output_type=output_type,
        translation=listify(translation),
        scale=listify(scale),
        prescale=prescale,
        rotation=rotation,
        perspective=listify(perspective),
        cutout_prob=cutout,
        cutout_size=listify(cutout_size),
        mixup_prob=mixup,
        mixup_alpha=mixup_alpha,
        hue=hue,
        saturation=saturation,
        brightness=brightness,
        gamma_corr=gamma_corr,
        color_inversion=color_inversion,
        flip_horizontally=flip_horizontally,
        flip_vertically=flip_vertically,
        seed=seed,
    )
    if y is None:
        return x_
    return x_, y_


class Augment(tf.keras.layers.Layer):
    """Data augmentation layer.
    Wraps `augment()` function.

    Args:
        training_only:    If set to `True`, the data augmentation is only applied
                          during training and is bypassed otherwise.
                          Useful when the layer makes part of the actual model to
                          train.
        **kwargs:         Data augmentation parameters. Same as `augment()` function
                          parameters.
    """

    def __init__(self, training_only, **kwargs):
        self.training_only = training_only

        # keep kwargs to call augment()
        self.args = kwargs

        # remove name from it, if any
        name = self.args.pop("name", None)

        # set default output_type
        self.output_type = self.args.pop("output_type", tf.float32)

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
        config["training_only"] = self.training_only
        config["output_type"] = self.output_type
        return config
