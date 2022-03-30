import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'

from fast_augment import augment, Augment, BYPASS_PARAMS
import numpy
import tensorflow as tf
import tempfile
import unittest


class ShapeTests(unittest.TestCase):
    """ Output shape verification
    """
    def test_default_output_size(self):
        input_batch = tf.zeros((5, 123, 234, 3), dtype=tf.uint8)
        output_batch = augment(input_batch)
        self.assertEqual(output_batch.shape, input_batch.shape)

    def test_specific_output_size(self):
        input_batch = tf.zeros((7, 55, 66, 3), dtype=tf.uint8)
        width, height = 88, 77
        output_batch = augment(input_batch, output_size=[width, height])
        self.assertEqual(output_batch.shape, (7, height, width, 3))


class ColorTests(tf.test.TestCase):
    """ Color-related tests
    """
    def test_identity_u8(self):
        # make random input
        input_batch = tf.random.uniform((5, 23, 45, 3), maxval=255)
        input_batch = tf.cast(input_batch, tf.uint8)

        # apply identity transformation
        output_batch = augment(input_batch, output_type=tf.uint8, **BYPASS_PARAMS)

        # compare: expected same output
        self.assertAllEqual(output_batch, input_batch)

    def test_identity_f32(self):
        # make random input
        input_batch = tf.random.uniform((5, 23, 45, 3), maxval=1)

        # apply identity transformation
        output_batch = augment(input_batch, output_type=tf.float32, **BYPASS_PARAMS)

        # compare: expected same output
        self.assertAllEqual(output_batch, input_batch)

    def test_center_pixel(self):
        # make random grayscale input
        input_batch = tf.random.uniform((32, 17, 17, 1), maxval=255)
        input_batch = tf.cast(input_batch, tf.uint8)
        input_batch = tf.tile(input_batch, (1, 1, 1, 3))

        # apply transformations keeping the center pixel color unchanged
        params = BYPASS_PARAMS.copy()
        params['flip_vertically'] = True
        params['flip_horizontally'] = True
        params['perspective'] = [10, 20]
        output_batch = augment(input_batch,
                               output_type=tf.uint8,
                               **params)

        # compare center pixel colors: expecting the same
        self.assertAllEqual(output_batch[:,8,8,:], input_batch[:,8,8,:])

    def test_color_inversion_f8(self):
        # make random input
        input_batch = tf.zeros((5, 23, 45, 3), tf.uint8)

        # apply color inversion only
        params = BYPASS_PARAMS.copy()
        params['color_inversion'] = True
        output_batch = augment(input_batch, output_type=tf.uint8, **params)

        # compare colors
        comp = numpy.logical_xor(input_batch == output_batch, input_batch == 255 - output_batch)
        self.assertTrue(numpy.all(comp))

    def test_color_inversion_f32(self):
        # make random input
        input_batch = tf.random.uniform((5, 23, 45, 3), dtype=tf.float32)

        # apply color inversion only
        params = BYPASS_PARAMS.copy()
        params['color_inversion'] = True
        output_batch = augment(input_batch, output_type=tf.float32, **params)

        # compare colors
        diff1 = input_batch - output_batch
        diff2 = 1 - input_batch - output_batch
        self.assertAllClose(diff1 * diff2, tf.zeros_like(diff1))


class MixupLabelsTests(tf.test.TestCase):
    """ Tests of labels computation with mixup
    """
    def test_no_mixup(self):
        # make random input
        input_batch = tf.random.uniform((8, 8, 8, 3), maxval=255)
        input_batch = tf.cast(input_batch, tf.uint8)
        input_labels = tf.random.uniform((8, 1000))

        # apply random transformation
        _, output_labels = augment(input_batch, input_labels)

        # compare labels: expected same
        self.assertAllEqual(input_labels, output_labels)

    def test_yes_mixup(self):
        # make random labels
        input_labels = tf.random.uniform((50,), minval=0, maxval=2, dtype=tf.int32)

        # make images from labels
        input_batch = tf.cast(255 * input_labels, tf.uint8)
        input_batch = tf.tile(tf.reshape(input_batch, (-1, 1, 1, 1)), (1, 5, 5, 3))

        # transform labels to one-hot
        input_proba = tf.one_hot(input_labels, 2, dtype=tf.float32)

        # apply mixup
        output_batch, output_proba = augment(input_batch,
                                             input_proba,
                                             rotation=0,
                                             flip_vertically=True,
                                             flip_horizontally=True,
                                             hue=0,
                                             saturation=0,
                                             brightness=0,
                                             gamma_corr=0,
                                             cutout=0,
                                             mixup=0.9)

        # check that probabilities sum up to 1
        self.assertAllClose(output_proba[:,0] + output_proba[:, 1], tf.ones((50)))

        # compare probabilities to center pixel values
        self.assertAllClose(output_proba[:, 1], output_batch[:, 3, 3, 0])


class SeedTests(tf.test.TestCase):
    def test_seed(self):
        # make random input
        input_batch = tf.random.uniform((16, 50, 50, 3), maxval=255)
        input_batch = tf.cast(input_batch, tf.uint8)

        # make random labels
        input_labels = tf.random.uniform((16,), minval=0, maxval=2, dtype=tf.int32)
        input_proba = tf.one_hot(input_labels, 20, dtype=tf.float32)

        # generate output batches
        output_batch1, output_proba1 = augment(input_batch, input_proba, mixup=0.75, seed=123)
        output_batch2, output_proba2 = augment(input_batch, input_proba, mixup=0.75, seed=234)
        output_batch3, output_proba3 = augment(input_batch, input_proba, mixup=0.75, seed=123)

        # compare
        self.assertNotAllEqual(output_batch1, output_batch2)
        self.assertNotAllEqual(output_proba1, output_proba2)
        self.assertAllEqual(output_batch1, output_batch3)
        self.assertAllEqual(output_proba1, output_proba3)



class DatatypeTests(tf.test.TestCase):
    """ Datatype verification
    """
    def test_uint8_vs_float32(self):
        # make random input
        input_batch = tf.random.uniform((64, 32, 32, 3), maxval=255)
        input_batch = tf.cast(input_batch, tf.uint8)

        # apply identity transformation
        output_batch_ref = augment(input_batch, output_type=tf.uint8, seed=96)
        output_batch_float = augment(input_batch, output_type=tf.float32, seed=96)

        # check output types
        self.assertTrue(output_batch_ref.dtype == tf.uint8)
        self.assertTrue(output_batch_float.dtype == tf.float32)

        # cast back to uint8 and compare: expecting the same output
        output_batch_test = tf.cast(255 * tf.clip_by_value(output_batch_float, 0, 1), tf.uint8)
        self.assertAllEqual(output_batch_ref, output_batch_test)


class KerasLayerTests(tf.test.TestCase):
    """ Testing Augment keras layer
    """
    def test_fitting_and_export(self):
        # build a model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3), dtype=tf.uint8),
            Augment(training_only=True),
            tf.keras.layers.Conv2D(10, kernel_size=5, strides=2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten()
        ])
        model.compile(loss='categorical_crossentropy')

        # make random input
        images = tf.random.uniform((20, 32, 32, 3), maxval=255)
        images = tf.cast(images, tf.uint8)

        # make random labels
        labels = tf.random.uniform((images.shape[0],), minval=0, maxval=2, dtype=tf.int32)
        proba = tf.one_hot(labels, 10, dtype=tf.float32)

        # fit
        model.fit(x=images, y=proba, verbose=False)

        # save and load
        custom_objects = {"Augment": Augment}
        with tf.keras.utils.custom_object_scope(custom_objects):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, '.hdf5')
                model.save(path)
                model = tf.keras.models.load_model(path)


    def test_in_model(self):
        # create input layers for images and labels
        x_in = tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.uint8)
        y_in = tf.keras.layers.Input(shape=(1000), dtype=tf.float32)

        # add augmentation layer
        x_out, y_out = Augment(training_only=False, mixup=0.75, seed=111)([x_in, y_in])

        # build a model
        model = tf.keras.models.Model(inputs=[x_in, y_in], outputs=[x_out, y_out])

        # make random input
        input_images = tf.random.uniform((5, 224, 224, 3), maxval=255)
        input_images = tf.cast(input_images, tf.uint8)
        input_prob = tf.random.uniform((5,), maxval=2, dtype=tf.int32)
        input_prob = tf.one_hot(input_prob, 1000)

        # run prediction
        output_images_test, output_prob_test = model.predict([input_images, input_prob])

        # generate reference
        output_images_ref, output_prob_ref = augment(x=input_images, y=input_prob, mixup=0.75, seed=111)

        # compare
        self.assertAllEqual(output_images_test, output_images_ref)
        self.assertAllEqual(output_prob_test, output_prob_ref)


    def test_in_model_without_labels(self):
        # create input layer
        x_in = tf.keras.layers.Input(shape=(224, 224, 3), dtype=tf.uint8)

        # add augmentation layer
        x_out = Augment(training_only=False, seed=111)(x_in)

        # build a model
        model = tf.keras.models.Model(inputs=x_in, outputs=x_out)

        # make random input
        input_images = tf.random.uniform((5, 224, 224, 3), maxval=255)
        input_images = tf.cast(input_images, tf.uint8)

        # run prediction
        output_images = model.predict(input_images)

        # generate reference
        output_images_ref = augment(x=input_images, seed=111)

        # compare
        self.assertAllEqual(output_images, output_images_ref)


if __name__ == '__main__':
    unittest.main()