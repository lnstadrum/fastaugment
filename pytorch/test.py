from fast_augment_torch import FastAugment, BYPASS_PARAMS
import numpy
import torch
import tempfile
import unittest


class ShapeTests(unittest.TestCase):
    """Output shape verification"""

    @classmethod
    def setUp(cls):
        cls.augment = FastAugment()

    def test_default_output_size(self):
        input_batch = torch.zeros(5, 123, 234, 3, dtype=torch.uint8).cuda()
        output_batch = self.augment(input_batch)
        self.assertEqual(output_batch.shape, input_batch.shape)

    def test_specific_output_size(self):
        input_batch = torch.zeros(7, 55, 66, 3, dtype=torch.uint8).cuda()
        width, height = 88, 77
        output_batch = self.augment(input_batch, output_size=[width, height])
        self.assertEqual(output_batch.shape, (7, height, width, 3))


class ColorTests(unittest.TestCase):
    """Color-related tests"""

    @classmethod
    def setUp(cls):
        cls.augment = FastAugment()

    def test_identity_u8(self):
        # make random input
        input_batch = torch.randint(size=(5, 23, 45, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()

        # apply identity transformation
        augment = FastAugment(**BYPASS_PARAMS)
        output_batch = augment(input_batch, output_type=torch.uint8)

        # compare: expected same output
        self.assertTrue(torch.equal(output_batch, input_batch))

    def test_identity_u32(self):
        # make random input
        input_batch = torch.randint(size=(5, 23, 45, 3), high=1).to(torch.float32)
        input_batch = input_batch.cuda()

        # apply identity transformation
        augment = FastAugment(**BYPASS_PARAMS)
        output_batch = augment(input_batch, output_type=torch.float32)

        # compare: expected same output
        self.assertTrue(torch.equal(output_batch, input_batch))

    def test_center_pixel(self):
        # make random grayscale input
        input_batch = torch.randint(size=(32, 17, 17, 1), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()
        input_batch = input_batch.repeat(1, 1, 1, 3)

        # apply transformations keeping the center pixel color unchanged
        params = BYPASS_PARAMS.copy()
        params["flip_vertically"] = True
        params["flip_horizontally"] = True
        params["perspective"] = [10, 20]
        augment = FastAugment(**params)
        output_batch = augment(input_batch, output_type=torch.uint8)

        # compare center pixel colors: expecting the same
        self.assertTrue(torch.equal(output_batch[:, 8, 8, :], input_batch[:, 8, 8, :]))

    def test_color_inversion_u8(self):
        # make random input
        input_batch = torch.zeros(5, 23, 45, 3).to(torch.uint8).cuda()

        # apply color inversion only
        params = BYPASS_PARAMS.copy()
        params["color_inversion"] = True
        output_batch = FastAugment(**params)(input_batch, output_type=torch.uint8)

        # compare colors
        input_batch = input_batch.cpu().numpy()
        output_batch = output_batch.cpu().numpy()
        comp = numpy.logical_xor(
            input_batch == output_batch, input_batch == 255 - output_batch
        )
        self.assertTrue(numpy.all(comp))

    def test_color_inversion_f32(self):
        # make random input
        input_batch = torch.rand(5, 23, 45, 3).cuda()

        # apply color inversion only
        params = BYPASS_PARAMS.copy()
        params["color_inversion"] = True
        output_batch = FastAugment(**params)(input_batch, output_type=torch.float32)

        # compare colors
        diff1 = input_batch - output_batch
        diff2 = 1 - input_batch - output_batch
        self.assertTrue(torch.allclose(diff1 * diff2, torch.zeros_like(diff1)))


class MixupLabelsTests(unittest.TestCase):
    """Tests of labels computation with mixup"""

    def test_no_mixup(self):
        # make random input
        input_batch = torch.randint(size=(8, 8, 8, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()
        input_labels = torch.rand(size=(8, 1000))

        # apply random transformation
        _, output_labels = FastAugment()(input_batch, input_labels)

        # compare labels: expected same
        self.assertTrue(torch.equal(input_labels, output_labels))

    def test_yes_mixup(self):
        # make random labels
        input_labels = torch.randint(size=(50,), high=2, dtype=torch.int32)

        # make images from labels
        input_batch = (255 * input_labels).to(torch.uint8).cuda()
        input_batch = input_batch.reshape(-1, 1, 1, 1).repeat(1, 5, 5, 3)

        # transform labels to one-hot
        input_proba = torch.nn.functional.one_hot(input_labels.to(torch.long), 2).to(
            torch.float32
        )

        # apply mixup
        augment = FastAugment(
            rotation=0,
            flip_vertically=True,
            flip_horizontally=True,
            hue=0,
            saturation=0,
            brightness=0,
            gamma_corr=0,
            cutout=0,
            mixup=0.9,
        )
        output_batch, output_proba = augment(input_batch, input_proba)

        # check that probabilities sum up to 1
        assert torch.allclose(output_proba[:, 0] + output_proba[:, 1], torch.ones((50)))

        # compare probabilities to center pixel values
        assert torch.allclose(output_proba[:, 1], output_batch[:, 3, 3, 0].cpu())


class SeedTests(unittest.TestCase):
    def test_seed(self):
        # make random input
        input_batch = torch.randint(size=(16, 50, 50, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()

        # make random labels
        input_labels = torch.randint(size=(16,), high=2).to(torch.long)
        input_proba = torch.nn.functional.one_hot(input_labels, 20).to(torch.float32)

        # generate output batches
        output_batch1, output_proba1 = FastAugment(mixup=0.75, seed=123)(
            input_batch, input_proba
        )
        output_batch2, output_proba2 = FastAugment(mixup=0.75, seed=234)(
            input_batch, input_proba
        )
        output_batch3, output_proba3 = FastAugment(mixup=0.75, seed=123)(
            input_batch, input_proba
        )

        # compare
        self.assertFalse(torch.equal(output_batch1, output_batch2))
        self.assertFalse(torch.equal(output_proba1, output_proba2))
        self.assertTrue(torch.equal(output_batch1, output_batch3))
        self.assertTrue(torch.equal(output_proba1, output_proba3))

    def test_seed_reset(self):
        # make random input
        input_batch = torch.randint(size=(16, 50, 50, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()

        # make random labels
        input_labels = torch.randint(size=(16,), high=2).to(torch.long)
        input_proba = torch.nn.functional.one_hot(input_labels, 20).to(torch.float32)

        augment = FastAugment(mixup=0.75, seed=123)

        # generate output batches
        output_batch1, output_proba1 = augment(input_batch, input_proba)
        output_batch2, output_proba2 = augment(input_batch, input_proba)

        # set back the initial seed and generate another output batch
        augment.set_seed(123)
        output_batch3, output_proba3 = augment(input_batch, input_proba)

        # compare
        self.assertFalse(torch.equal(output_batch1, output_batch2))
        self.assertFalse(torch.equal(output_proba1, output_proba2))
        self.assertTrue(torch.equal(output_batch1, output_batch3))
        self.assertTrue(torch.equal(output_proba1, output_proba3))


class DatatypeTests(unittest.TestCase):
    """Datatype verification"""

    def test_uint8_vs_float32(self):
        # make random input
        input_batch = torch.randint(size=(64, 32, 32, 3), high=255)
        input_batch = input_batch.to(torch.uint8).cuda()

        # apply identical transformation
        augment = FastAugment(seed=96)
        output_batch_ref = augment(input_batch, output_type=torch.uint8)
        augment.set_seed(96)
        output_batch_float = augment(input_batch, output_type=torch.float32)

        # check output types
        self.assertTrue(output_batch_ref.dtype == torch.uint8)
        self.assertTrue(output_batch_float.dtype == torch.float32)

        # cast back to uint8 and compare: expecting the same output
        output_batch_test = (255 * output_batch_float.clamp(0, 1)).to(torch.uint8)
        self.assertTrue(torch.equal(output_batch_ref, output_batch_test))


if __name__ == "__main__":
    unittest.main()
