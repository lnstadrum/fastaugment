#define EIGEN_USE_GPU

#include <vector>
#include <random>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "augment.h"


#define CUDASSERT(CALL, MSG) \
        OP_REQUIRES(context, CALL == cudaSuccess, errors::Internal(MSG));


template <typename val_t, typename mod_t>
inline val_t roundUp(val_t value, mod_t modulo) {
    return (value + modulo - 1) / modulo * modulo;
}


using namespace tensorflow;


static const float pi = 3.141592653589f;
static std::random_device randomDevice;
static std::default_random_engine rnd(randomDevice());


template<class Device, typename in_t, typename out_t>
class DataugOpKernel : public OpKernel {
    size_t textureAlignment, texturePitchAlignment;
    std::vector<int64> outputSize;
    float translation[2];                       //!< normalized translation range in X and Y directions
    float scale[2];                             //!< scaling factor range in X and Y directions
    float prescale;                             //!< constant scaling factor
    float rotation;                             //!< in-plane rotation range in radians
    float perspective[2];                       //!< out-of-plane rotation range in radians around horizontal and vertical axes (tilt and pan)
    float cutoutProb;                           //!< CutOut per-image probability
    float cutout[2];                            //!< CutOut normalized size range
    float mixupProb;                            //!< Mixup per-image probability
    float mixupAlpha;                           //!< Mixup alpha parameter
    float hue;                                  //!< hue shift range in radians
    float saturation;                           //!< saturation factor range
    float value;                                //!< value factor range
    bool flipHorizontally, flipVertically;      //!< if `true`, the image is flipped with 50% chance along a specific direction
    bool isotropicScaling;                      //!< if `true`, the scale factor is the same along X and Y direction

    size_t getPair(OpKernelConstruction* context, const char* attribute, float&a, float& b) {
        std::vector<float> listArg;
        if (context->GetAttr(attribute, &listArg) != Status::OK()) {
            context->CtxFailure(errors::InvalidArgument("Cannot get '" + std::string(attribute) + "' attribute value. Expected a list of floating point values."));
            return 0;
        }

        if (listArg.size() > 2)
            context->CtxFailure(errors::InvalidArgument("Cannot interpret '" + std::string(attribute) + "' attribute value of " + std::to_string(listArg.size()) + " elements"));
        a = b = 0;
        if (listArg.size() == 1)
            a = b = listArg[0];
        else if (listArg.size() == 2) {
            a = listArg[0];
            b = listArg[1];
        }
        return listArg.size();
    }

public:
    explicit DataugOpKernel(OpKernelConstruction* context) : OpKernel(context) {
        // get output size
        OP_REQUIRES_OK(context, context->GetAttr("output_size", &outputSize));
        if (!outputSize.empty() && outputSize.size() != 2)
            context->CtxFailure(errors::InvalidArgument("Invalid output_size: expected an empty list or a list of 2 entries, got " + std::to_string(outputSize.size())));

        // get translation range
        getPair(context, "translation", translation[0], translation[1]);

        // get scaling range
        size_t n = getPair(context, "scale", scale[0], scale[1]);
        isotropicScaling = (n == 1);
        OP_REQUIRES_OK(context, context->GetAttr("prescale", &prescale));

        // get rotation angle and convert to radians
        OP_REQUIRES_OK(context, context->GetAttr("rotation", &rotation));
        rotation *= pi / 180.0f;

        // get perspective angles and convert to radians
        getPair(context, "perspective", perspective[0], perspective[1]);
        perspective[0] *= pi / 180.0f;
        perspective[1] *= pi / 180.0f;

        // get CutOut parameters
        OP_REQUIRES_OK(context, context->GetAttr("cutout_prob", &cutoutProb));
        if (cutoutProb < 0 || cutoutProb > 1)
            context->CtxFailure(errors::InvalidArgument("Invalid CutOut probability: " + std::to_string(cutoutProb) + ", expected a value in [0, 1] range"));
        n = getPair(context, "cutout_size", cutout[0], cutout[1]);
        if (n == 0)
            cutoutProb = 0;

        // get Mixup parameters
        OP_REQUIRES_OK(context, context->GetAttr("mixup_prob", &mixupProb));
        if (mixupProb < 0 || mixupProb > 1)
            context->CtxFailure(errors::InvalidArgument("Invalid Mixup probability: " + std::to_string(mixupProb) + ", expected a value in [0, 1] range"));
        OP_REQUIRES_OK(context, context->GetAttr("mixup_alpha", &mixupAlpha));

        // get HSV transformation magnitudes
        OP_REQUIRES_OK(context, context->GetAttr("hue", &hue));
        hue *= pi / 180.0f;
        OP_REQUIRES_OK(context, context->GetAttr("saturation", &saturation));
        OP_REQUIRES_OK(context, context->GetAttr("value", &value));

        // get flipping flags
        OP_REQUIRES_OK(context, context->GetAttr("flip_horizontally", &flipHorizontally));
        OP_REQUIRES_OK(context, context->GetAttr("flip_vertically", &flipVertically));

        // get current device number
        int device;
        CUDASSERT(cudaGetDevice(&device), "Cannot get current CUDA device number");

        // query device properties
        cudaDeviceProp properties;
        CUDASSERT(cudaGetDeviceProperties(&properties, device), "Cannot get CUDA device properties");

        // get alignment values
        textureAlignment = properties.textureAlignment;
        texturePitchAlignment = properties.texturePitchAlignment;
    }


    void Compute(OpKernelContext* context) override {
        // grab the input tensor
        const Tensor& inputTensor = context->input(0);
        OP_REQUIRES(context, inputTensor.shape().dims() == 4, errors::InvalidArgument("Expected a 4-dimensional input tensor, got " + std::to_string(inputTensor.shape().dims()) + " dimensions"));
        const in_t* inputPtr = inputTensor.flat<in_t>().data();

        // get input sizes
        const int64
            batchSize = inputTensor.shape().dim_size(0),
            inputHeight = inputTensor.shape().dim_size(1),
            inputWidth = inputTensor.shape().dim_size(2),
            inputChannels = inputTensor.shape().dim_size(3);

        // check number of input channels
        OP_REQUIRES(context, inputChannels == 3, errors::InvalidArgument("Expected a 3-channel input tensor, got " + std::to_string(inputChannels) + " channels"));

        // get CUDA stream
        auto stream = context->eigen_device<Device>().stream();

        // allocate a temporary buffer ensuring starting address and pitch alignment
        Tensor buffer;
        int64 pitchBytes = roundUp(inputWidth * 4 * sizeof(in_t), texturePitchAlignment);
        const size_t bufferSizeBytes = textureAlignment + batchSize * inputHeight * pitchBytes;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT8, { (int64)bufferSizeBytes }, &buffer));

        // compute aligned starting address
        auto unalignedBufferAddress = buffer.flat<uint8_t>().data();
        auto bufferPtr = reinterpret_cast<in_t*>(roundUp(reinterpret_cast<size_t>(unalignedBufferAddress), textureAlignment));

        // pad the input to have 4 channels and an aligned pitch
        padChannels(stream, inputPtr, bufferPtr, inputWidth, inputHeight, batchSize, pitchBytes / (4 * sizeof(in_t)));
        CUDASSERT(cudaGetLastError(), "Cannot pad the input image");

        // get input labels tensor
        const auto& labelsShape = context->input(1).shape();
        const bool noLabels = labelsShape.dims() == 0 || (labelsShape.dims() == 1 && labelsShape.dim_size(0) == 0);
        if (!noLabels) {
            OP_REQUIRES(context, labelsShape.dims() == 2,
                errors::InvalidArgument("Expected a 2-dimensional input_labels tensor, got " + std::to_string(labelsShape.dims()) + " dimensions"));
            OP_REQUIRES(context, labelsShape.dim_size(0) == batchSize,
                errors::InvalidArgument("First dimension of the input labels tensor is expected to match the batch size, but got " + std::to_string(labelsShape.dim_size(0))));
        }
        const float* inputLabelsPtr = noLabels ? nullptr : context->input(1).flat<float>().data();

        // prepare parameters samplers
        std::vector<Params> paramsCpu;
        paramsCpu.resize(batchSize);
        std::uniform_real_distribution<float>
            xShiftFactor(-translation[0], translation[0]),
            yShiftFactor(-translation[1], translation[1]),
            xScaleFactor(prescale - scale[0], prescale + scale[0]),
            yScaleFactor(prescale - scale[1], prescale + scale[1]),
            rotationAngle(-rotation, rotation),
            tiltAngle(-perspective[0], perspective[0]),
            panAngle(-perspective[1], perspective[1]),
            cutoutApplication(0.0f, 1.0f),
            cutoutPos(0.0f, 1.0f),
            cutoutSize(cutout[0], cutout[1]),
            mixupApplication(0.0f, 1.0f),
            mixShift(-0.1f, 0.1f),
            hueShift(-hue, hue),
            saturationFactor(1 - saturation, 1 + saturation),
            valueFactor(1 - value, 1 + value);
        std::uniform_int_distribution<size_t>
            flipping(0, 1), mixIdx(1, batchSize - 1);
        std::gamma_distribution<> mixupGamma(mixupAlpha, 1);

        // sample transformation parameters
        for (size_t i = 0; i < paramsCpu.size(); ++i) {
            auto &img = paramsCpu[i];

            setColorTransform(img, hueShift(rnd), saturationFactor(rnd), valueFactor(rnd));

            const float
                scaleX = xScaleFactor(rnd),
                scaleY = isotropicScaling ? scaleX : yScaleFactor(rnd);
            setGeometricTransform(img, panAngle(rnd), tiltAngle(rnd), rotationAngle(rnd), scaleX, scaleY);

            img.translation[0] = xShiftFactor(rnd);
            img.translation[1] = yShiftFactor(rnd);
            img.flags = 0;
            if (flipHorizontally)
                img.flags = FLAG_HORIZONTAL_FLIP * flipping(rnd) + FLAG_MIX_HORIZONTAL_FLIP * flipping(rnd);
            if (flipVertically)
                img.flags += FLAG_VERTICAL_FLIP * flipping(rnd) + FLAG_MIX_VERTICAL_FLIP * flipping(rnd);

            // CutOut params
            if (cutoutApplication(rnd) < cutoutProb) {
                img.flags += FLAG_CUTOUT;
                img.cutoutPos[0] = cutoutPos(rnd);
                img.cutoutPos[1] = cutoutPos(rnd);
                img.cutoutSize[0] = 0.5f * cutoutSize(rnd);
                img.cutoutSize[1] = 0.5f * cutoutSize(rnd);
            }

            // Mixup params
            if (mixupApplication(rnd) < mixupProb) {
                img.mixImgIdx = (i + mixIdx(rnd)) % batchSize;
                float x = mixupGamma(rnd);
                img.mixFactor = x / (x + mixupGamma(rnd));      // beta distribution generation trick using gamma distribution
                if (img.mixFactor > 0.5)
                    img.mixFactor = 1 - img.mixFactor;
                    // make sure the current image has higher contribution to avoid duplicates
            }
            else {
                img.mixImgIdx = i;
            }
        }

        // create temporary tensor for parameters
        Tensor paramsGpu;
        const size_t paramsSizeBytes = sizeof(Params) * batchSize;
        OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT8, { (int64)paramsSizeBytes }, &paramsGpu));

        // copy parameters to GPU
        auto paramsGpuPtr = reinterpret_cast<Params*>(paramsGpu.flat<uint8_t>().data());
        CUDASSERT(cudaMemcpyAsync(paramsGpuPtr, paramsCpu.data(), paramsSizeBytes, cudaMemcpyHostToDevice, stream), "Cannot copy processing parameters to GPU");

        // create an output tensor
        const size_t
            outputWidth = outputSize.empty() ? inputWidth : outputSize[0],
            outputHeight = outputSize.empty() ? inputHeight : outputSize[1];
        Tensor* outputTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, { (int64)batchSize, (int64)outputHeight, (int64)outputWidth, 3 }, &outputTensor));

        // create output labels tensor
        Tensor* outputLabelsTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, labelsShape, &outputLabelsTensor));

        // compute the output
        try {
            compute(stream,
                    bufferPtr, outputTensor->flat<out_t>().data(),  // input and output pointers
                    inputWidth, inputHeight, pitchBytes,            // input sizes
                    outputWidth, outputHeight,                      // output sizes
                    batchSize,                                      // batch size
                    paramsGpuPtr);                                  // transformation description
        }
        catch (std::exception& ex) {
            context->CtxFailure(errors::InvalidArgument(ex.what()));
        }

        // fill output labels
        if (!noLabels) {
            const int64 numClasses = labelsShape.dim_size(1);
            const float* inLabel = inputLabelsPtr;
            float* outLabel = outputLabelsTensor->flat<float>().data();
            for (int64 n = 0; n < batchSize; ++n, inLabel += numClasses, outLabel += numClasses) {
                float f = paramsCpu[n].mixFactor;
                const float* mixLabel = inputLabelsPtr + paramsCpu[n].mixImgIdx * numClasses;
                for (int64 i = 0; i < numClasses; ++i)
                    outLabel[i] = (1 - f) * inLabel[i] + f * mixLabel[i];
            }
        }
    }
};


class SeedOpKernel : public OpKernel {
    int seed;
public:
    explicit SeedOpKernel(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    }

    void Compute(OpKernelContext* context) {
        rnd = std::default_random_engine(seed);
    }
};


// Register operations kernels
#define REGISTER_KERNEL(IN_T, OUT_T) \
    REGISTER_KERNEL_BUILDER(Name("Augment").Device(DEVICE_GPU).HostMemory("input_labels").HostMemory("output_labels"), DataugOpKernel<Eigen::GpuDevice, IN_T, OUT_T>)

REGISTER_KERNEL(uint8_t, float);

REGISTER_KERNEL_BUILDER(Name("SetSeed").Device(DEVICE_CPU), SeedOpKernel);


// Register operations
REGISTER_OP("Augment")
    .Input("input: uint8")
    .Input("input_labels: float")
    .Output("output: float")
    .Output("output_labels: float")
    .Attr("output_size:       list(int) = []")
    .Attr("translation:       list(float) = []")
    .Attr("scale:             list(float) = []")
    .Attr("prescale:          float = 1")
    .Attr("rotation:          float = 0")
    .Attr("perspective:       list(float) = []")
    .Attr("flip_horizontally: bool = false")
    .Attr("flip_vertically:   bool = false")
    .Attr("hue:               float = 0")
    .Attr("saturation:        float = 0")
    .Attr("value:             float = 0")
    .Attr("cutout_prob:       float = 0")
    .Attr("cutout_size:       list(float) = [0]")
    .Attr("mixup_prob:        float = 0")
    .Attr("mixup_alpha:       float = 0.4")
    .Doc(R"doc(
        Applies a set of randomized geometrical and color transformations to images in the batch.
        Each image is sampled only once through a bilinear interpolator.
        The exact transformation differs from one image to another one in the batch.

        For the geometrical transformations, the application order is as follows:
            - horizontal and/or vertical flipping,
            - perspective distortion,
            - in-plane image rotation and scaling,
            - translation.

        Args:
            input:              A `Tensor` containing the input batch in channels-last (NHWC) format. 3-channel color images are expected in the batch (C=3).
            input_labels:       A `Tensor` containing input labels in one-hot format. Optional, can be empty. If not empty, its outermost dimension is expected to match the batch size.
            output_size:        A list [W, H] specifying the output batch width and height in pixels. If not specified, the input batch size is used (default).
            translation:        Normalized image translation range along X and Y axis. `0.1` corresponds by shifting the image by at most 10% of its size in both directions.
                                If one value given, the same range applies for X and Y axes. If empty, no translation is applied.
            scale:              Scaling factor range along X and Y axes. `0.1` corresponds to stretching the image by at most 10%.
                                If one value given, the applied scaling keeps the aspect ratio: the same factor is used along X and Y axes. If empty, no scaling is applied.
            prescale:           A constant scaling factor applied to all images. Can be used to shift the random scaling factor distribution from its default average equal to 1 and
                                crop out image borders.
            rotation:           Rotation angle range in degrees. The images are rotated in both clockwise and counter-clockwise direction by a random angle limited by the given value.
            perspective:        Perspective distortion range: maximum tilting and panning angles in degrees.
                                The image plane is rotated in 3D around X and Y axes (tilt and pan respectively) by random angles limited by the passed values in both directions.
                                If one value given, the same range applies for both axes. If empty, no perspective distoriton is induced.
            flip_horizontally:  A boolean. If `True`, the images are flipped horizontally with 50% chance.
            flip_vertically:    A boolean. If `True`, the images are flipped vertically with 50% chance.
            hue:                Hue shift range in degrees. The image pixels color hues are shifted by a random angle limited by the given value.
                                A hue shift of +/-120 degrees transforms green in red/blue and vice versa.
            saturation:         Color saturation factor range. The image pixels color saturation values are scaled by a random factor sampled in range `[1 - s, 1 + s]` for `s` equal to `saturation`.
                                A fully desaturated image becomes grayscale.
            value:              Value (brightness) factor range. The image pixels values are scaled by a random factor sampled in range `[1 - v, 1 + v]` for `s` equal to `value`.
                                Scaling the image by a small factor makes it darker.
            cutout_prob:        Probability of CutOut being applied to an image in the batch.
                                CutOut erases a part of an image. See the original paper for more details: https://arxiv.org/pdf/1708.04552.pdf
            cutout_size:        CutOut size normalized range. Width and height of erased areas are sampled from this given range. If a single value is given, the sizes are constant.
                                If no value is provided, CutOut application is disabled.
            mixup_prob:         Probability of Mixup being applied to an image in the batch.
                                Mixup is applied across the batch. Every two images being mixed undergo the same set transformations except for flipping.
            mixup_alpha:        Mixup `alpha` parameter. See the original paper for more details: https://arxiv.org/pdf/1710.09412.pdf

        Returns:
            A `Tensor` with a set of transformations applied to images in the input batch, and another `Tensor` containing the image labels in one-hot format.
    )doc")

    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* ctx) {
        // get output size parameter
        std::vector<int64> outputSize;
        TF_RETURN_IF_ERROR(ctx->GetAttr("output_size", &outputSize));
        if (!outputSize.empty() && outputSize.size() != 2)
            return errors::InvalidArgument("Invalid output_size: expected an empty list or a list of 2 entries, got " + std::to_string(outputSize.size()));

        // return output shape
        if (outputSize.empty())
            ctx->set_output(0, ctx->input(0));
        else
            ctx->set_output(0, ctx->MakeShape({
                ctx->Dim(ctx->input(0), 0),
                outputSize[0],
                outputSize[1],
                3
            }));

        ctx->set_output(1, ctx->input(1));

        return Status::OK();
    });


REGISTER_OP("SetSeed")
    .Attr("seed: int");