#define EIGEN_USE_GPU

#include <vector>
#include <random>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#pragma GCC diagnostic pop

#include "kernel_base.hpp"


#define CUDASSERT(CALL, MSG) \
        OP_REQUIRES(context, CALL == cudaSuccess, errors::Internal(MSG));


template <typename val_t, typename mod_t>
inline val_t roundUp(val_t value, mod_t modulo) {
    return (value + modulo - 1) / modulo * modulo;
}


using namespace tensorflow;


class TFTempGPUBuffer {
private:
    Tensor buffer;
public:
    TFTempGPUBuffer(size_t sizeBytes, OpKernelContext* context) {
        OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT8, { (int64_t)sizeBytes }, &buffer));
    }

    inline uint8_t* operator()() {
        return buffer.flat<uint8_t>().data();
    }
};


template<class Device, typename in_t, typename out_t>
class FastAugmentTFOpKernel :
    public OpKernel,
    protected fastaugment::KernelBase<TFTempGPUBuffer, OpKernelContext*>,
    private fastaugment::Settings
{
    std::vector<int64_t> outputSize;

    size_t getPair(OpKernelConstruction* context, const char* attribute, float& a, float& b) {
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
    explicit FastAugmentTFOpKernel(OpKernelConstruction* context):
        OpKernel(context)
    {
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

        // get flipping flags
        OP_REQUIRES_OK(context, context->GetAttr("flip_horizontally", &flipHorizontally));
        OP_REQUIRES_OK(context, context->GetAttr("flip_vertically", &flipVertically));

        // get CutOut parameters
        OP_REQUIRES_OK(context, context->GetAttr("cutout_prob", &cutoutProb));
        n = getPair(context, "cutout_size", cutout[0], cutout[1]);
        if (n == 0)
            cutoutProb = 0;

        // get Mixup parameters
        OP_REQUIRES_OK(context, context->GetAttr("mixup_prob", &mixupProb));
        OP_REQUIRES_OK(context, context->GetAttr("mixup_alpha", &mixupAlpha));

        // get HSV transformation magnitudes
        OP_REQUIRES_OK(context, context->GetAttr("hue", &hue));
        hue *= pi / 180.0f;
        OP_REQUIRES_OK(context, context->GetAttr("saturation", &saturation));
        OP_REQUIRES_OK(context, context->GetAttr("brightness", &brightness));

        // get gamma correction range
        OP_REQUIRES_OK(context, context->GetAttr("gamma_corr", &gammaCorrection));

        // get color inversion flag
        OP_REQUIRES_OK(context, context->GetAttr("color_inversion", &colorInversion));

        // get random seed
        OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));

        // check parameters
        try {
            check();
        }
        catch (std::exception& ex) {
            context->CtxFailure(errors::InvalidArgument(ex.what()));
        }

        // prepare texture parameters
        try {
            fastaugment::KernelBase<TFTempGPUBuffer, OpKernelContext*>::queryTextureParams();
        }
        catch (std::exception& ex) {
            context->CtxFailure(errors::Internal(ex.what()));
        }
    }


    void Compute(OpKernelContext* context) override {
        using namespace fastaugment;

        // grab the input tensor
        const Tensor& inputTensor = context->input(0);
        const auto& inputShape = inputTensor.shape();
        OP_REQUIRES(context, inputShape.dims() == 3 || inputShape.dims() == 4,
            errors::InvalidArgument("Expected a 3- or 4-dimensional input tensor, got " + std::to_string(inputShape.dims()) + " dimensions"));

        // get input sizes
        const bool isBatch = inputShape.dims() == 4;
        const int64_t
            batchSize = isBatch ? inputShape.dim_size(0) : 0,
            inputHeight = inputShape.dim_size(isBatch ? 1 : 0),
            inputWidth = inputShape.dim_size(isBatch ? 2 : 1),
            inputChannels = inputShape.dim_size(isBatch ? 3 : 2),
            outputWidth = outputSize.empty() ? inputWidth : outputSize[0],
            outputHeight = outputSize.empty() ? inputHeight : outputSize[1];

        // check number of input channels
        OP_REQUIRES(context, inputChannels == 3, errors::InvalidArgument("Expected a 3-channel input tensor, got " + std::to_string(inputChannels) + " channels"));

        // get CUDA stream
        auto stream = context->eigen_device<Device>().stream();

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

        // create an output tensor
        Tensor* outputTensor = nullptr;
        if (isBatch)
            OP_REQUIRES_OK(context, context->allocate_output(0, { batchSize, outputHeight, outputWidth, 3 }, &outputTensor));
        else
            OP_REQUIRES_OK(context, context->allocate_output(0, { outputHeight, outputWidth, 3 }, &outputTensor));

        // create output labels tensor
        Tensor* outputLabelsTensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, labelsShape, &outputLabelsTensor));

        // compute the output
        try {
            fastaugment::KernelBase<TFTempGPUBuffer, OpKernelContext*>::run<in_t, out_t>(
                *this,
                inputTensor.flat<in_t>().data(),
                outputTensor->flat<out_t>().data(),
                inputLabelsPtr,
                outputLabelsTensor->flat<float>().data(),
                batchSize,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                noLabels ? 0 : labelsShape.dim_size(1),
                stream,
                context
            );
        }
        catch (std::exception& ex) {
            context->CtxFailure(errors::InvalidArgument(ex.what()));
        }
    }
};


// Register operations kernels
#define REGISTER_KERNEL(IN_T, OUT_T) \
    REGISTER_KERNEL_BUILDER(                            \
            Name("Augment")                             \
            .Device(DEVICE_GPU)                         \
            .TypeConstraint<IN_T>("input_type")       \
            .TypeConstraint<OUT_T>("output_type")       \
            .HostMemory("input_labels")                 \
            .HostMemory("output_labels"),               \
        FastAugmentTFOpKernel<Eigen::GpuDevice, IN_T, OUT_T>)

REGISTER_KERNEL(uint8_t, uint8_t);
REGISTER_KERNEL(uint8_t, float);
REGISTER_KERNEL(float, float);


// Register operations
REGISTER_OP("Augment")
    .Attr("input_type: {uint8, float}")
    .Attr("output_type: {uint8, float}")
    .Input("input: input_type")
    .Input("input_labels: float")
    .Output("output: output_type")
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
    .Attr("brightness:        float = 0")
    .Attr("gamma_corr:        float = 0")
    .Attr("color_inversion:   bool = false")
    .Attr("cutout_prob:       float = 0")
    .Attr("cutout_size:       list(float) = [0]")
    .Attr("mixup_prob:        float = 0")
    .Attr("mixup_alpha:       float = 0.4")
    .Attr("seed:              int = 0")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* ctx) {
        auto inputRank = ctx->Rank(ctx->input(0));
        if (inputRank != 3 && inputRank != 4)
            errors::InvalidArgument("Expected a 3- or 4-dimensional input tensor, got " + std::to_string(inputRank) + " dimensions");

        // get output size parameter
        std::vector<int64_t> outputSize;
        TF_RETURN_IF_ERROR(ctx->GetAttr("output_size", &outputSize));
        if (!outputSize.empty() && outputSize.size() != 2)
            return errors::InvalidArgument("Invalid output_size: expected an empty list or a list of 2 entries, got " + std::to_string(outputSize.size()));

        // return output shape
        if (outputSize.empty())
            ctx->set_output(0, ctx->input(0));
        else if (inputRank == 3)
            ctx->set_output(0, ctx->MakeShape({
                outputSize[0],
                outputSize[1],
                3
            }));
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