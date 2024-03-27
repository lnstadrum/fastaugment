#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "kernel_base.hpp"

void normalizePair(const std::vector<float> input, const char *attrName, float &x, float &y)
{
    if (input.size() == 0 || input.size() > 2)
        throw std::invalid_argument("Expecting one or two entries in '" + std::string(attrName) +
                                    "' argument, but got " + std::to_string(input.size()) + ".");

    if (input.size() == 1)
        x = y = input[0];
    else
    {
        x = input[0];
        y = input[1];
    }
}

/**
 * Temporary GPU memory buffer implementation using CUDACachingAllocator
 * The cache needs to be cleaned.
 */
class TorchTempGPUBuffer
{
  private:
    c10::DataPtr ptr;
    c10::cuda::CUDAStream stream;

  public:
    TorchTempGPUBuffer(size_t sizeBytes, c10::cuda::CUDAStream stream)
        : ptr(c10::cuda::CUDACachingAllocator::get()->allocate(sizeBytes)), stream(stream)
    {
    }

    ~TorchTempGPUBuffer()
    {
        c10::cuda::CUDACachingAllocator::recordStream(ptr, stream);
    }

    inline uint8_t *operator()()
    {
        return reinterpret_cast<uint8_t *>(ptr.get());
    }
};

class TorchKernel : public fastaugment::KernelBase<TorchTempGPUBuffer, c10::cuda::CUDAStream>
{
  private:
    using Base = fastaugment::KernelBase<TorchTempGPUBuffer, c10::cuda::CUDAStream>;

    fastaugment::Settings settings;

    template <typename... Args>
    inline void launchKernel(const torch::Tensor &input, torch::Tensor &output, Args... args)
    {
        if (input.scalar_type() == torch::kUInt8)
        {
            if (output.scalar_type() == torch::kUInt8)
            {
                Base::run(settings, input.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(), args...);
                return;
            }
            if (output.scalar_type() == torch::kFloat)
            {
                Base::run(settings, input.data_ptr<uint8_t>(), output.data_ptr<float>(), args...);
                return;
            }
        }
        else if (input.scalar_type() == torch::kFloat)
        {
            if (output.scalar_type() == torch::kFloat)
            {
                Base::run(settings, input.data_ptr<float>(), output.data_ptr<float>(), args...);
                return;
            }
        }

        throw std::runtime_error("Unsupported input/output datatype combination");
    }

  public:
    TorchKernel(const std::vector<float> &translation, const std::vector<float> &scale, float prescale, float rotation,
                const std::vector<float> &perspective, float cutout, const std::vector<float> &cutoutSize, float mixup,
                float mixupAlpha, float hue, float saturation, float brightness, float gammaCorrection,
                bool colorInversion, bool flipHorizontally, bool flipVertically, int seed)
    {
        // get translation range
        normalizePair(translation, "translation", settings.translation[0], settings.translation[1]);

        // get scaling parameters
        normalizePair(scale, "scale", settings.scale[0], settings.scale[1]);
        settings.isotropicScaling = (scale.size() == 1);
        settings.prescale = prescale;

        // get rotation in radians
        settings.rotation = rotation * pi / 180.0f;

        // get perspective angles and convert to radians
        normalizePair(perspective, "perspective", settings.perspective[0], settings.perspective[1]);
        settings.perspective[0] *= pi / 180.0f;
        settings.perspective[1] *= pi / 180.0f;

        // get flipping flags
        settings.flipHorizontally = flipHorizontally;
        settings.flipVertically = flipVertically;

        // get CutOut parameters
        settings.cutoutProb = cutout;
        settings.cutout[0] = settings.cutout[1] = 0.0f;
        if (cutoutSize.empty())
            settings.cutoutProb = 0;
        else
            normalizePair(cutoutSize, "cutout_size", settings.cutout[0], settings.cutout[1]);

        // get Mixup parameters
        settings.mixupProb = mixup;
        settings.mixupAlpha = mixupAlpha;

        // get color correction parameters
        settings.hue = hue * pi / 180.0f;
        settings.saturation = saturation;
        settings.brightness = brightness;
        settings.colorInversion = colorInversion;

        // get gamma correction range
        settings.gammaCorrection = gammaCorrection;

        // set random seed
        if (seed != 0)
            fastaugment::KernelBase<TorchTempGPUBuffer, c10::cuda::CUDAStream>::setRandomSeed(seed);

        // check settings
        settings.check();

        // query texture parameters
        Base::queryTextureParams();
    }

    std::vector<torch::Tensor> operator()(const torch::Tensor &input, const torch::Tensor &labels,
                                          const std::vector<int64_t> &outputSize, bool isFloat32Output,
                                          bool outputMapping)
    {
        // check output size
        if (!outputSize.empty() && outputSize.size() != 2)
            throw std::invalid_argument("Invalid output_size: expected an empty list or a list of 2 entries, "
                                        "got " +
                                        std::to_string(outputSize.size()));

        // check the input tensor
        if (input.dim() < 3 || input.dim() > 5)
            throw std::invalid_argument("Expected a 3-, 4- or 5-dimensional input tensor, got " +
                                        std::to_string(input.dim()) + " dimensions");
        if (!input.is_cuda())
            throw std::invalid_argument("Expected an input tensor in GPU memory (likely missing a .cuda() "
                                        "call)");
        if (!input.is_contiguous())
            throw std::invalid_argument("Expected a contiguous input tensor");
        if (input.scalar_type() != torch::kUInt8 && input.scalar_type() != torch::kFloat)
            throw std::invalid_argument("Expected uint8 or float input tensor");

        // get input sizes
        const int64_t batchSize = input.dim() >= 4 ? input.size(0) : 0, groups = input.dim() == 5 ? input.size(1) : 0,
                      inputHeight = input.size(input.dim() - 3), inputWidth = input.size(input.dim() - 2),
                      inputChannels = input.size(input.dim() - 1),
                      outputWidth = outputSize.empty() ? inputWidth : outputSize[0],
                      outputHeight = outputSize.empty() ? inputHeight : outputSize[1];

        // check number of input channels
        if (inputChannels != 3)
            throw std::invalid_argument("Expected a 3-channel channels-last (*HW3) input tensor, got " +
                                        std::to_string(inputChannels) + " channels");

        // get CUDA stream
        auto stream = c10::cuda::getCurrentCUDAStream(input.device().index());

        // get input labels tensor
        const bool noLabels = labels.dim() == 0 || (labels.dim() == 1 && labels.size(0) == 0);
        const auto numClasses = noLabels ? 0 : labels.size(labels.dim() - 1);
        if (!noLabels)
        {
            const auto expectedDims = input.dim() - 2;
            if (labels.dim() != expectedDims)
                throw std::invalid_argument("Expected a " + std::to_string(expectedDims) +
                                            "-dimensional input_labels tensor, got " + std::to_string(labels.dim()) +
                                            " dimensions");
            if (labels.size(0) != batchSize)
                throw std::invalid_argument("First dimension of the input labels tensor is expected to match "
                                            "the batch size, but got " +
                                            std::to_string(labels.size(0)));

            const auto expectedNumElems = batchSize * std::max<int64_t>(groups, 1) * numClasses;
            if (labels.numel() != expectedNumElems)
                throw std::invalid_argument("Expected " + std::to_string(expectedNumElems) +
                                            " elements in the labels tensor but got " + std::to_string(labels.numel()));

            if (!labels.is_cpu())
                throw std::invalid_argument("Expected an input_labels tensor stored in RAM (likely missing a "
                                            ".cpu() call)");
            if (labels.scalar_type() != torch::kFloat)
                throw std::invalid_argument("Expected a floating-point input_labels tensor");
        }
        const float *inputLabelsPtr = noLabels ? nullptr : labels.expect_contiguous()->data_ptr<float>();

        // allocate output tensors
        auto outputOptions =
            torch::TensorOptions().device(input.device()).dtype(isFloat32Output ? torch::kFloat32 : torch::kUInt8);
        std::vector<int64_t> outputShape{outputHeight, outputWidth, 3};
        if (groups > 0)
            outputShape.emplace(outputShape.begin(), groups);
        if (batchSize > 0)
            outputShape.emplace(outputShape.begin(), batchSize);

        torch::Tensor output = torch::empty(outputShape, outputOptions);
        torch::Tensor outputLabels = torch::empty_like(labels);

        torch::Tensor mapping;
        if (outputMapping)
        {
            auto opts = torch::TensorOptions().dtype(torch::kFloat32);
            mapping = batchSize > 0 ? torch::empty({batchSize, 3, 3}, opts) : torch::empty({3, 3}, opts);
        }
        auto outputMappingPtr = outputMapping ? mapping.expect_contiguous()->data_ptr<float>() : nullptr;

        // launch the kernel
        launchKernel(input, output, inputLabelsPtr, outputLabels.data_ptr<float>(), outputMappingPtr,
                     std::max<int64_t>(batchSize, 1), std::max<int64_t>(groups, 1), inputHeight, inputWidth,
                     outputHeight, outputWidth, numClasses, stream.stream(), stream);

        return {output, outputLabels, mapping};
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    py::class_<TorchKernel>(module, "FastAugment")
        .def(py::init<const std::vector<float> &, const std::vector<float> &, float, float, const std::vector<float> &,
                      float, const std::vector<float> &, float, float, float, float, float, float, bool, bool, bool,
                      int>(),
             py::arg("translation") = std::vector<float>{0}, py::arg("scale") = std::vector<float>{0},
             py::arg("prescale") = 1.0f, py::arg("rotation") = 0.0f, py::arg("perspective") = std::vector<float>{0},
             py::arg("cutout") = 0.0f, py::arg("cutout_size") = std::vector<float>{}, py::arg("mixup") = 0.0f,
             py::arg("mixup_alpha") = 0.4f, py::arg("hue") = 0.0f, py::arg("saturation") = 0.0f,
             py::arg("brightness") = 0.0f, py::arg("gamma_corr") = 0.0f, py::arg("color_inversion") = false,
             py::arg("flip_horizontally") = false, py::arg("flip_vertically") = false, py::arg("seed") = 0)

        .def("set_seed", &TorchKernel::setRandomSeed, py::arg("seed"))

        .def("__call__", &TorchKernel::operator(), py::arg("input"), py::arg("input_labels"), py::arg("output_size"),
             py::arg("is_float32_output"), py::arg("output_mapping") = false);
}