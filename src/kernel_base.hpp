#pragma once

#include <random>
#include <vector>

#include "augment.h"

static const float pi = 3.141592653589f;

namespace fastaugment
{
static std::random_device randomDevice;

/**
 * @brief Data augmentation settings.
 * Define the random distribution to sample per-image augmentation parameters
 * from.
 */
typedef struct
{
    float translation[2];  //!< normalized translation range in X and Y directions
    float scale[2];        //!< scaling factor range in X and Y directions
    float prescale;        //!< constant scaling factor
    float rotation;        //!< in-plane rotation range in radians
    float perspective[2];  //!< out-of-plane rotation range in radians around
                           //!< horizontal and vertical axes (tilt and pan)
    float cutoutProb;      //!< CutOut per-image probability
    float cutout[2];       //!< CutOut normalized size range
    float mixupProb;       //!< Mixup per-image probability
    float mixupAlpha;      //!< Mixup alpha parameter
    float hue;             //!< hue shift range in radians
    float saturation;      //!< saturation factor range
    float brightness;      //!< brightness range
    float gammaCorrection; //!< gamma correction range
    bool colorInversion;   //!< if `true`, colors in every image are inverted with
                           //!< 50% chance
    bool flipHorizontally,
        flipVertically;    //!< if `true`, the image is flipped with 50% chance
                           //!< along a specific direction
    bool isotropicScaling; //!< if `true`, the scale factor is the same along X
                           //!< and Y direction

    inline void check()
    {
        if (mixupProb < 0 || mixupProb > 1)
            throw std::invalid_argument("Invalid Mixup probability: " + std::to_string(mixupProb) +
                                        ", expected a value in [0, 1] range");

        if (cutoutProb < 0 || cutoutProb > 1)
            throw std::invalid_argument("Invalid CutOut probability: " + std::to_string(cutoutProb) +
                                        ", expected a value in [0, 1] range");

        if (gammaCorrection < 0 || gammaCorrection > 0.9)
            throw std::invalid_argument("Bad gamma correction factor range: " + std::to_string(gammaCorrection) +
                                        ". Expected a value in [0, 0.9] range.");
    }
} Settings;

/**
 * @brief Frontend-agnostic base class performing the data augmentation.
 * @tparam in_t                     input tensor scalar datatype
 * @tparam out_t                    output tensor scalar datatype
 * @tparam TempGPUBuffer            class representing temporary GPU buffers
 * @tparam BufferAllocationArgs     arguments to send to TempGPUBuffer
 * constructor
 */
template <class TempGPUBuffer, typename... BufferAllocationArgs> class KernelBase
{
  private:
    size_t textureAlignment, texturePitchAlignment;
    size_t maxTextureHeight; //!< maximum texture height in pixels allowed by the
                             //!< GPU

    template <typename val_t, typename mod_t> static inline val_t roundUp(val_t value, mod_t modulo)
    {
        return (value + modulo - 1) / modulo * modulo;
    }

  protected:
    std::default_random_engine rnd;

    KernelBase() : rnd(randomDevice())
    {
    }

    static inline void reportCudaError(cudaError_t status, const std::string &message)
    {
        if (status != cudaSuccess)
            throw std::runtime_error(message);
    }

    /**
     * @brief Queries GPU for texture alignment parameters and texture limits.
     */
    void queryTextureParams()
    {
        // get current device number
        int device;
        reportCudaError(cudaGetDevice(&device), "Cannot get current CUDA device number");

        // query device properties
        cudaDeviceProp properties;
        reportCudaError(cudaGetDeviceProperties(&properties, device), "Cannot get CUDA device properties");

        // get alignment values
        textureAlignment = properties.textureAlignment;
        texturePitchAlignment = properties.texturePitchAlignment;

        // get max texture height
        maxTextureHeight = (size_t)properties.maxTexture2D[1];
    }

    /**
     * @brief Performs data augmentation.
     * Samples augmentation parameters per image in the batch, sends them to GPU
     * memory, launches the CUDA kernel.
     * @param settings          Randomization settings
     * @param inputPtr          Pointer to the input batch tensor in GPU memory
     * @param outputPtr         Pointer to the output batch tensor in GPU memory
     * @param inputLabelsPtr    Pointer to the input class probabilities tensor
     * in host memory
     * @param outputLabelsPtr   Pointer to the output class probabilities tensor
     * in host memory
     * @param outputMappingPtr  Pointer to the output homography tensor in host memory
     * @param batchSize         Batch size; 0 if 3-dimensional input tensor is
     * given
     * @param inputHeight       Input batch height in pixels
     * @param inputWidth        Input batch width in pixels
     * @param outputHeight      Output batch height in pixels
     * @param outputWidth       Output batch width in pixels
     * @param numClasses        Number of classes
     * @param stream            CUDA stream
     * @param allocationArgs    Frontend-specific TempGPUBuffer allocation
     * arguments
     */
    template <typename in_t, typename out_t>
    void run(const Settings &settings, const in_t *inputPtr, out_t *outputPtr, const float *inputLabelsPtr,
             float *outputLabelsPtr, float *outputMappingPtr, int64_t batchSize, int64_t inputHeight,
             int64_t inputWidth, int64_t outputHeight, int64_t outputWidth, int64_t numClasses, cudaStream_t stream,
             BufferAllocationArgs... allocationArgs)
    {
        // correct batchSize value (can be zero if input is a 3-dim tensor)
        const bool isBatch = batchSize > 0;
        if (!isBatch)
            batchSize = 1;

        // compute scale factors to keep the aspect ratio
        float arScaleX = 1, arScaleY = 1;
        if (inputWidth * outputHeight >= inputHeight * outputWidth)
            arScaleX = (float)(outputWidth * inputHeight) / (inputWidth * outputHeight);
        else
            arScaleY = (float)(outputHeight * inputWidth) / (inputHeight * outputWidth);

        // allocate a temporary buffer ensuring starting address and pitch
        // alignment
        const int64_t pitchBytes = roundUp(inputWidth * 4 * sizeof(in_t), texturePitchAlignment);
        const size_t bufferSizeBytes = textureAlignment + batchSize * inputHeight * pitchBytes;
        TempGPUBuffer buffer(bufferSizeBytes, allocationArgs...);
        auto unalignedBufferAddress = buffer();

        // align its address
        auto bufferPtr =
            reinterpret_cast<in_t *>(roundUp(reinterpret_cast<size_t>(unalignedBufferAddress), textureAlignment));

        // pad the input to have 4 channels and an aligned pitch
        padChannels(stream, inputPtr, bufferPtr, inputWidth, inputHeight, batchSize, pitchBytes / (4 * sizeof(in_t)));
        reportCudaError(cudaGetLastError(), "Cannot pad the input image");

        // check if no labels but mixup
        if (inputLabelsPtr == nullptr && settings.mixupProb != 0)
            throw std::runtime_error("Cannot apply mixup: input class probabilities are not provided");

        // prepare parameters samplers
        std::vector<Params> paramsCpu;
        paramsCpu.resize(batchSize);
        std::uniform_real_distribution<float> xShiftFactor(-settings.translation[0], settings.translation[0]),
            yShiftFactor(-settings.translation[1], settings.translation[1]),
            xScaleFactor(settings.prescale - settings.scale[0], settings.prescale + settings.scale[0]),
            yScaleFactor(settings.prescale - settings.scale[1], settings.prescale + settings.scale[1]),
            rotationAngle(-settings.rotation, settings.rotation),
            tiltAngle(-settings.perspective[0], settings.perspective[0]),
            panAngle(-settings.perspective[1], settings.perspective[1]), cutoutApplication(0.0f, 1.0f),
            cutoutPos(0.0f, 1.0f), cutoutSize(settings.cutout[0], settings.cutout[1]), mixupApplication(0.0f, 1.0f),
            mixShift(-0.1f, 0.1f), hueShift(-settings.hue, settings.hue),
            saturationFactor(1 - settings.saturation, 1 + settings.saturation),
            brightnessFactor(1 - settings.brightness, 1 + settings.brightness),
            gammaCorrFactor(1 - settings.gammaCorrection, 1 + settings.gammaCorrection);
        std::uniform_int_distribution<size_t> flipping(0, 1), mixIdx(1, batchSize - 1);
        std::gamma_distribution<> mixupGamma(settings.mixupAlpha, 1);

        // sample transformation parameters
        for (size_t i = 0; i < paramsCpu.size(); ++i)
        {
            auto &img = paramsCpu[i];
            img.flags = 0;

            // color correction
            setColorTransform(img, hueShift(rnd), saturationFactor(rnd), brightnessFactor(rnd));
            img.gammaCorr = gammaCorrFactor(rnd);
            if (settings.colorInversion)
                img.flags += FLAG_COLOR_INVERSION * flipping(rnd);

            // geometric transform (homography)
            const float scaleX = xScaleFactor(rnd), scaleY = settings.isotropicScaling ? scaleX : yScaleFactor(rnd);
            setGeometricTransform(img, panAngle(rnd), tiltAngle(rnd), rotationAngle(rnd), arScaleX * scaleX,
                                  arScaleY * scaleY);

            // translation and flipping
            img.translation[0] = xShiftFactor(rnd);
            img.translation[1] = yShiftFactor(rnd);
            if (settings.flipHorizontally)
                img.flags += FLAG_HORIZONTAL_FLIP * flipping(rnd) + FLAG_MIX_HORIZONTAL_FLIP * flipping(rnd);
            if (settings.flipVertically)
                img.flags += FLAG_VERTICAL_FLIP * flipping(rnd) + FLAG_MIX_VERTICAL_FLIP * flipping(rnd);

            // CutOut params
            if (cutoutApplication(rnd) < settings.cutoutProb)
            {
                img.flags += FLAG_CUTOUT;
                img.cutoutPos[0] = cutoutPos(rnd);
                img.cutoutPos[1] = cutoutPos(rnd);
                img.cutoutSize[0] = 0.5f * cutoutSize(rnd);
                img.cutoutSize[1] = 0.5f * cutoutSize(rnd);
            }

            // Mixup params
            if (mixupApplication(rnd) < settings.mixupProb)
            {
                img.mixImgIdx = (i + mixIdx(rnd)) % batchSize;
                float x = mixupGamma(rnd);
                img.mixFactor = x / (x + mixupGamma(rnd)); // beta distribution generation
                                                           // trick using gamma distribution
                if (img.mixFactor > 0.5)
                    img.mixFactor = 1 - img.mixFactor;
                // making sure the current image has higher contribution to avoid
                // duplicates
            }
            else
            {
                img.mixImgIdx = i;
            }
        }

        // create temporary tensor for parameters
        const size_t paramsSizeBytes = sizeof(Params) * batchSize;
        TempGPUBuffer paramsGpu(paramsSizeBytes, allocationArgs...);

        // copy parameters to GPU
        auto paramsGpuPtr = reinterpret_cast<Params *>(paramsGpu());
        reportCudaError(
            cudaMemcpyAsync(paramsGpuPtr, paramsCpu.data(), paramsSizeBytes, cudaMemcpyHostToDevice, stream),
            "Cannot copy processing parameters to GPU");

        // compute the output
        try
        {
            compute(stream, bufferPtr,
                    outputPtr, // input and output pointers
                    inputWidth, inputHeight,
                    pitchBytes, // input sizes
                    outputWidth,
                    outputHeight, // output sizes
                    batchSize,    // batch size
                    maxTextureHeight,
                    paramsGpuPtr); // transformation description
        }
        catch (std::exception &ex)
        {
            throw std::runtime_error(ex.what());
        }

        // fill output labels
        if (inputLabelsPtr && outputLabelsPtr)
        {
            const float *inLabel = inputLabelsPtr;
            float *outLabel = outputLabelsPtr;
            for (int64_t n = 0; n < batchSize; ++n, inLabel += numClasses, outLabel += numClasses)
            {
                float f = paramsCpu[n].mixFactor;
                const float *mixLabel = inputLabelsPtr + paramsCpu[n].mixImgIdx * numClasses;
                for (int64_t i = 0; i < numClasses; ++i)
                    outLabel[i] = (1 - f) * inLabel[i] + f * mixLabel[i];
            }
        }

        // fill output mapping tensor
        if (outputMappingPtr)
        {
            float *ptr = outputMappingPtr;
            for (size_t i = 0; i < paramsCpu.size(); ++i, ptr += 9)
            {
                // compute homography in normalized coordinates following the kernel implementation
                const auto &a = paramsCpu[i].geom;
                ptr[0] = 2.0f * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
                ptr[1] = -2.0f * (a[1][0] * a[2][2] - a[1][2] * a[2][0]);
                ptr[2] = a[2][2] * (a[1][0] - a[1][1]) + a[1][2] * (a[2][1] - a[2][0]);

                ptr[3] = -2.0f * (a[0][1] * a[2][2] - a[0][2] * a[2][1]);
                ptr[4] = 2.0f * (a[0][0] * a[2][2] - a[0][2] * a[2][0]);
                ptr[5] = a[2][2] * (a[0][1] - a[0][0]) + a[0][2] * (a[2][0] - a[2][1]);

                ptr[6] = 2.0f * (a[0][1] * a[1][2] - a[0][2] * a[1][1]);
                ptr[7] = -2.0f * (a[0][0] * a[1][2] - a[0][2] * a[1][0]);
                ptr[8] =
                    2.0f * (a[0][0] * a[1][1] * a[2][2] - a[0][0] * a[1][2] * a[2][1] - a[0][1] * a[1][0] * a[2][2] +
                            a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1] - a[0][2] * a[1][1] * a[2][0]) +
                    a[0][2] * (a[1][1] - a[1][0]) + a[1][2] * (a[0][0] - a[0][1]);

                // take into account flipping
                if (paramsCpu[i].flags & FLAG_HORIZONTAL_FLIP)
                {
                    ptr[2] += ptr[0];
                    ptr[0] = -ptr[0];
                    ptr[5] += ptr[3];
                    ptr[3] = -ptr[3];
                    ptr[8] += ptr[6];
                    ptr[6] = -ptr[6];
                }

                if (paramsCpu[i].flags & FLAG_VERTICAL_FLIP)
                {
                    ptr[2] += ptr[1];
                    ptr[1] = -ptr[1];
                    ptr[5] += ptr[4];
                    ptr[4] = -ptr[4];
                    ptr[8] += ptr[7];
                    ptr[7] = -ptr[7];
                }

                // use input pixel coordinates
                ptr[0] /= inputWidth;
                ptr[1] /= inputHeight;
                ptr[2] += 0.5f * (ptr[0] + ptr[1]);
                ptr[3] /= inputWidth;
                ptr[4] /= inputHeight;
                ptr[5] += 0.5f * (ptr[3] + ptr[4]);
                ptr[6] /= inputWidth;
                ptr[7] /= inputHeight;
                ptr[8] += 0.5f * (ptr[6] + ptr[7]);

                // take into account the random translation
                const float *translation = paramsCpu[i].translation;
                ptr[0] -= (translation[0] - 0.5f) * ptr[6];
                ptr[1] -= (translation[0] - 0.5f) * ptr[7];
                ptr[2] -= (translation[0] - 0.5f) * ptr[8];
                ptr[3] -= (translation[1] - 0.5f) * ptr[6];
                ptr[4] -= (translation[1] - 0.5f) * ptr[7];
                ptr[5] -= (translation[1] - 0.5f) * ptr[8];

                // use output pixel coordinates
                ptr[0] *= outputWidth;
                ptr[1] *= outputWidth;
                ptr[2] *= outputWidth;
                ptr[3] *= outputHeight;
                ptr[4] *= outputHeight;
                ptr[5] *= outputHeight;
                ptr[0] -= 0.5f * ptr[6];
                ptr[1] -= 0.5f * ptr[7];
                ptr[2] -= 0.5f * ptr[8];
                ptr[3] -= 0.5f * ptr[6];
                ptr[4] -= 0.5f * ptr[7];
                ptr[5] -= 0.5f * ptr[8];
            }
        }
    }

  public:
    /**
     * @brief Seeds the random number generator
     *
     * @param seed              the seed value
     */
    void setRandomSeed(int seed)
    {
        rnd.seed(seed);
    }
};
} // namespace fastaugment
