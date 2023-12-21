# TCD-Computer-Vision-Project-2023

This project seeks to implement the Anchor-based Plain Net for Mobile Image Super-Resolution [Du Zongcai, et al. 2021] in Tensorflow 2.10.1 and python 3.10.13 on Windows 11 64-bit.

**Please note** that as of the time of writing, the latest Python version is 3.12, the latest Windows OS is 11, and while the latest Tensorflow version is 2.15, versions 2.11 onwards no longer support GPU acceleration on Windows Native. Python 3.11 onwards does not support Tensorflow 2.10 or earlier. You may use later versions of Python and Tensorflow on Windows with WSL2, or on other operating systems. The provided conda environment files assume native Windows, thus there are separate environments depending on whether you wish to train on GPU or CPU. We recommend using GPU if possible, as training on CPU is very slow.

## Requirements

### Microsoft Windows

- Windows 7 or higher (64-bit)
- Modern AVX supported CPU (see [CPU support](https://www.tensorflow.org/install/source_windows#cpu_support))
- [Win32 Long Paths enabled](https://superuser.com/a/1119980)
- [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, 2019, and 2022](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)
- [Miniconda for Windows](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
- For GPU support: An Nvidia GPU with CUDA 3.5 or higher and Nvidia GPU drivers 450.80.02 or higher

### Other Operating Systems

See the [Tensorflow 2 installation documentation for your system](https://www.tensorflow.org/install/pip).

## Setup

1. Clone this repository
2. Using conda shell, navigate to the repository folder and create a new conda environment with `conda env create -f gpu.yml` OR `conda env create -f cpu.yml` depending on whether you want to use GPU or CPU.
3. Activate the environment with `conda activate tfgpu` OR `conda activate tfcpu`
4. Verify the conda environment with `conda env list` and `conda info`
5. Verify CPU support with `python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`
6. Verify GPU support with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
7. Download the DIV2K dataset [bicubic LR x3 testing](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip), [bicubic LR x3 validation](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip), [HR testing](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip), and [HR validation](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) .zip archives.
8. Extract the four DIV2K .zip archives into the `data` folder.

## Training

For preliminary Residual Learning training, run the following command:

```pwsh
python srmodel.py
```

Afterwards, run Quantization Aware Training by adding the `-q` flag to the previous command:

```pwsh
python srmodel.py -q
```

Finally, Quantize the model generating the mobile-ready .tflite file with the `-g` flag:

```pwsh
python srmodel.py -g
```

The above command also evaluates the model on the DIV2K validation set on your machine and prints the average PSNR score.

## Testing

Test the model on a modern Android mobile device by copying the generated .tflite file to the phone, installing [AI Benchmark](https://play.google.com/store/apps/details?id=org.benchmark.demo&pcampaignid=web_share), entering "PRO MODE", selecting "CUSTOM MODEL", and selecting the .tflite file generated in the previous step. Our results are as follows:

### AI Benchmark Options

- Input values range (min / max): 0, 255
- Inference Mode: INT8
- \# of CPU Threads: 4
- \# of Inferece Iterations: 100
- Delay between Inference Iterations, ms: 0

### Results (Snapdragon 888 on Asus ROG Phone 5)

- `srmodel.py -g` PSNR: 29.98
- AI Benchmark CPU Inference Time: 187ms
- AI Benchmark GPU (TFLite GPU Delegate) Inference Time: 53.6ms
- AI Benchmark Accelerator (Qualcomm QNN HTP) Inference Time: 10.5ms
