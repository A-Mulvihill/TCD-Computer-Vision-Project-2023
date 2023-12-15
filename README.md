# TCD-Computer-Vision-Project-2023

This project seeks to implement the Anchor-based Plain Net for Mobile Image Super-Resolution [Du Zongcai, et al. 2021] in Tensorflow 2.10.1 and python 3 on Windows 11 64-bit.  
**Please note** that as of the time of writing, the latest Python version is 3.11, the latest Windows OS is 11, and while the latest Tensorflow version is 2.15, versions 2.11 onwards no longer support GPU acceleration on Windows Native. You may use later versions of Python and Tensorflow on Windows with WSL2, or on other operating systems. The provided conda environment files assume native Windows, thus there are separate environments for GPU and CPU support.

## Requirements

### Microsoft Windows

- Windows 7 or higher (64-bit)
- Modern AVX supported CPU (see [CPU support](https://www.tensorflow.org/install/source_windows#cpu_support))
- [Win32 Long Paths enabled](https://superuser.com/a/1119980)
- [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, 2019, and 2022](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022)
- Python 3.9-3.11
- Pip 19.0 or higher
- [Miniconda for Windows](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
- For GPU support: Nvidia GPU with CUDA 3.5 or higher and Nvidia GPU drivers 450.80.02 or higher

### Other Operating Systems

See the [Tensorflow 2 installation documenttation for your system](https://www.tensorflow.org/install/pip).

## Setup

1. Clone this repository
2. Create a new conda environment with `conda create --name tfgpu --f gpu.yml` OR `conda create --name tfcpu --f cpu.yml` depending on whether you want to use GPU or CPU.
3. Activate the environment with `conda activate <env>`
4. Verify the conda environment with `conda env list` and `conda info`
5. Verify CPU support with `python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`
6. Verify GPU support with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## Execution
