from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

"""
change AT_CHECK to TORCH_CHECK
https://discuss.pytorch.org/t/my-pytorch-model-stop-logging-without-error/77686/7
"""

setup(
    name='masked_conv',
    ext_modules=[
        CUDAExtension('deform_conv_cuda', [
            'src/deform_conv_cuda.cpp',
            'src/deform_conv_cuda_kernel.cu',
        ],
        define_macros=[('WITH_CUDA', None)],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
        ]})],
        cmdclass={'build_ext': BuildExtension})

