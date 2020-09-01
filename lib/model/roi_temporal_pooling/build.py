from __future__ import print_function
import os
import torch
from torch.utils.ffi import create_extension


sources = ['src/roi_temporal_pooling.c']
headers = ['src/roi_temporal_pooling.h']

extra_objects = []
defines = []
with_cuda = False

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_temporal_pooling_cuda.c']
    headers += ['src/roi_temporal_pooling_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True
    extra_objects = ['src/roi_temporal_pooling_kernel.cu.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]  # roi_temporal_pooling_kernel.cu.o文件的绝对路径

ffi = create_extension(
    name='_ext.roi_temporal_pooling',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
