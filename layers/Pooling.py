import numpy as np
from numpy.lib.stride_tricks import as_strided

from Layers.Base import Base

class Pooling(Base):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = np.asarray(stride_shape)
        self.pooling_shape = np.asarray(pooling_shape)


    def forward(self, input_tensor):
        self.original_shape = input_tensor.shape
        pooled_tensor = []
        self.index_tensor = []

        for input_sample in input_tensor:
            pooled_sample = []
            max_indices = []
            output_shape = (np.asarray(input_sample[0].shape) - self.pooling_shape) // self.stride_shape + 1
            for input_slice in input_sample:
                strides = (*(np.asarray(input_slice.strides) * self.stride_shape), *input_slice.strides)
                shape = (*output_shape, *self.pooling_shape)

                strided_slice = as_strided(input_slice, shape=shape, strides=strides).reshape(-1, np.prod(self.pooling_shape))
                pooled_slice = strided_slice.max(axis=1).reshape(output_shape)
                pooled_sample.append(pooled_slice)

                indices = np.arange(np.prod(input_slice.shape)).reshape(input_slice.shape)
                strides = (*(np.asarray(indices.strides) * self.stride_shape), *indices.strides)
                strided_indices = as_strided(indices, shape=shape, strides=strides).reshape(strided_slice.shape)
                
                max_idx = strided_indices[np.arange(len(strided_slice)), np.argmax(strided_slice, axis=1)]
                max_idx = np.unravel_index(max_idx, indices.shape)
                max_indices.append(max_idx)
                
            pooled_tensor.append(pooled_sample)
            self.index_tensor.append(max_indices)
        
        return np.asarray(pooled_tensor)


    def backward(self, error_tensor):
        next_error_tensor = np.zeros(self.original_shape)
        for i, (error_sample, index_set) in enumerate(zip(error_tensor, self.index_tensor)):
            for j, (error_slice, slice_indices) in enumerate(zip(error_sample, index_set)):
                for k, (x, y) in enumerate(zip(*slice_indices)):
                    next_error_tensor[i][j][x, y] += error_slice.flatten()[k]

        return next_error_tensor