import numpy as np
from scipy import ndimage, misc


x8, y8 = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8))
filt8 = x8**2+y8**2 < 1.1


class RandomCrop3D(object):
    """Crop randomly the image in a sample.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size
    def __call__(self, data):
        h = data.shape[0]
        x = np.random.randint(0, h - self.output_size, size=3)

        return data[x[0]:(x[0] + self.output_size),
               x[1]:(x[1] + self.output_size),
               x[2]:(x[2] + self.output_size)]


class RandomCrop2D(object):
    """Crop randomly the image in a sample.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size
    def __call__(self, data):
        h = data.shape[0]
        x = np.random.randint(0, h - self.output_size, size=2)

        return data[x[0]:(x[0] + self.output_size),
               x[1]:(x[1] + self.output_size)]


class RandomAddNoise(object):
    """Crop randomly the image in a sample.
    """
    def __init__(self, noise_amplitude):
        assert isinstance(noise_amplitude, float)
        if isinstance(noise_amplitude, float):
            self.noise_amplitude = noise_amplitude
    def __call__(self, data):
        return (data + np.random.normal(0, self.noise_amplitude, size=data.shape)).astype(np.float32)


class RandomLogNorm(object):
    """Fix densities"""
    def __init__(self, output_size=0):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size
    def __call__(self, data):
        data -= data.mean()
        data /= 160
        #         print(data.max(), data.min())
        return data


class ReadNP(object):
    """Fix densities"""
    def __init__(self, output_size=0):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, data):
        return data['arr_0']


class Rotate(object):
    """Fix densities"""
    def __init__(self, output_size=0):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, data):
        return data['arr_0']


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 8, 8)
    return x