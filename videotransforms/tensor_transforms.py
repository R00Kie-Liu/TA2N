import random

from videotransforms.utils import functional as F


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation

    Given mean: m and std: s
    will  normalize each channel as channel = (channel - mean) / std

    Args:
        mean (int): mean value
        std (int): std value
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of stacked images or image
            of size (C, H, W) to be normalized

        Returns:
            Tensor: Normalized stack of image of image
        """
        breakpoint()
        # if len(tensor.shape) == 4: # T, C, W, H
        #     for i in tensor:
        #         tensor[i:i+1] = F.normalize(tensor[i:i+1], self.mean, self.std)
        #     return tensor
        # else:
        return F.normalize(tensor, self.mean, self.std)

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, annos=None):
        '''
        Input: T, C, W, H
        return: T, C, W, H
        '''
        clip = False
        if len(tensor.shape) == 4:
            clip = True
            T, C, W, H = tensor.size()
            tensor = tensor.view(-1, W, H)

        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        if clip:
            return tensor.view(T, C, W, H)
        else:
            return tensor

class SpatialRandomCrop(object):
    """Crops a random spatial crop in a spatio-temporal
    numpy or tensor input [Channel, Time, Height, Width]
    """

    def __init__(self, size):
        """
        Args:
            size (tuple): in format (height, width)
        """
        self.size = size

    def __call__(self, tensor):
        h, w = self.size
        _, _, tensor_h, tensor_w = tensor.shape

        if w > tensor_w or h > tensor_h:
            error_msg = (
                'Initial tensor spatial size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial tensor is ({t_w}, {t_h})'.format(
                    t_w=tensor_w, t_h=tensor_h, w=w, h=h))
            raise ValueError(error_msg)
        x1 = random.randint(0, tensor_w - w)
        y1 = random.randint(0, tensor_h - h)
        cropped = tensor[:, :, y1:y1 + h, x1:x1 + h]
        return cropped
