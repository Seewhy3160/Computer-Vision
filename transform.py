import numpy as np
from skimage import io
import os.path as osp

def load_image(file_name):
    """
    Load image from disk
    :param file_name:
    :return: image: numpy.ndarray
    """
    if not osp.exists(file_name):
        print('{} not exist'.format(file_name))
        return
    image = io.imread(file_name)
    return np.array(image)

def save_image(image, file_name):
    """
    Save image to disk
    :param image: numpy.ndarray
    :param file_name:
    :return:
    """
    io.imsave(file_name,image)

def cs4243_resize(image, new_width, new_height):
    """
    10 points
    Implement the algorithm of nearest neighbor interpolation for image resize,
    :param image: ndarray
    :param new_width: int
    :param new_height: int
    :return: new_image: numpy.ndarray
    """
    new_image = np.zeros((new_height, new_width, 3), dtype='uint8')
    if len(image.shape)==2:
        new_image = np.zeros((new_height, new_width), dtype='uint8')
    ###Your code here####

    ###
    return new_image

def cs4243_rgb2grey(image):
    """
    5 points
    Implement the rgb2grey function
    weights for different channel: (R,G,B)=(0.299, 0.587, 0.114)
    Please scale the value to [0,1] by dividing 255
    :param image: numpy.ndarray
    :return: grey_image: numpy.ndarray
    """
    if len(image.shape) != 3:
        print('Image should have 3 channels')
        return
    ###Your code here####

    ###

    return image/255.

def cs4243_rotate180(kernel):
    """
     5 points
    Rotate the matrix by 180
    :param kernel:
    :return:
    """
    ###Your code here####

    ###
    return kernel

def cs4243_guassian_kernel(ksize, sigma):
    """
     10 points
    Implement the simplified Guassian kernel below:
    k(x,y)=exp((x^2+y^2)/(-2sigma^2))
	Note that Guassian kernel should be central symmentry.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    """
    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    ###Your code here####

    ###

    return kernel / kernel.sum()

def cs4243_filter(image, kernel):
    """
    15 points
    Implement the convolution operation in a naive 4 nested for-loops,
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return:
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####

    ###

    return filtered_image

def pad_zeros(image, pad_height, pad_width):
    """
    Pad the image with zero pixels, e.g., given matrix [[1]] with pad_height=1 and pad_width=2, obtains:
    [[0 0 0 0 0]
    [0 0 1 0 0]
    [0 0 0 0 0]]
    :param image: numpy.ndarray
    :param pad_height: int
    :param pad_width: int
    :return padded_image: numpy.ndarray
    """
    height, width = image.shape
    new_height, new_width = height+pad_height*2, width+pad_width*2
    padded_image = np.zeros((new_height, new_width))
    padded_image[pad_height:new_height-pad_height, pad_width:new_width-pad_width] = image
    return padded_image

def cs4243_filter_fast(image, kernel):
    """
    20 points
    Implement a fast version of filtering algorithm.
    Do element-wise multiplication between
    the kernel and a image region.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####
    #You may find the functions pad_zeros() and cs4243_rotate180() useful
    ###

    return filtered_image

def cs4243_filter_faster(image, kernel):
    """
    25 points
    Implement a faster version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and arrange them into a matrix of shape (Hi*Wi, Hk*Wk),also arrage the flipped
    kernel to be of shape (Hk*Hk, 1), then do matrix multiplication
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####
    #You may find the functions pad_zeros() and cs4243_rotate180() useful
    ###

    return filtered_image

def cs4243_downsample(image, ratio):
    """
    10 points
    Downsample the image to its 1/(ratio^2),which means downsample the width to 1/ratio, and the height 1/ratio.
    for example:
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = downsample(A, 2)
    B=[[1, 3], [7, 9]]
    :param image:numpy.ndarray
    :param ratio:int
    :return:
    """

    ###Your code here####

    ###
    return downsample_image