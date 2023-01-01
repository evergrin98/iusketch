from tensorflow import keras
from tensorflow.keras import layers
import einops

class Conv2Plus1D(keras.layers.Layer):
    '''
        5-rank Spatial Convolution
    '''
    def __init__(self, filters, kernel_size, stride, padding):
        """
            (b, t, h, w, c) -> (b, t, h/stride, w/stride, f)
            A sequence of convolutional layers that first apply the convolution operation over the
            spatial dimensions, and then the temporal dimension. 
        """
        super().__init__()
        self.seq = keras.Sequential([
            # Spatial decomposition
            layers.Conv3D(filters=filters,
                        kernel_size=(1, kernel_size[1], kernel_size[2]),
                        strides=(1, stride, stride),
                        padding=padding),
            # Temporal decomposition
            # layers.Conv3D(filters=filters,
            #             kernel_size=(kernel_size[0], 1, 1),
            #             strides=1,
            #             padding=padding)
            ])

        # self.conv1 = layers.Conv3D(filters=filters,
        #             kernel_size=(1, kernel_size[1], kernel_size[2]),
        #             strides=(1, stride, stride),
        #             padding=padding)
        # # Temporal decomposition
        # self.conv2 = layers.Conv3D(filters=filters,
        #             kernel_size=(kernel_size[0], 1, 1),
        #             strides=1,
        #             padding=padding)

    def call(self, x):
        # x1 = self.conv1(x)
        # x2 = self.conv2(x1)
        # print("x shape", x.shape, x1.shape, x2.shape)
        # return x2
        return self.seq(x)


class TConv2Plus1D(keras.layers.Layer):
    '''
        5-rank Spatial TransposeConvolution
    '''
    def __init__(self, filters, kernel_size, stride, padding):
        """
            (b, t, h, w, c) -> (b, t, h*stride, w*stride, f)
            A sequence of convolutional layers that first apply the convolution operation over the
            spatial dimensions, and then the temporal dimension. 
        """
        super().__init__()
        self.seq = keras.Sequential([
            # Spatial decomposition
            layers.Conv3DTranspose(filters=filters,
                        kernel_size=(1, kernel_size[1], kernel_size[2]),
                        strides=(1, stride, stride),
                        padding=padding),
            # Temporal decomposition
            # layers.Conv3DTranspose(filters=filters, 
            #             kernel_size=(kernel_size[0], 1, 1),
            #             strides=1,
            #             padding=padding)
            ])

    def call(self, x):
        return self.seq(x)


class ResizeVideo(keras.layers.Layer):
    """
    Use the einops library to resize the tensor.
    """
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        """
        Args:
            video: Tensor representation of the video, in the form of a set of frames.
        Return:
            A downsampled size of the video according to the new height and width it should be resized to.
        """
        # b stands for batch size, t stands for time, h stands for height, 
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t = old_shape['t'])
        return videos