from tensorflow import keras
from tensorflow.keras import layers
from models.layer_conv import Conv2Plus1D, TConv2Plus1D



class Encoder5D(keras.layers.Layer):
    '''
        5-rank Encoder
    '''
    def __init__(self, kernel_count, filters, kernel_size, stride, padding):
        """
            conv : (b, t, h, w, c) -> (b, t, h/stride, w/stride, f)
            kernel_count만큼 반복.
        """
        super().__init__()
        self.seq = keras.Sequential()
        
        for cnt in range(kernel_count):
            self.seq.add( Conv2Plus1D(filters*2**cnt, kernel_size, stride, padding))
            self.seq.add( layers.BatchNormalization())
            self.seq.add( layers.ReLU())
            # self.seq.add( keras.activations.sigmoid)


    def call(self, x):
        return self.seq(x)
    
class Decoder5D(keras.layers.Layer):
    '''
        5-rank Decoder
    '''
    def __init__(self, kernel_count, filters, out_channel, kernel_size, stride, padding):
        """
            (b, t, h, w, c) -> (b, t, h*stride, w*stride, f)
            kernel_count만큼 반복.
        """
        super().__init__()
        self.seq = keras.Sequential()

        for cnt in range(kernel_count-1, -1, -1):
            self.seq.add( TConv2Plus1D(filters*2**cnt, kernel_size, stride, padding))
            self.seq.add( layers.BatchNormalization())
            self.seq.add( layers.ReLU())

        self.seq.add(Conv2Plus1D(out_channel, kernel_size, 1, padding="same"))
        self.seq.add( layers.BatchNormalization())
        self.seq.add( layers.ReLU())

    def call(self, x):
        return self.seq(x)
    
