from tensorflow import keras
from tensorflow.keras import layers
from models.layer_conv import Conv2Plus1D




class ConvLstmSeries(keras.layers.Layer):
    '''
        5-rank Colvolution LSTM
    '''
    def __init__(self, filter_cnt, final_filter_cnt, kernel_sizes):
        """
            (b, t, h, w, c) -> (b, t, h/stride, w/stride, f)
            A sequence of convolutional layers that first apply the convolution operation over the
            spatial dimensions, and then the temporal dimension.
        """
        super().__init__()
        self.seq = keras.Sequential()

        for kernel_size in kernel_sizes:
            self.seq.add(
                layers.ConvLSTM2D(
                            filters=filter_cnt,
                            kernel_size=kernel_size,
                            padding="same",
                            return_sequences=True,
                            activation="relu",
                        )
                )
            self.seq.add(layers.BatchNormalization())
            # if use_bn is True:
            #     # self.seq.add(layers.BatchNormalization())
            # self.seq.add(layers.LayerNormalization())
            # self.seq.add(layers.ReLU())

        # 출력의 channel depth를 맞춰주기 위해.
        if final_filter_cnt > 0:
            self.seq.add(Conv2Plus1D(final_filter_cnt, (1, 3, 3), 1, "same"))

    def call(self, x):
        return self.seq(x)

