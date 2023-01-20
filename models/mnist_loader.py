from classes.video_clip import VideoClip

import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras



class MnistLoader(tf.keras.utils.Sequence):
    '''
    MnistLoader tf.keras.utils.Sequence를 상속.
        def __len__(self):
        def __getitem__(self, index):
        on_epoch_end
    5d의 batch dataset을 만들어 주는 base 클래스.
    '''

    # class variable
    data_sets = None
    indexes = None
    train_index = None
    val_index = None

    @staticmethod
    def load_dataset(max_count=1000, step=8, img_w=64, img_h=64):

        if MnistLoader.data_sets is None:
            fpath = keras.utils.get_file(
                "moving_mnist.npy",
                "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
            )
            MnistLoader.data_sets = np.load(fpath)
            # dataset: 5d(bthwc) dataset
            MnistLoader.data_sets = np.swapaxes(MnistLoader.data_sets, 0, 1)
            MnistLoader.data_sets = np.expand_dims(MnistLoader.data_sets, axis=-1)
            MnistLoader.data_sets = MnistLoader.data_sets[:max_count, : step+1, ...]
            
            if img_w != 64 or img_h != 64:
                clips = []
                for datas in MnistLoader.data_sets:
                    clip = VideoClip()
                    clip.from_array(datas, do_norm=False)
                    clip.resize(img_w, img_h, inplace=True)
                    clips.append(clip.to_array())
                    
                MnistLoader.data_sets = np.stack(clips)

            MnistLoader.indexes = np.arange(MnistLoader.data_sets.shape[0])
            np.random.shuffle(MnistLoader.indexes)
            MnistLoader.train_index = MnistLoader.indexes[: int(0.9 * MnistLoader.data_sets.shape[0])]
            MnistLoader.val_index = MnistLoader.indexes[int(0.9 * MnistLoader.data_sets.shape[0]) :]


    def reset_dataset():
        del MnistLoader.data_sets
        MnistLoader.data_sets = None


    def create_shifted_frames(data):
        x = data[:, 0 : data.shape[1] - 1, :, :]
        y = data[:, 1 : data.shape[1], :, :]
        return x, y


    def __init__(self, is_train=True, imgw=64, imgh=64, time_step=20, batch_size=16, max_count=1000):
        '''
        dataset: 5d(bthwc) dataset
        batch_size: batch_size입니다.
        img_size: preprocess에 사용할 입력이미지의 크기입니다.
        '''
        self.load_dataset(max_count=max_count, step=time_step, img_w=imgw, img_h=imgh)

        self.is_train = is_train

        if self.is_train:
            self.dataset = MnistLoader.data_sets[MnistLoader.train_index] / 255
        else:
            self.dataset = MnistLoader.data_sets[MnistLoader.val_index] / 255

        self.data_count = self.dataset.shape[0]

        self.img_w = imgw
        self.img_h = imgh
        self.time_step = time_step
        self.batch_size = batch_size

        # self.on_epoch_end()


    def __len__(self):
        '''
        Generator의 length
        계속해서 생성 가능하므로 임의의 값으로 설정.
        '''
        return self.data_count // self.batch_size


    def __getitem__(self, index):
        ''' 입력데이터와 라벨을 생성. '''

        data = self.dataset[index*self.batch_size: (index+1)*self.batch_size]

        x, y = MnistLoader.create_shifted_frames(data)

        return x, y




if __name__ == "__main__":
    """ 
    main함수.
    """
    import os
    import glob
    import matplotlib.pyplot as plt
    from classes.image_frame import ImgFrame

    IMG_PATH = '/home/evergrin/iu/datas/data_set'

    img_list = glob.glob(os.path.join(IMG_PATH, "*.gif"))

    dgen = MnistLoader(is_train=True, imgw=32, imgh=32, 
            time_step=5, batch_size=2, max_count=1000)

    it = iter(dgen)
    x, y = next(it)

    frm = ImgFrame(img=x[0][-1][:, :, :], do_norm=False)
    img = frm.to_image()
    plt.imshow(img, cmap='gray')

    frm = ImgFrame(img=y[0][-1][:, :, :], do_norm=False)
    img = frm.to_image()
    plt.imshow(img, cmap='gray')
    # label = x[:, -1, :, :]
    # print(label.shape)
    # label2 = np.expand_dims(label, axis=1)
    # print(label2.shape)
    
    # frm = ImgFrame(img=y[0][-1][:, :, :], do_norm=False)
    # img = frm.to_image()
    # plt.imshow(img, cmap='gray')
    # print(x.shape, y.shape)

    # for i in range(10):
    #     x, y = next(it)
        
    #     frm = ImgFrame(img=x[0][-1][:, :, :], do_norm=False)
    #     print(x.shape, y.shape)

    