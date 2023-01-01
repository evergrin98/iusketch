from classes.video_clip import VideoClip

import numpy as np
import tensorflow as tf



class DataSetGenerator(tf.keras.utils.Sequence):
    '''
    DataSetGenerator는 tf.keras.utils.Sequence를 상속.
        def __len__(self):
        def __getitem__(self, index):
        on_epoch_end
    5d의 batch dataset을 만들어 주는 base 클래스.
    '''
    def __init__(self, imgs=[], imgw=64, imgh=64, time_step=20, batch_size=16, is_train=False, for_enc=False):
        '''
        dataset: 5d(bthwc) dataset
        batch_size: batch_size입니다.
        img_size: preprocess에 사용할 입력이미지의 크기입니다.
        is_train: 이 Generator가 학습용인지 테스트용인지 구분합니다.
        '''
        self.imgs = imgs
        self.img_w = imgw
        self.img_h = imgh
        self.time_step = time_step
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = True
        self.for_enc = for_enc
        self.data_count = len(self.imgs)

        # self.on_epoch_end()


    def __len__(self):
        '''
        Generator의 length
        계속해서 생성 가능하므로 임의의 값으로 설정.
        '''
        return 1000


    def __getitem__(self, index):
        ''' 입력데이터와 라벨을 생성. '''

        if self.shuffle is True:
            np.random.shuffle(self.imgs)

        clips = []
        for idx in range(self.batch_size):

            # gif파일을 VideoClip으로 변경.(norm되어 있음.)
            clip = VideoClip(gif_path=self.imgs[idx])

            # clip 사이즈(width, height) 설정.
            clip.resize(self.img_w, self.img_h, inplace=True)

            # raw clip에서 TIME_STEP 개수의 프레임만 가져온다.
            if self.for_enc:
                clip = clip.random_clips(count=self.time_step + 1)
            else:
                clip = clip.sequential_clips(count=self.time_step + 1, reverse=False)

            # 그려진 부분을 누적하여 frame을 생성.
            clip = clip.stacked_frames_clip(step=1)

            #TODO: augmentation.


            clips.append(clip.to_array())

        clips = np.stack(clips)

        if self.for_enc:
            return self.create_shifted_frames(clips, offset=0)
        else:
            return self.create_shifted_frames(clips, offset=1)


    def create_shifted_frames(self, clips, offset=1):
        ''' frame들로 x, y 데이터를 생성함. '''
        time_step = clips.shape[1]
        x = clips[:, 0 : time_step - offset, :, :]
        y = clips[:, offset : time_step, :, :]
        return x, y


    # def on_epoch_end(self):
    #     return self







if __name__ == "__main__":
    """ 
    main함수.
    """
    import os
    import glob

    IMG_PATH = '/home/evergrin/iu/datas/imgs/data_set/'

    img_list = glob.glob(os.path.join(IMG_PATH, "*.gif"))

    dgen = DataSetGenerator(imgs=img_list, batch_size=4, time_step=5, is_train=False)

    it = iter(dgen)

    for i in range(10):
        x, y = next(it)
        print(x.shape, y.shape)

    