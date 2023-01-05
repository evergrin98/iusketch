from classes.video_clip import VideoClip

import random
import cv2
import numpy as np
import tensorflow as tf

from albumentations import  Compose, HorizontalFlip, CropAndPad



class DataSetGenerator(tf.keras.utils.Sequence):
    '''
    DataSetGenerator는 tf.keras.utils.Sequence를 상속.
        def __len__(self):
        def __getitem__(self, index):
        on_epoch_end
    5d의 batch dataset을 만들어 주는 base 클래스.
    '''
    def __init__(self, imgs=[], imgw=64, imgh=64, time_step=20, batch_size=16, is_train=False, for_enc=False, for_label=False):
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
        self.for_label = for_label
        self.data_count = len(self.imgs)

        self.augmentation_list = [None, ]
        self.augmentation_list.append(Compose([HorizontalFlip(p=1.0)])) # 좌우대칭
        self.augmentation_list.append(Compose([CropAndPad(
            percent=(-0.2, 0.2), p=1.0, pad_mode=cv2.BORDER_CONSTANT, pad_cval=1.0, keep_size=True)])) # crop and pad(size same)

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

            # 프레임 개수에 따라 leap_step을 랜덤 선택하고 프레임을 골라서 누적한다.
            frm_cnt = clip.count() - 2
            max_leap_step = frm_cnt // (self.time_step + 1)
            leap_step = random.randint(1, max_leap_step)
            pick_count = leap_step * (self.time_step + 1)

            # clip 사이즈(width, height) 설정.
            clip.resize(self.img_w, self.img_h, inplace=True)

            # raw clip에서 TIME_STEP 개수의 프레임만 가져온다.
            if self.for_enc:
                clip = clip.random_clips(count=pick_count, include_top=self.for_label)
            else:
                clip = clip.sequential_clips(count=pick_count, reverse=False, include_top=self.for_label)

            # 그려진 부분을 누적하여 frame을 생성.
            clip = clip.stacked_frames_clip(step=leap_step, included_label=self.for_label)

            # 랜덤하게 augmentation 실행.
            augment = random.choices(population=self.augmentation_list, k=1)[0]
            clip = clip.augmentation(augment)

            # clip은 grayscale이고, augmentation후 이미지가 흐려질 수 있으므로 threshold 0.7정도는 되어야 함.
            clip.threshold(threshold=0.7, low=0.0, high=1.0, inplace=True)
            clips.append(clip.to_array())

        clips = np.stack(clips)

        if self.for_label:
            # x, y는 x의 마지막프레임인 전체 이미지를 라벨로 사용.
            x = clips[:, : -1, :, :, :]
            y = list()
            for idx in range(self.batch_size):
                yy = [ clips[idx][-1] for i in range(x.shape[1]) ]
                y.append(np.stack(yy))

            y = np.stack(y)

            return x, y

        elif self.for_enc:
            # x, y가 동일한 이미지 사용.
            return self.create_shifted_frames(clips, offset=0)

        else:
            # x, y가 한 step씩 달라진 이미지 사용.
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
    import matplotlib.pyplot as plt

    IMG_PATH = '/home/evergrin/iu/datas/data_set'

    img_list = glob.glob(os.path.join(IMG_PATH, "*.gif"))

    dgen = DataSetGenerator(imgs=img_list, batch_size=4, time_step=5, for_enc=False, for_label=True)

    it = iter(dgen)
    x, y = next(it)

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

    