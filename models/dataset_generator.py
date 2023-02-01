from classes.video_clip import VideoClip

import random
import cv2
import numpy as np
import tensorflow as tf

from albumentations import  ReplayCompose, Compose, HorizontalFlip, CropAndPad, SafeRotate, ShiftScaleRotate, Affine



class DataSetGenerator(tf.keras.utils.Sequence):
    '''
    DataSetGenerator는 tf.keras.utils.Sequence를 상속.
        def __len__(self):
        def __getitem__(self, index):
        on_epoch_end
    5d의 batch dataset을 만들어 주는 base 클래스.
    '''
    def __init__(self, imgs=[], imgw=64, imgh=64, time_step=20, batch_size=16, 
                 seq_type='forward', label_type='1step', aug_prob=0.4, 
                 stacked=True, overlap=False, fill_box=False, invert=False):
        '''
        dataset: 5d(bthwc) dataset
        batch_size: batch_size입니다.
        img_size: preprocess에 사용할 입력이미지의 크기입니다.
        seq_type: random, forward, reverse
        label_type: 1step, all, same
        '''
        self.imgs = imgs
        self.img_w = imgw
        self.img_h = imgh
        self.time_step = time_step
        self.batch_size = batch_size
        self.shuffle = True
        self.seq_type = seq_type
        self.label_type = label_type
        self.stacked = stacked
        self.overlap = overlap
        self.fill_box = fill_box
        self.invert = invert
        self.data_count = len(self.imgs)

        self.augmentation_list = [None, ]
        self.augmentation_list.append(ReplayCompose([HorizontalFlip(p=1.0)])) # 좌우대칭
        self.augmentation_list.append(ReplayCompose([CropAndPad(
                                        percent=(-0.2, 0.2), p=1.0, pad_mode=cv2.BORDER_CONSTANT, 
                                        pad_cval=1.0, keep_size=True)])) # crop and pad(size same)
        self.augmentation_list.append(ReplayCompose([SafeRotate(
                                        limit=[-45, 45], interpolation=1, border_mode=cv2.BORDER_CONSTANT, 
                                        value=1.0, mask_value=None, always_apply=False, p=1.0)])) # saferotate(size same)

        self.augmentation = ReplayCompose([
                                            HorizontalFlip(p=aug_prob), 
                                            CropAndPad( percent=(-0.2, 0.2), p=aug_prob,
                                               pad_mode=cv2.BORDER_CONSTANT, pad_cval=1.0, keep_size=True),
                                            SafeRotate(limit=[-45, 45], interpolation=1, border_mode=cv2.BORDER_CONSTANT, 
                                                      value=1.0, mask_value=None, always_apply=False, p=aug_prob),
                                            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=aug_prob, 
                                                             value=1.0, border_mode=cv2.BORDER_CONSTANT),
                                            # Affine(translate_percent=0.2, p=aug_prob, cval=1.0, mode=cv2.BORDER_CONSTANT),
                                            Affine(translate_percent=None, shear=15, p=aug_prob, cval=1.0, mode=cv2.BORDER_CONSTANT),
                                           ])
        self.on_epoch_end()


    def __len__(self):
        '''
        Generator의 length
        계속해서 생성 가능하므로 임의의 값으로 설정.
        '''
        return self.data_count // self.batch_size


    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.imgs)


    def __getitem__(self, index):
        ''' 입력데이터와 라벨을 생성. '''

        clips = []
        
        idx = index * self.batch_size
        while True:

            # gif파일을 VideoClip으로 변경.(norm되어 있음.)
            clip = VideoClip(gif_path=self.imgs[idx % self.data_count])
            
            idx += 1

            # 프레임 개수에 따라 leap_step을 랜덤 선택하고 프레임을 골라서 누적한다.
            frm_cnt = clip.count() - 2
            max_leap_step = max(1, frm_cnt // (self.time_step + 1))

            leap_step = random.randint(1, max_leap_step)
            if self.seq_type == 'all' or self.seq_type == 'arandom' \
                or self.seq_type == 'aforward' or self.seq_type == 'rest':
                leap_step = 1

            pick_count = leap_step * (self.time_step + 1)


            # clip 사이즈(width, height) 설정.
            clip.resize(self.img_w, self.img_h, inplace=True)

            # raw clip에서 TIME_STEP 개수의 프레임만 가져온다.
            use_top_frame = False
            if self.label_type == 'all':
                use_top_frame = True

            if self.fill_box :
                clip.filled_frames_clip()

            try:
                if self.seq_type == 'all':
                    clip = clip.all_clips(count=pick_count, include_top=use_top_frame, shuffle='none', overlap=self.overlap)
                elif self.seq_type == 'rest':
                    clip = clip.all_clips(count=pick_count, include_top=True, shuffle='random', overlap=self.overlap)
                elif self.seq_type == 'arandom':
                    clip = clip.all_clips(count=pick_count, include_top=use_top_frame, shuffle='random', overlap=self.overlap)
                elif self.seq_type == 'aforward':
                    clip = clip.all_clips(count=pick_count, include_top=use_top_frame, shuffle='forward', overlap=self.overlap)
                elif self.seq_type == 'random':
                    clip = clip.random_clips(count=pick_count, step=leap_step, include_top=use_top_frame)
                elif self.seq_type == 'reverse':
                    clip = clip.sequential_clips(count=pick_count, step=leap_step, reverse=True, include_top=use_top_frame)
                else: # 'forward'
                    clip = clip.sequential_clips(count=pick_count, step=leap_step, reverse=False, include_top=use_top_frame)

                # 그려진 부분을 누적하여 frame을 생성.
                if self.stacked :
                    clip = clip.stacked_frames_clip(step=1, included_label=use_top_frame)
            except:
                continue

            # 랜덤한 augment를 clip의 모든 frame에 동일하게 적용.
            # augment = random.choices(population=self.augmentation_list, k=1)[0]
            clip = clip.augmentation(self.augmentation)

            # clip은 grayscale이고, augmentation후 이미지가 흐려질 수 있으므로 threshold 0.7정도는 되어야 함.
            # clip.threshold(threshold=0.7, low=0.0, high=1.0, inplace=True)
            clips.append(clip)
            
            if self.batch_size == len(clips):
                break

        clips_arry = []
        for clip in clips:
            clips_arry.append(clip.to_array())

        clips_arry = np.stack(clips_arry)
        
        if self.invert:
            clips_arry = 1 - clips_arry

        if self.label_type == 'all':
            # x, y는 x의 마지막프레임인 전체 이미지를 라벨로 사용.
            x = clips_arry[:, : -1, :, :, :]
            y = list()
            for idx in range(self.batch_size):
                yy = [ clips_arry[idx][-1] for i in range(x.shape[1]) ]
                y.append(np.stack(yy))

            y = np.stack(y)

        elif self.label_type == 'same':
            # x, y가 동일한 이미지 사용.
            x, y = self.create_shifted_frames(clips_arry, offset=0)
        
        elif self.label_type == 'rest':
            # x는 random.,y 는 전체이미지에서 미완성 부분
            x, y = self.create_rest_frames(clips)
            
            return x, y

        else: # '1step'
            # x, y가 한 step씩 달라진 이미지 사용.
            x, y = self.create_shifted_frames(clips_arry, offset=1)

        # TODO : 라벨은 그대로 두고 입력만 augmentation하는 경우... 

        return x, y



    def create_rest_frames(self, clips):
        ''' label에서 x를 제외한 부분을 y로 생성함.. '''

        xes = []
        yes = []

        for clip in clips:
            label = clip.clips[-1]
            y_clip = clip.minus_from(label, stack=True, included_label=True)
            vclip = VideoClip(frames=clip.clips[0:-1])
            xes.append(vclip.to_array())
            yes.append(y_clip.to_array())

        return np.stack(xes), np.stack(yes)


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
    from classes.image_frame import ImgFrame

    IMG_PATH = '/home/evergrin/iu/datas/data_set'

    img_list = glob.glob(os.path.join(IMG_PATH, "*.gif"))

    def arry5d_to_img(arry5d, save_as='', threshold=0.0):
        frmimg_cnt = arry5d.shape[1]
        fig, axes = plt.subplots(nrows = 1, ncols = frmimg_cnt, figsize=(15, 3))

        for idx, num in enumerate(range(0, frmimg_cnt)):
            frm = ImgFrame(img=arry5d[0][idx][:, :, :], do_norm=False)
            min_val = np.min(frm.arry)
            max_val = np.max(frm.arry)
            frm.arry = (frm.arry - min_val) / (max_val - min_val)

            min_val = np.min(frm.arry)
            max_val = np.max(frm.arry)
            # print("min,max: ", np.min(frm.arry), np.max(frm.arry))

            if threshold > 0.0:
                frm.threshold(threshold=threshold)

            img = frm.to_image()
            axes[idx].imshow(img, cmap='gray')

        plt.show()
    
    
    dgen = DataSetGenerator(imgs=img_list, batch_size=4, time_step=5, seq_type='aforward', label_type='1step', stacked=True, overlap=False)

    it = iter(dgen)
    x, y = next(it)

    arry5d_to_img(x, threshold=0.)
    
    print('end')
    
    # frm = ImgFrame(img=x[0][-1][:, :, :], do_norm=False)
    # img = frm.to_image()
    # plt.imshow(img, cmap='gray')

    # frm = ImgFrame(img=y[0][-1][:, :, :], do_norm=False)
    # img = frm.to_image()
    # plt.imshow(img, cmap='gray')
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

    