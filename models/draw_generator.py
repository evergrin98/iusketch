from classes.video_clip import VideoClip

import copy
import random
import cv2
import numpy as np
import tensorflow as tf

from classes.image_frame import ImgFrame

from albumentations import  ReplayCompose, Compose, HorizontalFlip, CropAndPad, SafeRotate, ShiftScaleRotate, Affine



class DrawGenerator(tf.keras.utils.Sequence):
    '''
    DataSetGenerator는 tf.keras.utils.Sequence를 상속.
        def __len__(self):
        def __getitem__(self, index):
        on_epoch_end
    5d의 batch dataset을 만들어 주는 base 클래스.
    '''
    def __init__(self, imgs=[], imgw=64, imgh=64, time_step=20, batch_size=16, stack_step=2,
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
        self.stack_step = stack_step
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
                                        percent=(0, 0.2), p=1.0, pad_mode=cv2.BORDER_CONSTANT, 
                                        pad_cval=1.0, keep_size=True)])) # crop and pad(size same)
        self.augmentation_list.append(ReplayCompose([SafeRotate(
                                        limit=[-15, 15], interpolation=1, border_mode=cv2.BORDER_CONSTANT, 
                                        value=1.0, mask_value=None, always_apply=False, p=1.0)])) # saferotate(size same)

        self.augmentation = ReplayCompose([
                                            HorizontalFlip(p=aug_prob), 
                                            CropAndPad( percent=(0.0, 0.1), p=aug_prob, #negative crop, positive pad
                                               pad_mode=cv2.BORDER_CONSTANT, pad_cval=1.0, keep_size=True),
                                            SafeRotate(limit=[-15, 15], interpolation=1, border_mode=cv2.BORDER_CONSTANT, 
                                                      value=1.0, mask_value=None, always_apply=False, p=aug_prob),
                                            ShiftScaleRotate(shift_limit=0.1, scale_limit=(0, 0.1), rotate_limit=0, p=aug_prob, 
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

        datas = []
        
        idx = index * self.batch_size
        while True:

            # gif파일을 VideoClip으로 변경.(norm되어 있음.)
            clip = VideoClip(gif_path=self.imgs[idx % self.data_count])

            idx += 1

            del clip.clips[0]
            del clip.clips[-1]

            # TODO: top frame check.
            clip = clip.augmentation(self.augmentation)

            if self.fill_box :
                clip.filled_frames_clip()

            frm_cnt = clip.count()

            # clip index를 두번 반복해서 roundQueue처럼 동작.
            idx_list = [ idx for idx in range(0, frm_cnt)]
            idx_list.extend(idx_list)

            # 그림 그려진 마지막 포인트를 랜덤하게 설정함
            last_pos_idx = random.randrange(0, frm_cnt)
            last_frm = clip.clips[last_pos_idx]
            
            img_pick_count = (self.time_step+1) * self.stack_step
            xy_idx_list = idx_list[last_pos_idx:last_pos_idx + img_pick_count]

            img_idx_list = [last_pos_idx]

            # 남아 있는 frame idx를 구함.
            rest_idx_list = [ idx for idx in range(0, frm_cnt)]

            for idx in xy_idx_list:
                if idx in rest_idx_list:
                    rest_idx_list.remove(idx)

            # 남아 있는 frame으로 random한 이미지를 stack하여 생성.
            # 랜덤한 프레임 패치를 여러번 가져와 붙인다.
            patch_count = random.randint(5, 30)
            for _ in range(patch_count):
                start_pos = random.randint(0, len(rest_idx_list))
                get_count = random.randint(0, len(rest_idx_list))
                end_pos = min(len(rest_idx_list), start_pos + get_count)
                patch_frames = rest_idx_list[start_pos:end_pos]
                img_idx_list.extend(patch_frames)

                for idx in rest_idx_list:
                    if idx in patch_frames:
                        rest_idx_list.remove(idx)
                if not rest_idx_list:
                    break

            img_idx_list = list(set(img_idx_list))
            img_frames = [ clip.clips[idx] for idx in img_idx_list ]
            img_clip = VideoClip(frames=img_frames)
            img_frm = img_clip.stacked_frame()

            xy_frames = [ clip.clips[idx] for idx in xy_idx_list ]

            stacked_xy_frames = []
            stacked_frame = None
            for idx, frame in enumerate(xy_frames):
        
                if stacked_frame is None:
                    stacked_frame = frame
                else:
                    stacked_frame.append_channel(frame)

                img_frame = ImgFrame(stacked_frame.merged(), do_norm=False)

                if (idx + 1) % self.stack_step == 0:
                    stacked_xy_frames.append(img_frame)
                    stacked_frame = None

            x_frames = copy.deepcopy(stacked_xy_frames[0:-1])
            y_frames = copy.deepcopy(stacked_xy_frames[1:])

            for xframe in x_frames:
                # stacked_img = ImgFrame(img_frm)
                # stacked_img.append_channel(xframe)
                # stacked_arry = stacked_img.merged()
                # xframe.append_channel(stacked_arry, do_norm=False)
                xframe.append_channel(img_frm, do_norm=False)

            datas.append([x_frames, y_frames])
            if self.batch_size == len(datas):
                break

        x_arry = []
        y_arry = []
        for xydata in datas:
            xdata = [ img_frm.arry for img_frm in xydata[0]]
            ydata = [ img_frm.arry for img_frm in xydata[1]]
            xdata = np.stack(xdata)
            ydata = np.stack(ydata)
            x_arry.append(xdata)
            y_arry.append(ydata)

        x_arry = np.stack(x_arry)
        y_arry = np.stack(y_arry)

        if self.invert:
            x_arry = 1 - x_arry
            y_arry = 1 - y_arry

        # xframe의 channel 0: 입력, channel 1: 그려진 이미지.
        return x_arry, y_arry


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

    # def arry5d_to_img(arry5d, save_as='', threshold=0.0):
    #     frmimg_cnt = arry5d.shape[1]
    #     fig, axes = plt.subplots(nrows = 1, ncols = frmimg_cnt, figsize=(15, 3))

    #     for idx, num in enumerate(range(0, frmimg_cnt)):
    #         frm = ImgFrame(img=arry5d[0][idx][:, :, :], do_norm=False)
    #         min_val = np.min(frm.arry)
    #         max_val = np.max(frm.arry)
    #         frm.arry = (frm.arry - min_val) / (max_val - min_val)

    #         min_val = np.min(frm.arry)
    #         max_val = np.max(frm.arry)
    #         # print("min,max: ", np.min(frm.arry), np.max(frm.arry))

    #         if threshold > 0.0:
    #             frm.threshold(threshold=threshold)

    #         img = frm.to_image()
    #         axes[idx].imshow(img, cmap='gray')

    #     plt.show()
    
    
    dgen = DrawGenerator(imgs=img_list, batch_size=4, time_step=5)

    it = iter(dgen)
    x, y = next(it)
    print("x:", x.shape)
    print("y:", y.shape)

    # arry5d_to_img(x, threshold=0.)
    
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

    