import numpy as np
import random

from PIL import Image
import imageio as iio
# import imageio.v3 as iio    # gif
# from PIL.PngImagePlugin import PngImageFile

from classes.image_frame import ImgFrame




class VideoClip():
    '''
    ImgFrame(h,w,c)으로 VideoClip을 생성..
    w,h이 동일한 frame만 보관.
    0번이 처음이고 가장 아래 frame임.
    '''
    def __init__(self, frames=[], gif_path="", shape=None, max_clip=100):
        ''' 
        VideoClip 생성자.
        frames가 있으면 복사 생성함. 
        '''
        self.max_clip = max_clip
        self.shape = shape
        self.clips = list()

        if len(frames) > 0:
            self.load_frames(frames)
        elif len(gif_path) > 0:
            self.load_gif(gif_path)


    def load_frames(self, frames):
        ''' 비어 있을때만 ImgFrame어레이를 복사하여 생성. '''

        if not self.isEmpty():
            raise Exception("clip not empty")

        if not isinstance(frames[0], ImgFrame):
            raise Exception("not ImgFrame array!!")

        for frame in frames:
            self.append(frame)


    def load_gif(self, gif_file, grayscale=True):
        '''
        gif파일로 frame들을 만든다.
        '''
        gif = iio.v3.imread(gif_file, index=None)
        if gif.ndim != 4:
            raise Exception("gif dim not correct")
        if gif.shape[3] != 3:
            raise Exception("gif chn not correct")

        frame_count = gif.shape[0]

        for idx in range(frame_count):
            img = Image.fromarray(gif[idx])
            # img = img.resize((self.img_h, self.img_w))
            if grayscale:
                img = img.convert("L")

            imgfrm = ImgFrame(img)
            self.append(imgfrm)


    def to_array(self, expand=False):
        '''
        [frames, height, width, channel]의 4차원 array로 전달.
        expand : True이면 5차원으로 확장함.
        '''
        frames = [ img_frm.arry for img_frm in self.clips]
        arry_4d = np.array(frames)
        if expand:
            return np.expand_dims(arry_4d, axis=0)
        else:
            return arry_4d


    def count(self):
        return len(self.clips)


    def isEmpty(self):
        return 0 == self.count()


    def isFull(self):
        return self.max_clip == self.count()


    def reset(self):
        self.shape = None
        self.clips.clear()


    def __str__(self):
        str = f"clip:{self.count()}/{self.max_clip}, {self.shape}"
        return str


    def resize(self, w, h, inplace=True):
        '''
        clips의 모든 frame을 resize한다.
        inplace인경우는, 내부 clip을 resize한다. 
        '''
        if inplace:
            # 내부 변수를 수정하므로, shape는 별도로 갱신한다.
            for img_frm in self.clips:
                img_frm.imgResize(w, h, inplace=True)
            
            self.shape = self.clips[0].shape()
            return self
        else:
            vclip = VideoClip()
            for img_frm in self.clips:
                img_frm = ImgFrame(img_frm.imgResize(w, h, inplace=False))
                vclip.append(img_frm)
            return vclip


    def append(self, imgfrm):
        '''
        imgframe을 stack에 추가
        stack의 frame과 shape가 같아야만 추가됨.
        '''
        if not isinstance(imgfrm, ImgFrame):
            raise Exception("not ImgFrame!!")
            # imgfrm = ImgFrame(imgfrm)

        if self.isEmpty():
            if self.shape is None:
                self.shape = imgfrm.shape()
            elif self.shape !=  imgfrm.shape():
                raise Exception("shape not match")
        elif self.isFull():
            raise Exception("stack full")
        elif self.shape != imgfrm.shape():
            raise Exception("img frm shape not same!!", self.shape, imgfrm.shape())

        self.clips.append(imgfrm)


    def merged(self):
        '''
        stack에서 use인 imgframe을 merge하여 합쳐진 imgframe을 생성
        '''

        imgfrm = None
        for img in self.clips:
            if img.use is True:
                if imgfrm is None:
                    imgfrm = img
                else:
                    imgfrm = imgfrm.add(img)

        return imgfrm


    def make_gif(self, gif_file, reverse=False, ratio=1.0):
        '''
        frame들을 gif파일로 만든다.
        '''
        frames = list()
        for img_frm in self.clips:
            if img_frm.use is True:

                if ratio != 1.0:
                    w, h = img_frm.img_wh()
                    w = int(w * ratio)
                    h = int(h * ratio)
                    img_frm.imgResize(w, h)

                img = img_frm.to_image()
                frames.append(img)

        if reverse:
            frames.reverse()

        iio.mimsave(gif_file, frames, "GIF", fps=5)
        print(gif_file, " saved")


    def stacked_frame(self, sidx=0, eidx=-1):
        '''
        모든 imgframe을 channel방향으로 stack하여 imgframe을 생성
        '''
        if eidx == 0:
            eidx = len(self.clips)

        imgfrm = ImgFrame(self.clips[sidx])
        for img in self.clips[sidx:eidx]:
            if img.use is True:
                imgfrm.append_channel(img)

        return ImgFrame(imgfrm.merged())


    def random_clips(self, count=20, include_top=False):
        '''
        clips에서 랜덤하게 count만큼 ImgFrame을 뽑아서 clip을 생성함.
        include_top : 첫번째 img는 전체 이미지이므로, 포함할지를 선택함.
          - gif생성시 처음과 끝 프레임 모두 전체 이미지여서 2장을 빼야함.
        '''

        idx_high = len(self.clips)
        idx_low = 0
        img_cnt = idx_high

        if not include_top:
            # gif생성시 처음과 끝 프레임 모두 전체 이미지여서 2장을 빼야함;;
            idx_low = 1
            idx_high -= 1
            img_cnt -= 2

        pick_cnt = min(img_cnt, count)

        frames = random.choices(population=self.clips[idx_low:idx_high], k=pick_cnt)
        return VideoClip(frames=frames)


    def sequential_clips(self, start_idx=None, count=20, include_top=False, reverse=False):
        '''
        clips에서 start_idx부터 순서대로 count만큼 ImgFrame을 뽑아서 clip을 생성함.
        start_idx가 None이 아니면 start_idx사용, None이면 random값 사용.
        include_top : 첫번째 img는 전체 이미지이므로, 포함할지를 선택함.
          - gif생성시 처음과 끝 프레임 모두 전체 이미지여서 2장을 빼야함.
        '''

        idx_high = len(self.clips)
        idx_low = 0
        img_cnt = idx_high
        if not include_top:
            idx_low = 1
            idx_high -= 1
            img_cnt -= 2

        pick_cnt = min(img_cnt, count)

        # clip index를 두번 반복해서 roundQueue처럼 동작.
        idx_list = [ idx for idx in range(idx_low, idx_high)]
        idx_list.extend(idx_list)

        if start_idx is None:
            # random start idx
            start_idx = random.randrange(idx_low, idx_high)

        idx_list = idx_list[start_idx:start_idx + pick_cnt]

        if reverse:
            idx_list.reverse()

        frames = [ self.clips[idx] for idx in idx_list ]

        return VideoClip(frames=frames)


    def stacked_frames_clip(self, step=1):
        '''
        clips에서 그려진 부분을 누적하여 frame생성.
        step==1인경우 0, 0-1, 0-2, 0-3, ... 0-n까지 stack한 frame들로 clip을 생성.
        step==2인 경우, 0, 0-2, 0-4, 0-6... 0-n까지 stack한 frame들로 clip생성.
        지금은 grayscale된 1채널만 동작함.
        '''
        if self.isEmpty():
            raise Exception("empty!!")

        vclip = VideoClip()
        stacked_frame = None

        for idx, frame in enumerate(self.clips):

            if stacked_frame is None:
                stacked_frame = frame
            else:
                stacked_frame.append_channel(frame)

            img_frame = ImgFrame(stacked_frame.merged())

            if idx % step == 0:
                vclip.append(img_frame)

        return vclip





if __name__ == "__main__":
    """ 
    main함수.
    """

    import os
    import glob
    from utils.files import dir_path_change


    IMG_LOAD_BASE_PATH = '/home/evergrin/iu/datas/imgs/raw_gif'
    IMG_SAVE_BASE_PATH = '/home/evergrin/iu/datas/imgs/data_set'


    gif_list = glob.glob(os.path.join(IMG_LOAD_BASE_PATH, "*.gif"))
    gif_file = gif_list[0]

    vclip = VideoClip()
    vclip.load_gif(gif_file, grayscale=True)

    arrys = []
    arry1 = vclip.to_array()
    arry2 = vclip.to_array()
    print(arry1.shape, arry2.shape)
    
    arrys.append(arry1)
    arrys.append(arry2)
    
    arrys = np.stack(arrys)
    print(arrys.shape)

    # for i in range(1):
    #     vclip = VideoClip()
    #     vclip.load_gif(gif_file, grayscale=True)

    #     # newclip = vclip.random_clips()
    #     newclip = vclip.sequential_clips(count=10, reverse=True)
    #     varry = newclip.to_array()
    #     print(varry.shape)

    #     stacked_clip = newclip.stacked_frames_clip(step=2)
    #     resized_clip = stacked_clip.resize(64, 32, inplace=False)

    #     new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "gif")
    #     stacked_clip.make_gif(new_file)

    #     new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "gif")
    #     resized_clip.make_gif(new_file)

    #     stacked_clip.resize(12, 12, inplace=True)
    #     new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "gif")
    #     stacked_clip.make_gif(new_file)

        