import copy
from PIL import Image
# from PIL.PngImagePlugin import PngImageFile
import numpy as np
import imageio
import imageio.v3 as iio    # gif

from classes.image_frame import ImgFrame




class VideoClip():
    '''
    ImgFrame(h,w,c)으로 VideoClip을 생성..
    w,h이 동일한 frame만 보관.
    0번이 처음이고 가장 아래 frame임.
    '''
    def __init__(self, shape=None, max_clip=100):
        self.max_clip = max_clip
        self.shape = shape
        self.clips = list()


    def load_gif(self, gif_file, grayscale=True):
        '''
        gif파일로 frame들을 만든다.
        '''
        gif = iio.imread(gif_file, index=None)
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
            raise Exception("img frm shape not same!!")

        self.clips.append(imgfrm)
        
        print(self)


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
        
        imageio.mimsave(gif_file, frames, "GIF", fps=5)
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
