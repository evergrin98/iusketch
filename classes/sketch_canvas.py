import tkinter as tk
from tkinter import filedialog, ttk

import apps.config_env as cfg

from PIL import Image

import math, os
import numpy as np

from utils.files import not_duplicated_path
from classes.canvas_zoombox import CanvasZoomBox
from classes.image_frame import ImgFrame
from classes.video_clip import VideoClip
from classes.canvas_drawer import CanvasDrawer
from classes.canvas_image import CanvasImage



class SketchCanvas():
    '''
    Sketch용 Canvas Widget
     - drawer : 마우스로 이미지 그리기.
     - crop box : 크기 변경 가능한 crop영역
     - frame, clip생성.
    '''
    def __init__(self, canvas, bg_color='white', pen_color='black'):
        self.canvas = canvas
        self.canvas.configure(bg=bg_color)

        # image clips
        self.img_clips = VideoClip()

        # canvas mouse draw
        self.drawer = CanvasDrawer(canvas, pen_color)

        # canvas image
        self.imager = CanvasImage(canvas, blurry=False)

        # crop box
        self.crop_box = CanvasZoomBox(canvas, w=256, h=256, max_x=512, max_y=512)



    def add_image_frame(self, crop=False):
        '''
        img_clips에 현재 보이는 이미지를 추가함.
        '''
        img_frame = self.to_image_frame(crop=crop)
        self.img_clips.append(img_frame)


    def clip_to_gif(self, save_path, reverse=True):
        '''
        img_clips에 현재 보이는 이미지를 추가함.
        '''
        self.img_clips.make_gif(save_path, reverse=reverse)


    def to_image(self, crop=False):
        '''
        canvas이미지만 crop하여 image로 변환.
        eps파일을 메모리에 생성할 수 없어서 임시 파일로 생성후 open함.

        postscrip시 dpi에 따라 크기가 변하므로,
        1. 이미지를 포함하여 전체 canvas를 영역을 postscript실시하고,
        2. canvas크기로 resize
        3. 이미지 영역만 crop함.
        4. image 리턴.
        '''

        self.crop_box.hide()
        self.imager.show()

        temp_file = cfg.TEMP_EPS_PATH
        self.canvas.postscript(file=temp_file, colormode = 'color')
        img = Image.open(temp_file).resize((self.canvas.winfo_width(), self.canvas.winfo_height()))

        if crop:
            img = img.crop((self.crop_box.rect()))
        else:
            img = img.crop((0, 0, *self.imager.image_wh()))

        self.crop_box.show()

        return img


    def to_image_frame(self, crop=False):
        '''
        canvas에 그려진 것들을 모두 합쳐서 image로 변환.
        eps파일을 메모리에 생성할 수 없어서 임시 파일로 생성후 open함.

        postscrip시 dpi에 따라 크기가 변하므로,
        1. 전체 canvas를 영역을 postscript실시하고,
        2. canvas크기로 resize
        3. 이미지 영역만 crop함.
        4. ImageFrame으로 변환.
        '''
        
        self.crop_box.hide()
        self.imager.hide()
            
        temp_file = cfg.TEMP_EPS_PATH
        self.canvas.postscript(file=temp_file, colormode = 'color')
        img = Image.open(temp_file).resize((self.canvas.winfo_width(), self.canvas.winfo_height()))

        if crop:
            img = img.crop((self.crop_box.rect()))
        else:
            # img = img.crop((0, 0, *self.imager.image_wh()))
            pass

        self.imager.show()
        self.crop_box.show()

        return ImgFrame(img)


    def last_draw_clip(self, max_count=20):
        '''
        inference시 사용할 수 있도록 마지막 frame을
        그림 그리는 순서대로 누적하여 clip생성함.
        '''
        clips = VideoClip()
        draw_items = self.drawer.get_draw_items()

        count = min(len(draw_items), max_count)
        draw_items = draw_items[-count:]

        # 모든 이미지를 지우고...
        for items in draw_items:
            for item in items:
                self.canvas.itemconfig(item, state='hidden')

        # 그림 그리는 각 스텝을 이미지로 저장함.
        for items in draw_items:
            for item in items:
                self.canvas.itemconfig(item, state='normal')

            frm = self.to_image_frame(crop=False)
            clips.append(frm)

        return clips


    def to_video_clip(self, save_path, crop=False, reverse=False):
        '''
        empty image는 추가 안함.
        reverse : False: 하나씩 그려가면서 이미지 저장.
        reverse : True: 하나씩 지워가면서 이미지 저장.
        '''
        clips = VideoClip()
        draw_items = self.drawer.get_draw_items()

        # 전체 이미지를 첫 이미지로 설정.
        frm = self.to_image_frame(crop=crop)
        clips.append(frm)

        # 모든 이미지를 지우고...
        for items in draw_items:
            for item in items:
                self.canvas.itemconfig(item, state='hidden')

        # 그림 그리는 각 스텝을 이미지로 저장함.
        prev_items = None
        for items in draw_items:
            for item in items:
                self.canvas.itemconfig(item, state='normal')

            if prev_items is not None:
                for pitem in prev_items:
                    self.canvas.itemconfig(pitem, state='hidden')

            prev_items = items
            frm = self.to_image_frame(crop=True)
            clips.append(frm)

        # 다시 모든 이미지를 표시(복구).
        for items in draw_items:
            for item in items:
                self.canvas.itemconfig(item, state='normal')

        # add all item image
        # TODO: 전체 이미지가 두번 추가되어 이부분은 막아야함.
        frm = self.to_image_frame(crop=True)
        clips.append(frm)

        new_path = not_duplicated_path(save_path)
        clips.make_gif(new_path, reverse=reverse)


    def undo(self):
        self.drawer.draw_undo()


    def redo(self):
        self.drawer.draw_redo()


    def reset(self):
        self.img_clips.reset()
        self.drawer.reset()
        self.imager.reset()
        self.canvas.delete('all')


    def update_crop_box(self):
        self.crop_box.update()


    def canvas_wh(self):
        # size가 514로 읽혀져 512로 값으로 설정.
        # return self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight()
        return cfg.CANVAS_SIZE


    def load_img(self, file_path, ratio=1.,):
        self.reset()
        w, h = self.canvas_wh()
        self.imager.load(file_path, ratio=ratio, max_w=w, max_h=h)
        self.crop_box.update_size(w, h)


