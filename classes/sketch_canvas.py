import tkinter as tk
from tkinter import filedialog, ttk

import config_env as cfg

import pyautogui as pg
import imageio
from PIL import Image, ImageTk, ImageGrab

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
            img = img.crop((0, 0, *self.imager.image_wh()))

        self.imager.show()
        self.crop_box.show()

        return ImgFrame(img)


    def to_video_clip(self, save_path, crop=False, reverse=False):
        clips = VideoClip()
        draw_items = self.drawer.get_draw_items()

        # items = items.reverse()
        for items in draw_items:
            for item in items:
                self.canvas.itemconfig(item, state='hidden')

        # self.drawer.all_hide()

        # add empty image
        frm = self.to_image_frame(crop=True)
        clips.append(frm)

        # add each item image
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

        for items in draw_items:
            for item in items:
                self.canvas.itemconfig(item, state='normal')

        # add all item image
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
        return self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight()


    def load_img(self, file_path, ratio=2.,):
        w, h = self.canvas_wh()
        self.imager.load(file_path, ratio=ratio, max_w=w, max_h=h)
        self.crop_box.update_size(w, h)
    
