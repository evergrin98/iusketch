import time, os

import config_env as cfg

import tkinter as tk
from tkinter import filedialog

import pyautogui as pg
import imageio
from PIL import Image, ImageTk, ImageGrab

from utils.guis import BaseGuiClass
from utils.files import dir_path_change

from classes.sketch_canvas import SketchCanvas



class FrameMaker(BaseGuiClass):
    '''
    FrameMaker 다이얼로그 생성 클래스.
    '''

    def __init__(self):
        ''' widget들을 생성. '''
        super().__init__()

        self.title("FrameMaker")
        self.geometry('550x700')
        self.resizable(0, 0)

        self.load_base_path = cfg.IMG_LOAD_BASE_PATH
        self.save_base_path = cfg.IMG_SAVE_BASE_PATH
        self.load_img_dir = self.load_base_path
        self.save_img_dir = self.save_base_path

        p = self._last_pos.copy()
        self.load_label = self.add_label("load path", **p)
        self.load_entry = self.add_entry(**self.side_xy(), w=30)
        self.load_entry.delete(0, 'end')
        self.load_entry.insert(0, self.load_img_dir)
        xy = self.side_xy(offset=1)
        self.load_dir_btn = self.add_button("select", command=self.load_dir_btn_handler, **xy, w=10)

        xy = self.next_xy(p, offset=1)
        self.save_label = self.add_label("save path", **xy)
        self.save_entry = self.add_entry(**self.side_xy(), w=30)
        self.save_entry.delete(0, 'end')
        self.save_entry.insert(0, self.save_img_dir)
        xy = self.side_xy(offset=1)
        self.save_dir_btn = self.add_button("select", command=self.save_dir_btn_handler, **xy, w=10)
        xy = self.side_xy(offset=1)
        self.open_dir_btn = self.add_button("open", command=self.open_dir_btn_handler, **xy, w=10)

        xy = self.next_xy(p, offset=3)
        self.img_label = self.add_label("None", **xy, w=30)
        xy = self.side_xy(offset=1)
        self.load_img_btn = self.add_button("select", command=self.load_img_btn_handler, **xy, w=10)

        # canvas 512*512 추가.
        self.canvas_wgt = self.add_canvas(x=2, y=6, w=64, h=32)
        self.canvas = SketchCanvas(self.canvas_wgt, 'yellow', 'black')

        xy = self.next_xy(offset=34)
        self.zoom_in_btn = self.add_button("+", command=lambda m="+": self.zoom_btn_handler(m), **xy, w=10)
        self.zoom_out_btn = self.add_button("-", command=lambda m="-": self.zoom_btn_handler(m), **self.side_xy(offset=1), w=10)

        crop_w, crop_h = self.canvas.crop_box.wh()
        self.crop_width_entry = self.add_entry(**self.side_xy(offset=3), w=10)
        self.crop_width_entry.delete(0, 'end')
        self.crop_width_entry.insert(0, crop_w)
        self.crop_height_entry = self.add_entry(**self.side_xy(offset=1), w=10)
        self.crop_height_entry.delete(0, 'end')
        self.crop_height_entry.insert(0, crop_h)

        # event bind
        self.bind("<Key>", self.keyboard_handler)
        self.crop_width_entry.bind('<Return>', self.keyboard_handler)
        self.crop_height_entry.bind('<Return>', self.keyboard_handler)


    def update_crop_box(self):
        ''' cropbox 크기를 갱신함.'''
        crop_w, crop_h = self.canvas.crop_box.wh()
        self.crop_width_entry.delete(0, 'end')
        self.crop_width_entry.insert(0, crop_w)
        self.crop_height_entry.delete(0, 'end')
        self.crop_height_entry.insert(0, crop_h)

        self.canvas.update_crop_box()


    def save_to_gif(self):
        ''' 저장된 clip을 gif 이미지로 저장함.'''

        file_full_name = os.path.basename(self.file_path)
        file_name = os.path.splitext(file_full_name)[0]
        file_name = file_name + ".gif"
        gif_file = os.path.join(self.save_img_dir, file_name)

        self.canvas.clip_to_gif(gif_file, True)


    def keyboard_handler(self, event):
        ''' keyboard입력 이벤트 처리.'''
        if event.keycode == 36: # enter key
            w = int(self.crop_width_entry.get())
            h = int(self.crop_height_entry.get())
            self.canvas.crop_box.set_size(w, h)
        elif event.char == '=':
            self.canvas.crop_box.zoom_in()
        elif event.char == '-':
            self.canvas.crop_box.zoom_out()
        elif event.keycode == 114: # right-arrow key
            self.canvas.crop_box.right()
        elif event.keycode == 113: # left-arrow key
            self.canvas.crop_box.left()
        elif event.keycode == 111: # up-arrow key
            self.canvas.crop_box.up()
        elif event.keycode == 116: # down-arrow key
            self.canvas.crop_box.down()
        elif event.char == 'c':
            ''' crop image & save to png '''
            file_name = dir_path_change(self.file_path, self.save_img_dir, 'png')
            img = self.canvas.to_image(crop=True)
            img = img.resize(cfg.RAW_IMG_SIZE)
            img = img.save(file_name)
            print(file_name + " saved")
            return
        elif event.char == 'a':
            ''' crop image & add to img_clips '''
            self.canvas.add_image_frame(crop=True)
            return
        elif event.char == 'f':
            ''' img_clips to gif file '''
            self.save_to_gif()
            return
        elif event.char == 'l':
            ''' open image '''
            self.load_img_btn_handler()
            return
        elif event.char == 'r':
            ''' reset changes '''
            self.canvas.reset()
            self.load_img_file(self.file_path)
            return
        elif event.char == 's':
            img = self.canvas.add_image_frame(crop=True)
            return
        elif event.char == 'v':
            ''' img_clips to gifs '''
            file_name = dir_path_change(self.file_path, self.save_img_dir, 'gif')
            self.canvas.to_video_clip(file_name, crop=True, reverse=False)
            return
        elif event.char == 'b':
            self.canvas.undo()
        elif event.char == 'n':
            self.canvas.redo()
        else:
            print(event, event.keycode)
            return

        self.update_crop_box()


    def zoom_btn_handler(self, event):
        ''' zoom버튼 이벤트 처리.'''
        if "+" == event:
            self.canvas.crop_box.zoom_in()
        elif "-" == event:
            self.canvas.crop_box.zoom_out()

        self.update_crop_box()


    def open_dir_btn_handler(self):
        os.system(f"nautilus {self.save_img_dir} &")


    def load_dir_btn_handler(self):
        ''' 이미지 로드 디렉토리 설정.'''
        dir_name = filedialog.askdirectory(initialdir=self.load_img_dir)
        if len(dir_name) > 0:
            self.load_img_dir = dir_name
            self.load_entry.delete(0, 'end')
            self.load_entry.insert(0, self.load_img_dir)


    def save_dir_btn_handler(self):
        ''' 이미지 저장 디렉토리 설정.'''
        dir_name = filedialog.askdirectory(initialdir=self.save_img_dir)
        if len(dir_name) > 0:
            self.save_img_dir = dir_name
            self.save_entry.delete(0, 'end')
            self.save_entry.insert(0, self.save_img_dir)


    def load_img_btn_handler(self):
        ''' 로드할 이미지 파일 선택.'''
        self.file_path = filedialog.askopenfilename(initialdir=self.load_img_dir)
        self.load_img_file(self.file_path)


    def load_img_file(self, file_path):
        ''' 이미지 파일을 다이얼로그로 로드함.'''
        self.canvas.load_img(file_path, ratio=2.0)
        file_name = os.path.basename(file_path)
        self.img_label.config(text=file_name)




if __name__ == "__main__":
    """ 
    main함수.
    """

    dlg = FrameMaker()
    dlg.runModal()
    
    