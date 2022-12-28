import tkinter as tk

import numpy as np
from classes.zoom_box import ZoomBox


class CanvasZoomBox(ZoomBox):
    '''
    Canvas의 일정 영역을 선택하는 ZoomBox
    '''
    def __init__(self, canvas, w=300, h=300, max_x=512, max_y=512, color='red'):
        super().__init__(w, h, max_x, max_y)
        self.canvas = canvas
        self.zoom_box_id = None
        self.box_color = color


    def update(self):
        if self.zoom_box_id is not None:
            self.canvas.delete(self.zoom_box_id)
        self.zoom_box_id = self.canvas.create_rectangle(*self.rect(), outline=self.box_color)
        self.canvas.focus_force()


    def update_size(self, w, h):
        self.set_size(w, h) # set_size를 직접호출하면 안됨.
        self.set_wh(w, h)
        self.update()


    def show(self):
        self.canvas.itemconfigure(self.zoom_box_id, state='normal')

    def hide(self):
        self.canvas.itemconfigure(self.zoom_box_id, state='hidden')