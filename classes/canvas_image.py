import tkinter as tk

import numpy as np
from PIL import Image, ImageTk


class CanvasImage():
    '''
    Canvas에 이미지 표시.
    '''
    def __init__(self, canvas, blurry=False):
        self.canvas = canvas
        self.container = None
        self.image = None
        self.photo = None
        self.ratio = 1.0
        self.blurry = blurry


    def load(self, img_path, ratio=1.0, max_w=0, max_h=0):
        self.image = Image.open(img_path)
        self.ratio = ratio        
        w, h = self.image.size

        if ratio != 1.0:
            img_size = (int(w*ratio), int(h*ratio))
            self.image = self.image.resize(img_size, Image.LANCZOS)

        w, h = self.image.size

        # image boundary check.
        nw = w
        nh = h
        if max_w != 0 and max_h != 0:
            if w >= max_w and h >= max_h:
                if w > h:
                    nw = max_w
                    nh = int((h * max_w) / w)
                else:
                    nh = max_h
                    nw = int((w * max_h) / h)
            elif w >= max_w:
                nw = max_w
                nh = int((h * max_w) / w)
            elif h >= max_h:
                nh = max_h
                nw = int((w * max_h) / h)

            img_size = (nw, nh)
            self.image = self.image.resize(img_size, Image.LANCZOS)
        # else:
        #     if ratio != 1.0:
        #         img_size = (int(w*ratio), int(h*ratio))
        #         self.image = self.image.resize(img_size, Image.LANCZOS)

        if self.blurry:
            self.image = self.image.point( lambda p: 255 if p > 200 else 100 )

        w, h = self.image.size
        self.photo = ImageTk.PhotoImage(self.image)
        ix = int((max_w - w) / 2)
        iy = int((max_h - h) / 2)

        if self.container is not None:
            # img_container가 있으면 다시 생성.
            del self.container

        self.container = self.canvas.create_image(ix, iy, anchor=tk.NW, image=self.photo)


    def image_wh(self):
        return self.image.size[0], self.image.size[1]

    def wh(self):
        return self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight()

    def reset(self):
        self.container = None

    def show(self):
        self.canvas.itemconfig(self.container, state='normal')

    def hide(self):
        self.canvas.itemconfig(self.container, state='hidden')
        