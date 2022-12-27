import tkinter as tk

import numpy as np


class CanvasDrawer():
    '''
    Canvas 마우스로 그림그리기.
    '''
    def __init__(self, canvas, pen_color='black'):
        self.canvas = canvas
        
        self.draw_items = list()
        self.draws_last = list()
        self.draws_undo = list()
        self.mouse_prev_x = None
        self.mouse_prev_y = None
        self.pen_color = pen_color
        self.pen_size = 2
        
        # bind event.
        self.canvas.bind("<B1-Motion>", self.mouse_move)
        self.canvas.bind("<Button-1>", self.mouse_left_click)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_left_release)
        self.canvas.bind("<Enter>", self.mouse_enter)
        self.canvas.bind("<Leave>", self.mouse_leave)


    def reset(self):
        self.draws_last.clear()
        self.draw_items.clear()
        self.draws_undo.clear()


    def get_draw_items(self):
        return self.draw_items


    def all_hide(self):
        # reverse_items = self.draw_items
        # reverse_items.reverse()
        for items in self.draw_items:
            for item in items:
                self.canvas.itemconfig(item, state='hidden')


    def all_show(self):
        for items in self.draw_items:
            for item in items:
                self.canvas.itemconfig(item, state='normal')


    def mouse_enter(self, event):
        ''' 이미지 로드 디렉토리 설정.'''
        self.canvas.config(cursor="star")


    def mouse_leave(self, event):
        ''' 이미지 로드 디렉토리 설정.'''
        self.canvas.config(cursor="")


    def mouse_left_click(self, event):
        ''' 
        왼쪽 마우스 클릭시.
        이미지를 두배 확대한 상태이므로, 
        원본으로 축소시 위치를 동일하게 가져가기 위해  x, y를 2의 배수로 변환함. 
        '''
        self.mouse_prev_x = event.x - (event.x % 2)
        self.mouse_prev_y = event.y - (event.y % 2)
        self.draws_last = list()


    def mouse_left_release(self, event):
        ''' 왼쪽 마우스 릴리즈시.'''
        self.mouse_prev_x = None
        self.mouse_prev_y = None

        if len(self.draws_last) > 0:
            self.draw_items.append(self.draws_last)
            self.draws_undo.clear()


    def mouse_move(self, event):
        ''' 
        마우스 드래그로 이미지에 드로우함.
        이미지를 두배 확대한 상태이므로, 
        원본으로 축소시 위치를 동일하게 가져가기 위해  x, y를 2의 배수로 변환함. 
        '''
        sxy = np.array([self.mouse_prev_x, self.mouse_prev_y])
        x =  event.x - (event.x % 2)
        y =  event.y - (event.y % 2)
        exy = np.array([x, y])
        s1, s2, s3, s4 = self.square_point(sxy, exy, self.pen_size)
        item = self.canvas.create_oval(sxy[0] - self.pen_size, 
                                       sxy[1] - self.pen_size, 
                                       sxy[0] + self.pen_size, 
                                       sxy[1] + self.pen_size, 
                                       fill=self.pen_color, outline=self.pen_color, width=1)
        self.draws_last.append(item)

        item = self.canvas.create_oval(exy[0] - self.pen_size, 
                                       exy[1] - self.pen_size, 
                                       exy[0] + self.pen_size, 
                                       exy[1] + self.pen_size, 
                                       fill=self.pen_color, outline=self.pen_color, width=1)
        self.draws_last.append(item)

        item = self.canvas.create_polygon(*s1, *s2, *s3, *s4, 
                                          fill=self.pen_color, 
                                          outline=self.pen_color, width=1)
        self.draws_last.append(item)

        self.mouse_prev_x = x
        self.mouse_prev_y = y


    def draw_undo(self):
        if len(self.draw_items) > 0:
            self.draws_undo.append(self.draw_items[-1])
            del self.draw_items[-1]
            for item_id in self.draws_undo[-1]:
                self.canvas.itemconfigure(item_id, state='hidden')
        

    def draw_redo(self):
        if len(self.draws_undo) > 0:
            self.draw_items.append(self.draws_undo[-1])
            del self.draws_undo[-1]
            for item_id in self.draw_items[-1]:
                self.canvas.itemconfigure(item_id, state='normal')


    def rotate90(self, sxy, exy):
        dxy = exy-sxy
        rot90 = np.array([[0, -1], [1, 0]])
        nxy = np.dot(rot90, dxy.T) + sxy
        return sxy, nxy
        
        
    def rotate270(self, sxy, exy):
        dxy = exy-sxy
        rot270 = np.array([[0, 1], [-1, 0]])
        nxy = np.dot(rot270, dxy.T) + sxy
        return sxy, nxy


    def distance(self, sxy, exy, dist):
        xylen = np.sqrt((exy[0] - sxy[0])**2 + (exy[1] - sxy[1])**2)
        ratio = dist / xylen
        nexy = (exy - sxy) * ratio + sxy
        return int(round(nexy[0], 1)), int(round(nexy[1], 1))


    def square_point(self, sxy, exy, r):
        s1 = self.distance(*self.rotate90(sxy, exy), r)
        s2 = self.distance(*self.rotate270(sxy, exy), r)
        s3 = self.distance(*self.rotate90(exy, sxy), r)
        s4 = self.distance(*self.rotate270(exy, sxy), r)
        return s1, s2, s3, s4
