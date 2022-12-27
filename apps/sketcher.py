import tkinter as tk

from utils.guis import BaseGuiClass
from classes.image_frame import ImgFrame
from classes.video_clip import VideoClip
from classes.sketch_canvas import SketchCanvas




class SketcherDialog(BaseGuiClass):
    '''
    이미지를 표시하는 다이얼로그 생성 클래스.
    '''

    def __init__(self):
        ''' widget들을 생성. '''
        super().__init__()

        self.title("Sketcher AI")
        self.geometry('640x650')  # widthxheight
        self.resizable(0, 0)
        
        p = self._last_pos.copy()
        xy = self.next_xy(p, offset=1)
        self.combo = self.add_combo(
            value_list=["아빠사자", "엄마사자", "아기사자"],
            **xy, w=30, h=2,
        )
        
        self.canvas_wgt = self.add_canvas(x=2, y=4, w=64, h=32)
        self.canvas = SketchCanvas(self.canvas_wgt)
        
        xy = self.side_xy(offset=2)
        xy = self.next_xy(xy, offset=2)
        self.ai_btn = self.add_button("Ai Draw", command=None, **xy, w=10, h=2)
        xy = self.next_xy(offset=2)
        self.finish_btn = self.add_button("Finish", command=None, **xy, w=10, h=2)

        xy = self.next_xy(offset=4)
        self.undo_btn = self.add_button("Undo", command=self.canvas.undo, **xy, w=10, h=2)
        xy = self.next_xy(offset=2)
        self.redo_btn = self.add_button("Redo", command=self.canvas.redo, **xy, w=10, h=2)

        xy = self.next_xy(offset=4)
        self.reset_btn = self.add_button("Reset", command=self.canvas.reset, **xy, w=10, h=2)

        xy = self.next_xy(p, offset=36)
        self.status_label = self.add_label("ready...", **xy, w=64, h=2)


        # event bind
        self.bind("<Key>", self.keyboard_handler)




    def update_status(self, status_str):
        ''' status 갱신함.'''
        self.status_label.config(text=status_str)


    def keyboard_handler(self, event):
        ''' keyboard입력 이벤트 처리.'''
        if event.char == 'r':
            self.canvas.reset()
            return
        elif event.char == 'b':
            self.canvas.undo()
            return
        elif event.char == 'n':
            self.canvas.redo()
            return
        else:
            print(event, event.keycode)
            return



if __name__ == "__main__":
    """ 
    main함수.
    """

    dlg = SketcherDialog()
    dlg.runModal()
    
    