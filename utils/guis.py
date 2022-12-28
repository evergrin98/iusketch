import tkinter as tk
from tkinter import ttk
# import tkthread ; tkthread.patch()


unit_x = 8 # 10
unit_y = 16 # 25
x_offset = 2
y_offset = 1

unit_line = 20

g_default_ipad = {'ipadx':10, 'ipady':10}


class BaseGuiClass(tk.Tk):
    '''
    gui를 생성을 위한 기본 클래스.
    '''
    
    def __init__(self):
        ''' tk의 master를 self로 생성한다. '''
        super().__init__()
        self._last_pos = {'x':x_offset, 'y':y_offset, 'w':0, 'h':1}   # x, y, w, h

    def wgt_xy(self, wgt):
        # 올바로 동작안함.
        return (wgt.winfo_x(), wgt.winfo_y())

    def side_xy(self, pos=None, offset=0):
        if pos is None:
            return {'x': self._last_pos['x'] + self._last_pos['w'] + offset,
                    'y': self._last_pos['y'] }
        else:
            return {'x': pos['x'] + pos['w'] + offset, 'y': pos['y'] }


    def next_xy(self, pos=None, offset=1):
        if pos is None:
            return {'x': self._last_pos['x'], 'y': self._last_pos['y'] + offset }
        else:
            return {'x': pos['x'], 'y': pos['y'] + offset }


    def unit_place(self, wgt, x=0, y=0, w=20, h=1):
        wgt.pack()
        self._last_pos = {'x':x, 'y':y, 'w':w, 'h':h}
        wgt.place(x=x*unit_x, y=y*unit_y, width=w*unit_x, height=h*unit_y)


    def add_label(self, text="", x=0, y=0, w=0, h=1):
        wgt = tk.Label(self, text=text, fg="black", relief="solid")
        if 0 == w: w = len(text)
        
        self.unit_place(wgt, x, y, w, h)
        return wgt


    def add_entry(self, x=0, y=0, w=0, h=1):
        wgt = tk.Entry(self)
        self.unit_place(wgt, x, y, w, h)
        return wgt


    def add_button(self, text="", command=None, x=0, y=0, w=0, h=1):
        wgt = tk.Button(self, text=text, command=command, 
                       overrelief="solid", 
                       repeatdelay=1000, repeatinterval=100)
        self.unit_place(wgt, x, y, w, h)
        return wgt


    def add_canvas(self, x=0, y=0, w=0, h=0):
        wgt = tk.Canvas(self, bd=0, bg='white', width=w*unit_x, height=h*unit_y)
        self.unit_place(wgt, x, y, w, h)
        return wgt


    def add_combo(self, value_list=["none"], idx=0, x=0, y=0, w=20, h=1, state="readonly"):
        wgt = ttk.Combobox(self,
                           state=state,
                           values=value_list,
                           )
        wgt.set(value_list[idx])
        self.unit_place(wgt, x, y, w, h)
        return wgt


    # def update(self):
    #     # tk.update()
    #     pass

    def runModal(self):
        self.mainloop()


'''
tk (https://076923.github.io/posts/Python-tkinter-14/ 참조)
the core of Tk is single threaded
PIL, cv2를 같이 사용.

부모widget을 master, 자식을 children이라함.
    root window(application window) > PanedWindow > frame widgets > widgets

Toplevel(외부 윈도우): Toplevel을 이용하여 다른 위젯들을 포함하는 외부 윈도우를 생성

Event객체.
    Event종류는 23강 Bind참조.
    command에 lambda를 사용하여 여러 파라메타를 전달 가능.
    command=lambda: command_args(arg1, arg2, arg3)

Bind를 이용하여 위젯들의 이벤트와 실행할 함수를 설정.
    widget.Bind("이벤트", 함수)

config 또는 dictionary형태로 파라메타를 설정.
    widget.config(text="strxx")
    widget['state'] = tk.NORMAL

Font, Canvas, PhotoImage, Sizegrip(크기 조절), Treeview(표)
font=tkinter.font.Font(family="맑은 고딕", size=20, slant="italic")
canvas=tkinter.Canvas(window, relief="solid", bd=2)
image=tkinter.PhotoImage(file="a.png")


위젯 배치: 먼저 선언한 순서대로 배치.
    pack: 상대 위치 배치, place와는 같이 사용 가능: 
        side(해당 구역으로 위젯을 이동): top, bottom, left, right
        anchor(현재 배치된 구역 안에서 특정 위치로 이동):  center, n, e, s, w, ne, nw, se, sw
        fill(할당된 공간에 맞게 크기가 변경): none, x, y, both
        expand(할당되지 않은 미사용 공간을 모두 할당): Boolean

    grid: 셀 단위로 배치, 여러 셀을 건너 뛰어 배치할 수 없다. place와는 같이 사용 가능
        row, column : 해당 구역으로 위젯을 이동시킵니다.
        rowspan, columnspan : 현재 배치된 구역에서 위치를 조정합니다.
        sticky : 현재 배치된 구역 안에서 특정 위치로 이동시킵니다.

    place: place의 절대 위치로 배치되며, 크기를 조정.
        x,y	:좌표 배치
        relx, rely :배치 비율: 0 ~ 1
        width, height : 위젯의 크기
        relwidth, relheight: 위젯의 크기 비율: 0 ~ 1
        anchor:위젯의 기준 위치 : nw n, e, w, s, ne, nw, se, sw
    
    
기본 파라메터:
    width,height : 크기.
    relief : 테두리 모양(flat	flat, groove, raised, ridge, solid, sunken)
    borderwidth : 테두리 두께 2
    background : 배경 색상
    foreground : 문자열 색상
    padx : 테두리와 내용의 가로 여백
    pady : 테두리와 내용의 세로 여백
'''

'''
make Button
command: 클릭이벤트 함수.

함수:
    invoke(): 버튼 실행
    flash(): 깜빡임
'''


'''
make Entry
command(event): 클릭이벤트 함수.event내용이 event_str로 전달됨 표시됨.

함수:
    bind(): key나 mouse 등의 event를 처리하여 메서드나 함수를 실행
    insert(index, “문자열”)	index 위치에 문자열 추가
    delete(start_index, end_index)	start_index부터 end_index까지의 문자열 삭제
    get()	기입창의 텍스트를 문자열로 반환
    index(index)	index에 대응하는 위치 획득
    icursor(index)	index 앞에 키보드 커서 설정
    select_adjust(index)	index 위치까지의 문자열을 블록처리
    select_range(start_index, end_index)	start_index부터 end_index까지 블록처리
    select_to(index)	키보드 커서부터 index까지 블록처리
    select_from(index)	키보드 커서의 색인 위치를 index 위치에 문자로 설정하고 선택
    select_present()	블록처리 되어있는 경우 True, 아닐 경우 False
    select_clear()	블록처리 해제
    xview()	가로스크롤 연결
    xview_scroll(num, str)        	                      가로스크롤의 속성 설정    
    
entry.bind("<Return>", command) # enter키를 누를때 실행할 함수
'''


if __name__ == "__main__":
    """ 
    main함수.
    """

    dlg = BaseGuiClass()
    dlg.runModal()
    
    
  