# zoom box.


class ZoomBox():
    '''
    Zoom 이되는 box를 관리하는 클래스.
    '''
    def __init__(self, w = 300, h = 300, max_x=500, max_y=500):
        '''ZoomBox 초기화.'''
        self.x_step = self.y_step = 5    # x/x축 이동 step크기
        self.min_x = self.min_y = 64    # ZoomBox최소 width/height
        self.width = w      # ZoomBox width
        self.height = h     # ZoomBox height
        self.set_size(max_x, max_y)


    def __str__(self):
        str = f"({self.width},{self.height}) {self.sx},{self.sy},{self.ex},{self.ey}"
        print(str)
        return str

    def set_size(self, max_x, max_y):
        '''
        이미지 변경시 이미지 크기를 x,y의 최대 크기로 설정함.
        '''
        self.max_x = max_x
        self.max_y = max_y
        self.width = min(self.width, self.max_x)
        self.height = min(self.height, self.max_y)
        self.sx = int((max_x - self.width) // 2)
        self.sy = int((max_y - self.height) // 2)
        self.ex = self.sx + self.width
        self.ey = self.sy + self.height

    def set_wh(self, w, h):
        '''
        ZoomBox w, h 설정.
        '''        
        self.width = min(self.max_x, w)
        self.height = min(self.max_y, h)
        self.sx = int( (self.sx + self.ex - self.width) // 2)
        self.sy = int( (self.sy + self.ey - self.height) // 2)
        self.ex = self.sx + self.width
        self.ey = self.sy + self.height

    def zoom_in(self, ratio=1.1):
        '''
        ZoomBox를 크게함.
        '''
        w = min(self.max_x, int(self.width * ratio))
        h = min(self.max_y, int(self.height * ratio))
        self.set_wh(w, h)

    def zoom_out(self, ratio=0.9):
        '''
        ZoomBox를 작게함.
        '''            
        w = max(self.min_x, int(self.width * ratio))
        h = max(self.min_y, int(self.height * ratio))
        self.set_wh(w, h)

    def left(self):
        ''' ZoomBox 왼쪽으로 이동.'''
        self.sx = max(self.sx - self.x_step, 0)
        self.ex = self.sx + self.width

    def right(self):
        ''' ZoomBox 오른쪽으로 이동.'''
        self.ex = min(self.ex + self.x_step, self.max_x)
        self.sx = self.ex - self.width

    def up(self):
        ''' ZoomBox 위로 이동.'''
        self.sy = max(self.sy - self.y_step, 0)
        self.ey = self.sy + self.height

    def down(self):
        ''' ZoomBox 아래로 이동.'''
        self.ey = min(self.ey + self.y_step, self.max_y)
        self.sy = self.ey - self.height
    
    def rect(self):
        ''' ZoomBox 좌표를 리턴.'''
        return (self.sx, self.sy, self.ex, self.ey)

    def wh(self):
        ''' ZoomBox w, h를 리턴.'''
        return self.ex - self.sx, self.ey - self.sy

