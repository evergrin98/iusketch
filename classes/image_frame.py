import copy
import numpy as np
import math
import cv2

from PIL import Image
from PIL.PngImagePlugin import PngImageFile



class ImgFrame():
    '''
        Image의 한 frame을 나타냄.  arry.ndim은 3이어야 함.
        arry(float, 0 - 1.0) : height, width, channel순서의 3차원(h, w, c)
        use: 사용여부
        arry는 항상 normalized된 상태로 보관.
        생성시에는 string, Image, ImgFrame, array 입력 가능함.
        나머지 함수는 ImgFrame이나 arry만 입력받음.
    '''

    # Class variable
    ary_dtype = np.float32
    img_dtype = np.uint8

    @staticmethod
    def norm(arry):
        if np.max(arry) > 1.0:
            arry = arry / 255.0
        return arry

    @staticmethod
    def denorm(arry):
        if True: #np.max(arry) <= 1.0:
            arry = arry * 255.
        return arry.astype(ImgFrame.img_dtype)

    @staticmethod
    def clip(arry, min_val=0.0, max_val=1.0):
        return np.clip(arry, min_val, max_val)


    def __init__(self, img, use=True, grayscale=False, do_norm=True):
        '''
            생성 인자로 string, Image, ImgFrame, array 입력 가능함.
            array입력시 do_norm 옵션 필히 확인!!
        '''
        if isinstance(img, str):
            pilImg = Image.open(img)
            self.load_from_img(pilImg, grayscale, use)

        elif isinstance(img, PngImageFile):
            # PIL.Image 입력시.
            self.load_from_img(img, grayscale, use)

        elif isinstance(img, ImgFrame):
            # ImgFrame 입력시.
            self.copy_from(img)

        else:
            # ndarray 입력시.
            self.arry = self.valid_arry(img, do_norm=do_norm)
            self.use = use


    def load_from_img(self, img, grayscale, use):
        ''' PIL.Image로부터 생성 '''
        if grayscale:
            img = img.convert('L')
        self.arry = self.valid_arry(np.asarray(img))
        self.use = use


    def copy_from(self, imgfrm):
        ''' ImgFrame으로부터 복사 생성 '''
        self.arry = self.valid_arry(imgfrm.arry, do_norm=False)
        self.use = imgfrm.use

    def valid_arry(self, arry, do_norm=True):
        '''
        ndim == 3(h,w,c)인 ndarray로 만듦.
        channel수는 고정하지 않음.
        '''
        if not isinstance(arry, np.ndarray):
            arry = np.array(arry, ImgFrame.ary_dtype)

        if arry.ndim == 2:
            # channel last로 확장.
            arry = np.expand_dims(arry, axis=-1)
        elif arry.ndim == 4:
            # 0번 axie를 날림.
            arry = np.squeeze(arry, axis=0)
        elif arry.ndim != 3:
            raise Exception("shape not correct!!")

        if do_norm:
            return self.norm(arry)
        else:
            return arry


    def valid_image(self, arry=None, channel=-1):
        '''
        arry로부터 Image를 생성함.
        image이므로 channel개수가 4까지만 지원됨.
        '''
        if arry is None:
            arry = self.arry

        channel_count = arry.shape[2]
        arry = self.denorm(arry)
        arry = arry.astype(ImgFrame.img_dtype)

        if channel < channel_count and channel >= 0:
                # grayscale은 2dim
                return Image.fromarray(np.squeeze(arry[:, :, channel:channel+1], axis=2), 'L')
        else:
            if 3 == channel_count:
                return Image.fromarray(arry, 'RGB')
            elif 4 == channel_count:
                return Image.fromarray(arry, 'RGBA')
            elif 1 == channel_count:
                # grayscale은 2dim
                return Image.fromarray(np.squeeze(arry, axis=2), 'L')
            else:
                raise Exception("invalid image shape:", arry.shape)
            

    def to_image(self, save_file="", channel=-1):

        if len(save_file) > 0:
            img = self.valid_image(channel=channel)
            img.save(save_file)
            return img
        else:
            return self.valid_image(channel=channel)


    def to_flatten_image(self):
        return self.valid_image(self.merged())


    def __str__(self):
        str = f"{self.arry.shape}, use:{self.use}"
        return str


    def shape(self):
        return self.arry.shape


    def img_wh(self):
        return self.arry.shape[0], self.arry.shape[1]


    def imgSum(self, img):
        if isinstance(img, np.ndarray):
            arry = self.valid_arry(img)
            arry = self.clip(self.arry + arry)
        elif isinstance(img, ImgFrame):
            arry = self.clip(self.arry + img.arry)
        else:
            raise Exception("invalid imgframe type")

        return ImgFrame(arry, self.use)


    def imgResize(self, width, height, inplace=True):
        '''
        이미지 크기 변경.
        channel이 4보다 클때는 지원안됨..
        '''
        img = self.valid_image()
        img = img.resize((width, height), Image.LANCZOS)

        arry = self.valid_arry(np.asarray(img, ImgFrame.ary_dtype))

        if inplace:
            self.arry = arry

        return arry


    def append_channel(self, img, do_norm=True):
        '''
        w,h가 동일한 frame을 channel방향으로 쌓는다.
        '''
        arry = None
        if isinstance(img, ImgFrame):
            # ImgFrame 입력시.
            arry = img.arry
        else:
            # ndarray 입력시.
            arry = self.valid_arry(img, do_norm=do_norm)

        if self.arry.shape[0:2] != arry.shape[0:2]:
            raise Exception("width x height not correct")

        self.arry = np.dstack((self.arry, arry))


    def merged(self, merge_fn=np.min):
        '''
        channel방향으로 어레이를 합침.
          - 채널1개(grayscale)만 고려함.
        합치는 방법은 흰색에 검은색 이미지는 np.min사용
        '''
        arry = merge_fn(self.arry, axis=2)
        arry = self.clip(arry)
        arry = self.valid_arry(arry)

        if arry.shape[0:2] != arry.shape[0:2]:
            raise Exception("width x height not correct")

        if arry.shape[2] != 1:
            raise Exception("merged channel not 1")

        return arry


    def threshold(self, threshold=0.5, low=0.0, high=1.0, inplace=True):
        ''' 
        threshold를 기준으로 low/high값으로 변환.
        '''
        arry = np.where(self.arry < threshold, low, high)
        if inplace:
            self.arry = arry
        else:
            return arry


    def minus_frame(self, imgfrm):
        '''
        흑백(배경 흰색), 0-1norm된 데이터에서...
        현재 이미지에서 imgfrm이미지를 뺀 이미지를 생성함.
        '''
        if not isinstance(imgfrm, ImgFrame):
            raise Exception("img is not ImgFrame")

        if self.arry.shape != imgfrm.arry.shape:
            raise Exception("img frame size not correct")

        arry = np.where(imgfrm.arry < 1.0, 1.0, self.arry)

        return ImgFrame(arry)


    def out_box(self):
        '''
        cv2 contour은 흰색의 물체를 검출.
        프레임의 전체 이미지 외곽 박스 좌표를 리턴함.
        '''
        img = self.valid_image()
        imgary = np.array(img)
        # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # imgray = np.invert(imgray)
        # print(imgary.shape)

        ret,thresh = cv2.threshold(src=imgary, thresh=200, maxval=255, type=0)
        thresh = cv2.bitwise_not(thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            # print("no out line!!!!")
            # self.to_image(save_file='/home/evergrin/iu/datas/imgs/data_set/error.png')
            return 0, 0, 0, 0

        sxs = []
        sys = []
        exs = []
        eys = []

        for cntur in contours:
            x,y,w,h = cv2.boundingRect(cntur)
            sxs.append(x)
            sys.append(y)
            exs.append(x+w)
            eys.append(y+h)

        sxs = np.array(sxs)
        sys = np.array(sys)
        exs = np.array(exs)
        eys = np.array(eys)

        sx,sy,ex,ey = np.min(sxs), np.min(sys), np.max(exs), np.max(eys)
        # print(sx,sy,ex,ey)

        return sx, sy, ex, ey

    def fill_box(self, val=0.0):
        ''' 
        out box를 val로 채운다.
        '''
        sx, sy, ex, ey = self.out_box()
        self.arry[sy:ey+1, sx:ex+1, :] = val


    def splitted_frames(self, dx=100, dy=100):
        '''
        외곽 박스 크기가 dx, dy보다 크면 split하여 
        동일한 크기의 frame 리스트로 생성하여 리턴함.
        '''
        sx, sy, ex, ey = self.out_box()
        w = ex-sx
        h = ey-sy
        x_cnt = max(1, w // dx )
        y_cnt = max(1, h // dy )

        # print(f"frame {w}x{h}, split x:{x_cnt}, y:{y_cnt}")

        if x_cnt == 1 and y_cnt == 1:
            return [self]

        x_ary = np.arange(sx, ex+1)
        xes = np.array_split(x_ary, x_cnt)
        y_ary = np.arange(sy, ey+1)
        yes = np.array_split(y_ary, y_cnt)

        xminmax = [ [ary[0], ary[-1] + 1] for ary in xes]
        yminmax = [ [ary[0], ary[-1] + 1] for ary in yes]
        
        frames = []
        for xmin, xmax in xminmax:
            for ymin, ymax in yminmax:
                emptyary = np.ones_like(self.arry)
                emptyary[ymin:ymax, xmin:xmax, ...] = self.arry[ymin:ymax, xmin:xmax, ...]
                if np.min(emptyary) < 0.9:
                    frames.append(ImgFrame(emptyary, do_norm=False))
                else:
                    # print('split empty frame skip')
                    pass

        return frames







if __name__ == "__main__":
    """ 
    test main함수.
    """
    import matplotlib.pyplot as plt
    

    file_path = '/home/evergrin/iu/datas/data_set2/aaa.png'

    imgfrm = ImgFrame(img=file_path, grayscale=True)
    imgfrm.fill_box()
    
    plt.imshow(imgfrm.to_image(), cmap='gray')
    # sx, sy, ex, ey = imgfrm.out_box()

    # img = imgfrm.to_image()
    # imgary = np.array(img)
    # emptyary = np.ones_like(imgary) * 255
    # w = (ex - sx) // 2
    # h = (ey - sy) // 2
    # emptyary[sy:sy+h, sx:sx+w, ...] = imgary[sy:sy+h, sx:sx+w, ...]

    # img = cv2.rectangle(emptyary, (sx,sy), (ex,ey), (0, 0, 255), 2)
    # plt.imshow(emptyary, cmap='gray')
    
    # print(sx, sy, ex, ey)
    imgfrms = imgfrm.splitted_frames(dx=50, dy=50)
    for imf in imgfrms:
        img = imf.to_image()
        plt.imshow(img, cmap='gray')

    arry1 = np.zeros((3, 4, 1))
    arry2 = np.ones((3, 4, 2))
    arry3 = np.dstack((arry1, arry2))
    # arry1 = np.array([[[0,0,1],[0, 0, 2]],[[0,0,1],[0, 0, 2]]])
    # arry2 = [[1,1,1],[2, 2, 2]]
    print("ary1:", arry1.shape, arry1)
    print("ary2:", arry2.shape, arry2)
    print("ary3:", arry3.shape, arry3)

    

    
    

