import copy
import numpy as np
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
        if np.max(arry) <= 1.0:
            arry = arry * 255.
        return arry.astype(ImgFrame.img_dtype)

    @staticmethod
    def clip(arry, min_val=0.0, max_val=1.0):
        return np.clip(arry, min_val, max_val)


    def __init__(self, img, use=True, grayscale=False):
        '''
            생성 인자로 string, Image, ImgFrame, array 입력 가능함.
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
            self.arry = self.valid_arry(img)
            self.use = use


    def load_from_img(self, img, grayscale, use):
        ''' PIL.Image로부터 생성 '''
        if grayscale:
            img = img.convert('L')
        self.arry = self.valid_arry(np.asarray(img))
        self.use = use


    def copy_from(self, imgfrm):
        ''' ImgFrame으로부터 복사 생성 '''
        self.arry = self.valid_arry(imgfrm.arry)
        self.use = imgfrm.use

    def valid_arry(self, arry):
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

        return self.norm(arry)


    def valid_image(self, arry=None):
        '''
        arry로부터 Image를 생성함.
        image이므로 channel개수가 4까지만 지원됨.
        '''
        if arry is None:
            arry = self.arry

        channel_count = arry.shape[2]
        arry = self.denorm(arry)
        arry = arry.astype(ImgFrame.img_dtype)

        if 3 == channel_count:
            return Image.fromarray(arry, 'RGB')
        elif 4 == channel_count:
            return Image.fromarray(arry, 'RGBA')
        elif 1 == channel_count:
            # grayscale은 2dim
            return Image.fromarray(np.squeeze(arry, axis=2), 'L')
        else:
            raise Exception("invalid image shape")


    def to_image(self, save_file=""):

        if len(save_file) > 0:
            img = self.valid_image()
            img.save(save_file)
        else:
            return self.valid_image()


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


    def imgResize(self, width, height):
        '''
        이미지 크기 변경.
        channel이 4보다 클때는 지원안됨..
        '''
        img = self.valid_image()
        img = img.resize((height, width))
        self.arry = self.valid_arry(np.asarray(img, ImgFrame.ary_dtype))


    def append_channel(self, img):
        '''
        w,h가 동일한 frame을 channel방향으로 쌓는다.
        '''
        arry = None
        if isinstance(img, ImgFrame):
            # ImgFrame 입력시.
            arry = img.arry
        else:
            # ndarray 입력시.
            arry = self.valid_arry(img)

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





if __name__ == "__main__":
    """ 
    test main함수.
    """
    
    arry1 = np.zeros((3, 4, 1))
    arry2 = np.ones((3, 4, 2))
    arry3 = np.dstack((arry1, arry2))
    # arry1 = np.array([[[0,0,1],[0, 0, 2]],[[0,0,1],[0, 0, 2]]])
    # arry2 = [[1,1,1],[2, 2, 2]]
    print("ary1:", arry1.shape, arry1)
    print("ary2:", arry2.shape, arry2)
    print("ary3:", arry3.shape, arry3)

    

    
    

