
'''
    환경 및 설정 파일.
'''
# Dataset 관련 상수.
DATA_IMG_W = 256
DATA_IMG_H = 256
DATA_TIME_STEP = 20
DATA_BATCH_SIZE = 8

DATA_IMG_SIZE = DATA_IMG_W, DATA_IMG_H

RAW_CLIP_PATH = '/home/evergrin/iu/datas/imgs/raw_imgs/raw_gif/'



# Dialog 상수.
RAW_IMG_W = 256
RAW_IMG_H = 256
RAW_IMG_SIZE = RAW_IMG_W, RAW_IMG_H


CANVAS_W = 512
CANVAS_H = 512
CANVAS_SIZE = CANVAS_W, CANVAS_H

'''
os에 관계없이 path를 설정하려면..  "c:/abc/def" 형식으로 입력.
'''
IMG_LOAD_BASE_PATH = '/home/evergrin/iu/datas/imgs/raw_imgs/cropped/'
IMG_SAVE_BASE_PATH = '/home/evergrin/iu/datas/imgs/raw_imgs/raw_gif/'
TEMP_EPS_PATH = '/home/evergrin/iu/datas/imgs/_temp.eps'
