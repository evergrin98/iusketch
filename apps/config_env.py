import os

'''
    환경 및 설정 파일.
'''

# Model 관련 상수.
DATA_IMG_W = 128 #256
DATA_IMG_H = 128 #256
DATA_TIME_STEP = 10
DATA_BATCH_SIZE = 8
DATA_IMG_SIZE = DATA_IMG_W, DATA_IMG_H


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
PROJECT_BASE_PATH = '/aiffel/aiffel/iu/code/'  #lms
# PROJECT_BASE_PATH = '/home/evergrin/iu/'    # local

RAW_CLIP_PATH = os.path.join(PROJECT_BASE_PATH, 'datas/data_set/')
MODEL_SAVE_PATH = os.path.join(PROJECT_BASE_PATH, 'datas/models/')
TEMP_DATA_PATH = os.path.join(PROJECT_BASE_PATH, 'datas/temp/')
USER_DRAW_FILE = os.path.join(TEMP_DATA_PATH, 'user_draw.gif')

IMG_LOAD_BASE_PATH = os.path.join(PROJECT_BASE_PATH, 'datas/imgs/raw_imgs/cropped/')
IMG_SAVE_BASE_PATH = os.path.join(PROJECT_BASE_PATH, 'datas/imgs/raw_imgs/raw_gif/')
TEMP_EPS_PATH = os.path.join(TEMP_DATA_PATH, '_temp.eps')
