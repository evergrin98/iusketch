import numpy as np
import random

from PIL import Image
import imageio as iio
# import imageio.v3 as iio    # gif
# from PIL.PngImagePlugin import PngImageFile

from classes.image_frame import ImgFrame
from albumentations import  ReplayCompose



class VideoClip():
    '''
    ImgFrame(h,w,c)으로 VideoClip을 생성..
    w,h이 동일한 frame만 보관.
    0번이 처음이고 가장 아래 frame임.
    '''
    def __init__(self, frames=[], gif_path="", shape=None, max_clip=4000):
        ''' 
        VideoClip 생성자.
        frames가 있으면 복사 생성함. 
        '''
        self.max_clip = max_clip
        self.shape = shape
        self.clips = list()

        if len(frames) > 0:
            self.load_frames(frames)
        elif len(gif_path) > 0:
            self.load_gif(gif_path)


    def load_frames(self, frames):
        ''' 비어 있을때만 ImgFrame어레이를 복사하여 생성. '''

        if not self.isEmpty():
            raise Exception("clip not empty")

        if not isinstance(frames[0], ImgFrame):
            raise Exception("not ImgFrame array!!")

        for frame in frames:
            self.append(frame)


    def load_gif(self, gif_file, grayscale=True):
        '''
        gif파일로 frame들을 만든다.
        '''
        gif = iio.v3.imread(gif_file, index=None)
        if gif.ndim != 4:
            raise Exception("gif dim not correct")
        if gif.shape[3] != 3:
            raise Exception("gif chn not correct")

        frame_count = gif.shape[0]

        for idx in range(frame_count):
            img = Image.fromarray(gif[idx])
            # img = img.resize((self.img_h, self.img_w))
            if grayscale:
                img = img.convert("L")

            imgfrm = ImgFrame(img)
            self.append(imgfrm)


    def from_array(self, arry_4d, do_norm=False):
        '''
        [frames, height, width, channel]의 4차원 array로 전달.
        expand : True이면 5차원으로 확장함.
        '''
        if arry_4d.ndim != 4:
            raise Exception("array not 4D")

        for i in range(arry_4d.shape[0]):
            imgfrm = ImgFrame(img=arry_4d[i], do_norm=do_norm)
            self.append(imgfrm)


    def to_array(self, expand=False):
        '''
        [frames, height, width, channel]의 4차원 array로 전달.
        expand : True이면 5차원으로 확장함.
        '''
        frames = [ img_frm.arry for img_frm in self.clips]
        arry_4d = np.array(frames)
        if expand:
            return np.expand_dims(arry_4d, axis=0)
        else:
            return arry_4d


    def count(self):
        return len(self.clips)


    def isEmpty(self):
        return 0 == self.count()


    def isFull(self):
        return self.max_clip == self.count()


    def reset(self):
        self.shape = None
        self.clips.clear()


    def __str__(self):
        str = f"clip:{self.count()}/{self.max_clip}, {self.shape}"
        return str


    def resize(self, w, h, inplace=True):
        '''
        clips의 모든 frame을 resize한다.
        inplace인경우는, 내부 clip을 resize한다. 
        '''
        if inplace:
            # 내부 변수를 수정하므로, shape는 별도로 갱신한다.
            for img_frm in self.clips:
                img_frm.imgResize(w, h, inplace=True)
            
            self.shape = self.clips[0].shape()
            return self
        else:
            vclip = VideoClip()
            for img_frm in self.clips:
                img_frm = ImgFrame(img_frm.imgResize(w, h, inplace=False), do_norm=False)
                vclip.append(img_frm)
            return vclip


    def append(self, imgfrm):
        '''
        imgframe을 stack에 추가
        stack의 frame과 shape가 같아야만 추가됨.
        '''
        if not isinstance(imgfrm, ImgFrame):
            raise Exception("not ImgFrame!!")
            # imgfrm = ImgFrame(imgfrm)

        if self.isEmpty():
            if self.shape is None:
                self.shape = imgfrm.shape()
            elif self.shape !=  imgfrm.shape():
                raise Exception("shape not match")
        elif self.isFull():
            raise Exception("stack full")
        elif self.shape != imgfrm.shape():
            raise Exception("img frm shape not same!!", self.shape, imgfrm.shape())

        self.clips.append(imgfrm)


    def merged(self):
        '''
        stack에서 use인 imgframe을 merge하여 합쳐진 imgframe을 생성
        '''

        imgfrm = None
        for img in self.clips:
            if img.use is True:
                if imgfrm is None:
                    imgfrm = img
                else:
                    imgfrm = imgfrm.add(img)

        return imgfrm


    def make_gif(self, gif_file, reverse=False, ratio=1.0, fps=5):
        '''
        frame들을 gif파일로 만든다.
        '''
        frames = list()
        for img_frm in self.clips:
            if img_frm.use is True:

                if ratio != 1.0:
                    w, h = img_frm.img_wh()
                    w = int(w * ratio)
                    h = int(h * ratio)
                    img_frm.imgResize(w, h)

                img = img_frm.to_image()
                frames.append(img)

        if reverse:
            frames.reverse()

        iio.mimsave(gif_file, frames, "GIF", fps=fps)
        print(gif_file, " saved")


    def stacked_frame(self, sidx=0, eidx=-1):
        '''
        모든 imgframe을 channel방향으로 stack하여 imgframe을 생성
        '''
        if eidx == 0:
            eidx = len(self.clips)

        imgfrm = ImgFrame(self.clips[sidx])
        for img in self.clips[sidx:eidx]:
            if img.use is True:
                imgfrm.append_channel(img)

        return ImgFrame(imgfrm.merged(), do_norm=False)


    def all_clips(self, count=20, include_top=False, use_all=False, shuffle='none', overlap=False):
        '''
        전체 clips을 count만큼의 그룹으로 나누어 전달.
        include_top : 첫번째 img는 전체 이미지이므로, 포함할지를 선택함.
        '''
        idx_high = len(self.clips)
        idx_low = 0
        img_cnt = idx_high

        # gif생성시 처음과 끝 프레임 모두 전체 이미지여서 2장을 빼야함;;
        if not use_all:
            idx_low = 1
            idx_high -= 1
            img_cnt -= 2
        
        pick_cnt = count
        if overlap:
            pick_cnt = count + 1
            
        if img_cnt < pick_cnt:
            raise Exception("count not match")

        idx_arry = np.arange(idx_low, idx_high)
        idxes = np.array_split(idx_arry, pick_cnt)
        
        frames = []
        for arry in idxes:

            clip = [ self.clips[idx] for idx in arry ]
            stacked_frame = None

            for frame in clip:
                if stacked_frame is None:
                    stacked_frame = frame
                else:
                    stacked_frame.append_channel(frame)

            img_frame = ImgFrame(stacked_frame.merged(), do_norm=False)
            frames.append(img_frame)

        if shuffle == 'random':
            frames = random.choices(population=frames, k=len(frames))
        elif shuffle == 'forward':
            # clip index를 두번 반복해서 roundQueue처럼 동작.
            idx_list = [ idx for idx in range(0, len(frames))]
            idx_list.extend(idx_list)

            # random start idx
            start_idx = random.randrange(0, len(frames))

            idx_list = idx_list[start_idx:start_idx + len(frames)]

            frames = [ frames[idx] for idx in idx_list ]

        # else: # 'none'
        
        overlaps = []
        if overlap:
            for idx in range(len(frames) - 1):
                frame = frames[idx]
                frame.append_channel(frames[idx+1])
                img_frame = ImgFrame(frame.merged(), do_norm=False)
                overlaps.append(img_frame)
        else:
            overlaps = frames
            

        if include_top:
            # 마지막에 전체 이미지 추가.
            overlaps.append(self.clips[0])

        return VideoClip(frames=overlaps)



    def random_clips(self, count=20, step=1, include_top=False):
        '''
        clips에서 랜덤하게 count만큼 ImgFrame을 뽑아서 clip을 생성함.
        include_top : 첫번째 img는 전체 이미지이므로, 포함할지를 선택함.
          - gif생성시 처음과 끝 프레임 모두 전체 이미지여서 2장을 빼야함.
        '''

        idx_high = len(self.clips)
        idx_low = 0
        img_cnt = idx_high

        # gif생성시 처음과 끝 프레임 모두 전체 이미지여서 2장을 빼야함;;
        idx_low = 1
        idx_high -= 1
        img_cnt -= 2

        pick_cnt = min(img_cnt, count)

        temp_frames = random.choices(population=self.clips[idx_low:idx_high], k=pick_cnt)

        frames = []
        stacked_frame = None

        for i, frame in enumerate(temp_frames):

            if stacked_frame is None:
                stacked_frame = frame
            else:
                stacked_frame.append_channel(frame)

            img_frame = ImgFrame(stacked_frame.merged(), do_norm=False)

            if (i + 1) % step == 0:
                frames.append(img_frame)
                stacked_frame = None

        if include_top:
            # 마지막에 전체 이미지 추가.
            frames.append(self.clips[0])

        return VideoClip(frames=frames)


    def sequential_clips(self, start_idx=None, count=20, step=1, include_top=False, reverse=False):
        '''
        clips에서 start_idx부터 순서대로 count만큼 ImgFrame을 뽑아서 clip을 생성함.
        start_idx가 None이 아니면 start_idx사용, None이면 random값 사용.
        include_top : 첫번째 img는 전체 이미지이므로, 포함할지를 선택함.
          - gif생성시 처음과 끝 프레임 모두 전체 이미지여서 2장을 빼야함.
        '''

        idx_high = len(self.clips)
        idx_low = 0
        img_cnt = idx_high

        # gif생성시 처음과 끝 프레임 모두 전체 이미지여서 2장을 빼야함;;
        idx_low = 1
        idx_high -= 1
        img_cnt -= 2

        pick_cnt = min(img_cnt, count)

        # clip index를 두번 반복해서 roundQueue처럼 동작.
        idx_list = [ idx for idx in range(idx_low, idx_high)]
        idx_list.extend(idx_list)

        if start_idx is None:
            # random start idx
            start_idx = random.randrange(idx_low, idx_high)

        idx_list = idx_list[start_idx:start_idx + pick_cnt]

        if reverse:
            idx_list.reverse()

        temp_frames = [ self.clips[idx] for idx in idx_list ]

        frames = []
        stacked_frame = None

        for idx, frame in enumerate(temp_frames):

            if stacked_frame is None:
                stacked_frame = frame
            else:
                stacked_frame.append_channel(frame)

            img_frame = ImgFrame(stacked_frame.merged(), do_norm=False)

            if (idx + 1) % step == 0:
                frames.append(img_frame)
                stacked_frame = None

        if include_top:
            # 마지막에 전체 이미지 추가.
            frames.append(self.clips[0])

        return VideoClip(frames=frames)


    def stacked_frames_clip(self, step=1, included_label=False):
        '''
        clips에서 그려진 부분을 누적하여 frame생성.
        step==1인경우 0, 0-1, 0-2, 0-3, ... 0-n까지 stack한 frame들로 clip을 생성.
        step==2인 경우, 0, 0-2, 0-4, 0-6... 0-n까지 stack한 frame들로 clip생성.
        지금은 grayscale된 1채널만 동작함.
        '''
        if self.isEmpty():
            raise Exception("empty!!")

        vclip = VideoClip()
        stacked_frame = None

        if included_label:
            clips = self.clips[:-1]
        else:
            clips = self.clips

        for idx, frame in enumerate(clips):

            if stacked_frame is None:
                stacked_frame = frame
            else:
                stacked_frame.append_channel(frame)

            img_frame = ImgFrame(stacked_frame.merged(), do_norm=False)

            if (idx+1) % step == 0:
                vclip.append(img_frame)

        if included_label:
            vclip.append(self.clips[-1])

        return vclip


    def filled_frames_clip(self, val=0.0):
        '''
        clips에서 그려진 부분을 val값으로 채워서 frame생성.
        지금은 grayscale된 1채널만 동작함.
        '''
        if self.isEmpty():
            raise Exception("empty!!")

        for frame in self.clips:
            frame.fill_box(val=val)

        return self


    def minus_from(self, imgfrm, included_label=False, stack=False):
        '''
        imgfrm에서 clip의 각 frame을 뺀 frame들로
        clip을생성하여 return함.
        흰바탕에 검은색 0-1 norm된 이미지만 지원.
        '''
        if included_label:
            clips = self.clips[:-1]
        else:
            clips = self.clips

        vclip = VideoClip()
        stacked_frame = None
        for idx, frame in enumerate(clips):

            if stacked_frame is None:
                stacked_frame = frame
            else:
                stacked_frame.append_channel(frame)

            if stack:
                stacked_frame = ImgFrame(stacked_frame.merged(), do_norm=False)
                minus_frame = imgfrm.minus_frame(stacked_frame)
            else:
                minus_frame = imgfrm.minus_frame(frame)
            
            vclip.append(minus_frame)

        return vclip


    def augmentation(self, augmentator=None):
        ''' 
        augumentation을 받아서 각 프레임에 대해서 실행함.
        '''
        if augmentator is None:
            return self

        vclip = VideoClip()

        # augment를 한번 실행하여 replay용 데이터를 얻는다.
        data = augmentator(image=self.clips[0].arry)

        for img_frm in self.clips:
            # replay용 데이터를 이용하여 replay하여 동일한 augment를 실행한다.
            augmentated = ReplayCompose.replay(data['replay'], image=img_frm.arry)
            vclip.append(ImgFrame(img=augmentated["image"]))
        return vclip


    def threshold(self, threshold=0.5, low=0.0, high=1.0, inplace=True):
        ''' 
        threshold를 기준으로 low/high값으로 변환한 VideoClip을 return
        '''
        clips = [ clip.threshold(threshold, low, high, inplace=inplace) for clip in self.clips ]
        if inplace:
            return self
        else:
            return VideoClip(frames=clips)


    def split(self, dx=100, dy=100, included_top=False):
        '''
        외곽 박스 크기가 dx, dy보다 크면 split하여 
        동일한 크기의 frame 리스트로 생성하여 리턴함.
        included_top : top frame 포함 여부.
        '''
        clips = None
        top_frame = None
        if included_top:
            top_frame = self.clips[0]
            clips = self.clips[1:-1]
        else:
            clips = self.clips


        frames = []
        for img_frm in clips:
            frames.extend(img_frm.splitted_frames(dx, dy))

        if included_top:
            frames.append(top_frame)
            frames.insert(0, top_frame)

        return VideoClip(frames=frames)



    def find_adjacent_box(self, sorted_boxes, item_idx, margin=2, prior_direction=-1, max_direction=45):

        # sorted_boxes에서 item_idx부터 이후로 검색하면서 박스가 겹치는 clip index리스트를 구한다.
        # 포함이면 360, 나머지는 각도를 0-315도로 45도 간격으로 표시.
        base_box = sorted_boxes[item_idx]
        bsx, bsy, bex, bey = base_box['rect']
        bsx = max(0, bsx - margin)
        bsy = max(0, bsy - margin)
        bex = bex + margin
        bey = bey + margin

        adjacent_idxes = []
        for idx in range(item_idx+1, len(sorted_boxes)):
            box = sorted_boxes[idx]
            sx, sy, ex, ey = box['rect']

            if sx <= bex and ex >= bsx and sy <= bey and ey >= bsy:
                # box가 겹치는 경우중에...
                
                right, left, up, down = False, False, False, False
                direction = -1
                distance = 0
                if (sx >= bsx and ex <= bex and sy >= bsy and ey <= bey) or \
                    (sx <= bsx and ex >= bex and sy <= bsy and ey >= bey):
                    # base에 포함됨 또는 base를 포함함.
                    direction = 0
                    distance = 0
                else:
                    # 겹치는 방향을 45도 단위로 나타냄.
                    if ex > bex:
                        right = True
                    if sx < bsx:
                        left = True
                    if sy < bsy:
                        up = True
                    if ey < bey:
                        down = True

                    if left and right and up:
                        direction = 0
                    elif left and right and down:
                        direction = 180
                    elif up and down and right:
                        direction = 90
                    elif up and down and left:
                        direction = 270
                    elif left and up:
                        direction = 45
                    elif left and down:
                        direction = 135
                    elif right and down:
                        direction = 225
                    elif right and up:
                        direction = 315
                    
                    #이전 방향이 있는 경우만 방향을 고려함.
                    if prior_direction < 0:
                        distance = 0
                        # 하나를 설정하면 그방향을 이전 방향으로 설정하여 모두다 추가되는걸 막는다.
                        prior_direction = direction
                    else:
                        distance = min((360 + direction - prior_direction) % 360, (360 + prior_direction - direction) % 360)

                info = {'idx': idx, 'dir': direction, 'dist': distance}

                if distance <= max_direction:
                    adjacent_idxes.append(info)

                # if margin == 0:
                #     # box근처라면 비슷한 방향만 추가.
                #     # if distance <= 90:
                #     adjacent_idxes.append(info)
                # else:
                #     if distance <= max_direction:
                #         adjacent_idxes.append(info)
                #         break
                    
                # elif margin < 3:
                #     # box근처라면 비슷한 방향만 추가.
                #     if distance < 90:
                #         adjacent_idxes.append(info)
                # elif margin < 6:
                #     if distance < 45:
                #         adjacent_idxes.append(info)
                #         break
                # elif margin < 9:
                #     if distance < 45:
                #         adjacent_idxes.append(info)
                #         break
                # else:
                #     # box에서 멀리까지 획이 없다면 한개만 추가.
                #     # 안그러면 그리는 먼곳에서 이상한 점이 나타난다.
                #     adjacent_idxes.append(info)
                #     break

        adjacent_idxes = sorted(adjacent_idxes, key=lambda x:x['dist'], reverse=False)

        return adjacent_idxes


    def adjacent_clips(self, included_top=0, seq_type=0, margin=2):
        '''
        각 frame의 외곽 box를 구하여 인접한 frame순으로 정렬한다.
        box의 관계. - 포함, 겹침, 떨어짐.
        '''
        clips = None
        top_frame = None
        if 0 == included_top:
            clips = self.clips
        elif 1 == included_top:
            top_frame = self.clips[0]
            clips = self.clips[1:]
        elif 2 == included_top:
            top_frame = self.clips[0]
            clips = self.clips[1:-1]

        boxes = []
        centers = []
        # 각 frame에서 외곽 박스와 센터를 찾는다.
        for idx, imgframe in enumerate(clips):
            sx, sy, ex, ey = imgframe.out_box()
            if ex == 0 and ey == 0:
                continue
            boxes.append({'idx':idx, 'rect': (sx, sy, ex, ey)})
            cx = (sx + ex) // 2
            cy = (sy + ey) // 2
            centers.append((cx, cy))

        # 100x100 sliding box로 가장 많은 center가 있는 박스의 중심을 기준점으로 정함.
        # 얼굴이 가장 복잡하므로 이부분이 얼굴의 중심으로 볼 수 있음.
        box_w = 100
        box_h = 100
        step = 10
        imgh = self.shape[0]
        imgw = self.shape[1]
        max_center_cnt = 0
        max_center_px = 0
        max_center_py = 0
        for posy in range(0, imgh - box_h, step):
            for posx in range(0, imgw - box_w, step):
                center_cnt = 0
                for cx, cy in centers:
                    if cx >= posx and cx < posx + box_w:
                        if cy >= posy and cy < posy + box_h:
                            center_cnt += 1

                if center_cnt > max_center_cnt:
                    max_center_cnt = center_cnt
                    max_center_px = posx
                    max_center_py = posy

        base_x = (max_center_px+box_w) // 2
        base_y = (max_center_py+box_h) // 2

        distances = []
        for idx, center in enumerate(centers):
            cx, cy = center
            distance = (base_x - cx)**2 + (base_y - cy)**2
            distances.append( {'idx': idx, 'dist': distance} )

        sorted_dist = sorted(distances, key=lambda x:x['dist'], reverse=False)


        frames = []

        if seq_type == 0:
            # 중심에서 가까운 순서대로 정렬함.
            for distidx in sorted_dist:
                idx = distidx['idx']
                frames.append(clips[idx])
        elif seq_type == 1:
            # 센터에서 가까운 획부터 시작해서 인접 박스로 진행
            sorted_boxes = []
            
            # 처음 획 box를 찾는다.
            first_idx = sorted_dist[0]['idx']
            first_box = boxes[0]
            for box in boxes:
                boxidx = box['idx']
                if first_idx == boxidx:
                    first_box = box
                    break

            # 처음 획의 box 중심점을 기준으로 거리를 다시 계산한다.
            sx, sy, ex, ey = first_box['rect']
            base_x = (sx + ex) // 2
            base_y = (sy + ey) // 2
            distances = []
            for idx, center in enumerate(centers):
                cx, cy = center
                distance = (base_x - cx)**2 + (base_y - cy)**2
                distances.append( {'idx': idx, 'dist': distance} )

            sorted_dist = sorted(distances, key=lambda x:x['dist'], reverse=False)
            
            # 박스를 첫버째 박스와 가까운 순서로 정리.
            for distidx in sorted_dist:
                idx = distidx['idx']
                for box in boxes:
                    boxidx = box['idx']
                    if idx == boxidx:
                        sorted_boxes.append(box)

            item_idx = 0
            last_dir = 0

            while True:

                next_idx_list = []
                # margin을 늘려가며 겹치는 box를 찾는다.
                for margin_val in range(1, margin):
                    next_idx_list = self.find_adjacent_box(sorted_boxes, item_idx, 
                                            margin=margin_val, prior_direction=last_dir, max_direction=45)
                    if len(next_idx_list) > 0:
                        break

                if len(next_idx_list) == 0:
                    for margin_val in range(1, margin):
                        next_idx_list = self.find_adjacent_box(sorted_boxes, item_idx, 
                                                margin=margin_val, prior_direction=last_dir, max_direction=90)
                        if len(next_idx_list) > 0:
                            break

                if len(next_idx_list) == 0:
                    for margin_val in range(1, margin):
                        next_idx_list = self.find_adjacent_box(sorted_boxes, item_idx, 
                                            margin=margin_val, prior_direction=last_dir, max_direction=135)
                        if len(next_idx_list) > 0:
                            break

                if len(next_idx_list) == 0:
                    for margin_val in range(1, margin):
                        next_idx_list = self.find_adjacent_box(sorted_boxes, item_idx, 
                                            margin=margin_val, prior_direction=-1)
                        if len(next_idx_list) > 0:
                            break

                
                if len(next_idx_list) == 0:
                    # 못찾았다면 다음 걸로 넘어감.
                    last_dir = -1
                    item_idx += 1
                else:
                    # del로 index가 꼬이지 않도록 작은 index부터 실행함.
                    next_idx_list = sorted(next_idx_list, key=lambda x:x['idx'], reverse=False)

                    # 다음 방향은 가장 가까운 방향으로 .
                    last_dir = next_idx_list[0]['dir']
                    
                    # del때문에 가까운 곳부터 그려야 index가 꼬이지 않는다.
                    for next_idx_data in next_idx_list:
                        next_idx = next_idx_data['idx']
                        next_box = sorted_boxes[next_idx].copy()
                        del sorted_boxes[next_idx]
                        sorted_boxes.insert(item_idx + 1, next_box)

                    item_idx += len(next_idx_list)

                if item_idx >= len(sorted_boxes) - 1:
                    break

            for boxidx in sorted_boxes:
                idx = boxidx['idx']
                frames.append(clips[idx])
                
        if 1 == included_top:
            frames.insert(0, top_frame)
        elif 2 == included_top:
            frames.insert(0, top_frame)
            frames.append(top_frame)

        return VideoClip(frames=frames)


if __name__ == "__main__":
    """ 
    main함수.
    """

    import cv2
    import os
    import glob
    from utils.files import dir_path_change
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    IMG_LOAD_BASE_PATH = '/home/evergrin/iu/datas/imgs/raw_gif'
    IMG_SAVE_BASE_PATH = '/home/evergrin/iu/datas/imgs/data_set'
    IMG_PATH = '/home/evergrin/iu/datas/data_set'


    gif_list = glob.glob(os.path.join(IMG_PATH, "*.gif"))
    gif_file = gif_list[3]

    vclip = VideoClip()
    vclip.load_gif(gif_file, grayscale=True)

    newclip = vclip.split(dx=50,dy=50, included_top=False)

    newclip = newclip.adjacent_clips(included_top=0, seq_type=1)

    # boxes = newclip.adjacent_clips(included_top=0)
    # topfrm = newclip.clips[0]
    # topimg = topfrm.to_image()
    # draw = ImageDraw.Draw(topimg)  

    # for idxbox in boxes:
    #     box = idxbox['rect']
    #     sp = (box[0], box[1])
    #     ep = (box[2], box[3])
    #     cx = (box[0] + box[2]) // 2
    #     cy = (box[1] + box[3]) // 2
    #     sp = (cx-1, cy-1)
    #     ep = (cx+1, cy+1)
    #     draw.rectangle([sp, ep], outline ="black")

    # plt.imshow(topimg, cmap='gray')

    stackedclip = newclip.stacked_frames_clip()
    new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "gif")
    stackedclip.make_gif(new_file, fps=2)


    # newclip = vclip.all_clips(count=5, include_top=True)
    
    # newclip = vclip.split(dx=50,dy=50, include_top=False)

    # new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "gif")
    # newclip.make_gif(new_file)

    # stacked_clip = newclip.stacked_frames_clip(step=1)
    # label_frm = newclip.clips[-1]
    # new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "png")
    # label_frm.to_image(save_file=new_file)
    
    # minclip = newclip.minus_from(label_frm, stack=True)
    
    # for cc in minclip.clips:
    #     new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "png")
    #     cc.to_image(save_file=new_file)
        
    # new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "gif")
    # minclip.make_gif(new_file)
        
    # img = imgfrm.to_image(save_file=new_file)
    # plt.imshow(img, cmap='gray')
    print('aaa')

    # arrys = []
    # arry1 = vclip.to_array()
    # arry2 = vclip.to_array()
    # print(arry1.shape, arry2.shape)
    
    # arrys.append(arry1)
    # arrys.append(arry2)
    
    # arrys = np.stack(arrys)
    # print(arrys.shape)

    # for i in range(1):
    #     vclip = VideoClip()
    #     vclip.load_gif(gif_file, grayscale=True)

    #     # newclip = vclip.random_clips()
    #     newclip = vclip.sequential_clips(count=10, reverse=True)
    #     varry = newclip.to_array()
    #     print(varry.shape)

    #     stacked_clip = newclip.stacked_frames_clip(step=2)
    #     resized_clip = stacked_clip.resize(64, 32, inplace=False)

    #     new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "gif")
    #     stacked_clip.make_gif(new_file)

    #     new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "gif")
    #     resized_clip.make_gif(new_file)

    #     stacked_clip.resize(12, 12, inplace=True)
    #     new_file = dir_path_change(gif_file, IMG_SAVE_BASE_PATH, "gif")
    #     stacked_clip.make_gif(new_file)

        