# ------------------------------------------------------------------------


"""
数据加载器声明：

本数据加载器的代码仅作为参考示例使用。由于项目中涉及到的私有库或特定数据格式，用户需根据自身的实际需求对代码进行修改和调整。

具体来说，如果您的数据存储在如 LMDB 等数据库中，请确保您了解并使用相应的库来加载和处理数据。本代码不提供对这些私有库的完整支持，用户需根据私有库的接口和功能自行实现必要的调整。

使用本数据加载器代码时，请注意：
1. 数据格式和路径：请根据您的实际数据结构和存储方式修改相关代码。
2. 库的依赖：如果使用LMDB或其他特定库，请确保正确导入并配置这些库。
3. 性能优化：根据数据量和硬件环境，可能需要进行性能上的优化。



"""

from turtle import forward
from torch.utils import data as data

from basicsr.utils.file_client import BaseStorageBackend
from basicsr.data.paired_image_dataset import PairedImageDataset
# from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils import imfrombytes
from basicsr.data.ffhq_dataset import FFHQDataset
import random
import time
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
# from basicsr.utils.registry import DATASET_REGISTRY
import os
import json
import cv2
import refile
import random
import numpy as np
from basicsr.data.degradations import random_add_gaussian_noise, random_mixed_kernels, random_add_jpg_compression
import torch
from basicsr.utils.matlab_functions import imresize
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
import math
import ffmpeg
import sys
import os
current_working_dir = os.getcwd()
sys.path.append(current_working_dir+'/')



from basicsr.utils.registry import DATASET_REGISTRY


def create_opencv_image_from_stringio(img_stream, cv2_img_flag=-1):
    img_array = np.asarray(bytearray(img_stream), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)[:,:,::-1]

def convert_opencv_image_to_string_io(image):
    img_encode = cv2.imencode('.png', image[:,:,::-1])[1]
    img_array = np.array(img_encode)
    img_str = img_array.tostring()
    return img_str




@DATASET_REGISTRY.register()
class VFHQFULLNTMEBASIC(data.Dataset):
    # 参数说明
    # r: 半径，每个时序多少帧（2*r+1)，当r设置为0时就是单帧
    # is_aligned: true or false 是否启用对齐，例如此时r=3，序列长度为7。读取第4帧的人脸5点参数，并根据第四帧人脸参数对7帧进行统一warp
    # dictpath: 人脸参数记录文件 默认位置为pad后的记录文件 's3://kepengxu/vfhqfull/pad_5landmark_matrix.txt'
    # degradation: 退化模式：可选 'blr' 和 'lr' 分别代表了盲退化和仅仅下采样
    # phase： train 或者 test
    # root: 
    def __init__(self, opt):
        super(VFHQFULLNTMEBASIC, self).__init__()
        data.Dataset.__init__(self)
        self.length = 2 * opt['r'] + 1
        self.is_aligned  = opt['is_aligned']
        self.dictpath = opt['dictpath']
        self.save_freq = 100
        self.dictlist = {}
        import time
        starttime = time.time()
        with refile.smart_open(self.dictpath,'r') as f:
            for lin in f:
                line = lin[:-1]
                key,v = line.split('||||,')
                vl = v.split(',')
                five_points = np.reshape(np.array([float(tv) for tv in vl[:10]]),(5,2))
                matrix = np.reshape(np.array([float(tv) for tv in vl[10:]]),(2,3))
                if key in self.dictlist.keys():
                    print('error')
                    sys.exit()
                self.dictlist[key] = {
                    '5landmark':five_points,
                    'matrix':matrix
                }
        

        self.degradation = opt['degradation']
        self.phase = opt['phase']
        self.mean = torch.Tensor(opt['mean']).view(1,3,1,1) if 'mean' in opt else None
        self.std = torch.Tensor(opt['std']).view(1,3,1,1) if 'std' in opt else None
        self.dirlist = []
        self.path2id = {}
        # need lmdb or other client like basicsr.data...
        self.file_client = ~
        
        
        if 'test' in self.phase:
            pl = ['testgt.filelist','testlr_blind.filelist']
        else:
            pl = ['traingt.filelist','trainlr_blind.filelist']
        for filep in pl[0:1]:
            with refile.smart_open(os.path.join(opt['root'],filep), 'r') as f:
                for lin in f:
                    line = lin[:-1]
                    path, id512 = line.split('||||||')
                    tdir = path.split('/')[1]
                    if (not tdir in self.dirlist) and filep == pl[0]:
                        self.dirlist.append(tdir)
                    self.path2id[path] = id512
        ttt = 0
        tl = []
        for key in self.path2id.keys():
            if key in self.dictlist.keys():
                continue
            else:
                ttt +=1
                tl.append(key)

        for filep in pl[1:2]:
            with refile.smart_open(os.path.join(opt['root'],filep), 'r') as f:
                for lin in f:
                    line = lin[:-1]
                    path, id512 = line.split('||||||')
                    tdir = path.split('/')[1]
                    if (not tdir in self.dirlist) and filep == pl[0]:
                        self.dirlist.append(tdir)
                    self.path2id[path] = id512
        # print()
        self.pickledump()
        print('inited dataset',time.time()-starttime)

    def pickledump(self,):
        import pickle
        import refile
        f = refile.smart_open('dataroot'+self.phase+'_'+'dictlist.pkl','wb')
        pickle.dump(self.dictlist, f)
        f.close()

        f = refile.smart_open('dataroot'+self.phase+'_'+'path2id.pkl','wb')
        pickle.dump(self.path2id, f)
        f.close()
    
        f = refile.smart_open('dataroot'+self.phase+'_'+'dirlist.pkl','wb')
        pickle.dump(self.dirlist, f)
        f.close()

    def pickleload(self):
        import pickle
        import refile
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dictlist.pkl','rb')
        self.dictlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'path2id.pkl','rb')
        self.path2id = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dirlist.pkl','rb')
        self.dirlist = pickle.load(f)
        f.close()


    def cal_warp_affine2d(self, landmark, scale=1.0):
        face_template=np.array([ 
                                [192.98138, 239.94708],
                                [318.90277, 240.1936],
                                [256.63416, 314.01935],
                                [201.26117, 371.41043],
                                [313.08905, 371.15118]
                                ])/scale
        landmarkt = np.array( landmark/scale)
        affine_matrix = cv2.estimateAffinePartial2D(landmarkt, face_template, method=cv2.LMEDS)[0]
        return affine_matrix

    def warp(self, input_img, affine_matrix, shape=(512,512),scale=1):
        shape = (int(shape[0]/scale),int(shape[1]/scale))
        cropped_face = cv2.warpAffine(
                input_img, affine_matrix, shape, borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray 
        return cropped_face


    def lr(self, inputs):
        lr_images = [ ]
        for v in inputs:
            t = imresize(np.array(v,np.float32)/255.0, 0.25)
            lr_images.append(t)
        return lr_images

    def __getitem__(self, index):
        start = random.randint(1, 20-self.length+1)
        end = start + self.length
        dirt = self.dirlist[index]
        input = []
        d = {}
        warppath = os.path.join('GT', dirt, '{:0>8d}.png'.format((start + end)//2))
        landmark5 = self.dictlist[warppath]['5landmark']
        gtmatrix = self.cal_warp_affine2d(landmark=landmark5)
        for i in range(start, end):
            path = os.path.join('GT',dirt, '{:0>8d}.png'.format(i))
            # bytestring = self.nw.get()
            bytestring = self.file_client.get(self.path2id[path])
            tinput = create_opencv_image_from_stringio(bytestring) # rgb
            if self.is_aligned:
                tinput = self.warp(tinput, gtmatrix)
            input.append(tinput)
        
        if self.degradation == 'lr':
            lrimages = self.lr(input)
            lrimages = np.array(lrimages, np.float32) # [t,h,w,c]
            lrimages = torch.from_numpy(lrimages).permute(0,3,1,2)
            if self.mean is not None or self.std is not None:
                lrimages = (lrimages - self.mean)/self.std
            d['lq'] = lrimages
            
        elif self.degradation == 'blr':
            lqinput = []
            d = {}
            for i in range(start, end):
                pathlq = os.path.join('LR_Blind', dirt, '{:0>8d}.png'.format(i))
                # bytestring = self.nw.get()
                bytestring = self.file_client.get(self.path2id[pathlq])
                tlqinput = create_opencv_image_from_stringio(bytestring) # rgb
                if self.is_aligned:
                    lqmatrix = self.cal_warp_affine2d(landmark=landmark5, scale=4.0)
                    tlqinput = self.warp(tlqinput, lqmatrix, shape=(512,512), scale=4.0 )
                    
                    # tlqinput = cv2.resize(tlqinput, (512,512))
                    
                
                lqinput.append(tlqinput)
            lqinput = np.array(np.array(lqinput)/255.0, np.float32) # [t,h,w,c]
            lqinput = torch.from_numpy(lqinput).permute(0,3,1,2)
            if self.mean is not None or self.std is not None:
                lqinput = (lqinput - self.mean)/self.std
            d['lq'] = lqinput
                
        
        input = np.array(input,np.float32)/255.0# [t,h,w,c]
        input = torch.from_numpy(input).permute(0,3,1,2)
        if self.mean is not None or self.std is not None:
            input = (input - self.mean)/self.std
        d['gt'] = input
        
        # mse = torch.mean((input-lrimages)*(input - lrimages))
        
        if self.length == 1:
            return {
                'lq': d['lq'][0],
                'gt': d['gt'][0]
                }
        
        return d
    
    def __len__(self,):
        return len(self.dirlist)
                

@DATASET_REGISTRY.register()
class VFHQFULLNTMEBASICV2TRAIN(data.Dataset):
    # 参数说明
    # r: 半径，每个时序多少帧（2*r+1)，当r设置为0时就是单帧
    # is_aligned: true or false 是否启用对齐，例如此时r=3，序列长度为7。读取第4帧的人脸5点参数，并根据第四帧人脸参数对7帧进行统一warp
    # dictpath: 人脸参数记录文件 默认位置为pad后的记录文件 's3://kepengxu/vfhqfull/pad_5landmark_matrix.txt'
    # degradation: 退化模式：可选 'blr' 和 'lr' 分别代表了盲退化和仅仅下采样
    # phase： train 或者 test
    # root: 'dataroot'
    def __init__(self, opt):
        super(VFHQFULLNTMEBASICV2TRAIN, self).__init__()
        print(opt)
        data.Dataset.__init__(self)
        self.length = 2 * opt['r'] + 1
        self.is_aligned  = opt['is_aligned']
        self.dictpath = opt['dictpath']
        self.save_freq = 100
        self.dictlist = {}
        self.opt = opt
        import time
        starttime = time.time()
        # with refile.smart_open(self.dictpath,'r') as f:
        #     for lin in f:
        #         line = lin[:-1]
        #         key,v = line.split('||||,')
        #         vl = v.split(',')
        #         five_points = np.reshape(np.array([float(tv) for tv in vl[:10]]),(5,2))
        #         matrix = np.reshape(np.array([float(tv) for tv in vl[10:]]),(2,3))
        #         if key in self.dictlist.keys():
        #             print('error')
        #             sys.exit()
        #         self.dictlist[key] = {
        #             '5landmark':five_points,
        #             'matrix':matrix
        #         }
        

        self.degradation = opt['degradation']
        self.phase = opt['phase1']
        self.mean = torch.Tensor(opt['mean']).view(1,3,1,1) if 'mean' in opt else None
        self.std = torch.Tensor(opt['std']).view(1,3,1,1) if 'std' in opt else None
        self.dirlist = []
        self.path2id = {}
        
        # !!! replace as lmdb or other client
        self.file_client = 
        
        
        # if 'test' in self.phase:
        #     pl = ['testgt.filelist','testlr_blind.filelist']
        # else:
        #     pl = ['traingt.filelist','trainlr_blind.filelist']
        # for filep in pl[0:1]:
        #     with refile.smart_open(os.path.join(opt['root'],filep), 'r') as f:
        #         for lin in f:
        #             line = lin[:-1]
        #             path, id512 = line.split('||||||')
        #             tdir = path.split('/')[1]
        #             if (not tdir in self.dirlist) and filep == pl[0]:
        #                 self.dirlist.append(tdir)
        #             self.path2id[path] = id512
        # ttt = 0
        # tl = []
        # for key in self.path2id.keys():
        #     if key in self.dictlist.keys():
        #         continue
        #     else:
        #         ttt +=1
        #         tl.append(key)

        # for filep in pl[1:2]:
        #     with refile.smart_open(os.path.join(opt['root'],filep), 'r') as f:
        #         for lin in f:
        #             line = lin[:-1]
        #             path, id512 = line.split('||||||')
        #             tdir = path.split('/')[1]
        #             if (not tdir in self.dirlist) and filep == pl[0]:
        #                 self.dirlist.append(tdir)
        #             self.path2id[path] = id512
        # print()
        self.pickleload()
        print('inited dataset',time.time()-starttime)
        
    def pickleload(self):
        import pickle
        import refile
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dictlist.pkl','rb')
        self.dictlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'path2id.pkl','rb')
        self.path2id = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dirlist.pkl','rb')
        self.dirlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'numdict.pkl','rb')
        self.numdict = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'list.pkl','rb')
        self.indexs = pickle.load(f)
        f.close()
        
        # self.indexs = []
        

    def cal_warp_affine2d(self, landmark, scale=1.0):
        face_template=np.array([ 
                                [192.98138, 239.94708],
                                [318.90277, 240.1936],
                                [256.63416, 314.01935],
                                [201.26117, 371.41043],
                                [313.08905, 371.15118]
                                ])/scale
        landmarkt = np.array( landmark/scale)
        affine_matrix = cv2.estimateAffinePartial2D(landmarkt, face_template, method=cv2.LMEDS)[0]
        return affine_matrix

    def warp(self, input_img, affine_matrix, shape=(512,512),scale=1):
        shape = (int(shape[0]/scale),int(shape[1]/scale))
        cropped_face = cv2.warpAffine(
                input_img, affine_matrix, shape, borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray 
        return cropped_face


    def lr(self, inputs):
        lr_images = [ ]
        for v in inputs:
            t = imresize(np.array(v,np.float32)/255.0, 0.25)
            lr_images.append(t)
        return lr_images

    def __getitem__(self, index):
        # dirt = self.dirlist[index]
        while True:
            try:
                dirt,center = self.indexs[index]
                start = center-self.length//2
                end = start + self.length
                
                
                input = []
                d = {}
                warppath = os.path.join('GT', dirt, '{:0>8d}.png'.format((start + end)//2))
                landmark5 = self.dictlist[warppath]['5landmark']
                gtmatrix = self.cal_warp_affine2d(landmark=landmark5)
                for i in range(start, end):
                    if i<1:
                        i=1
                    if i>(self.numdict[dirt]-2):
                        i=(self.numdict[dirt]-2)
                    path = os.path.join('GT',dirt, '{:0>8d}.png'.format(i))
                    # bytestring = self.nw.get()
                    bytestring = self.file_client.get(self.path2id[path])
                    tinput = create_opencv_image_from_stringio(bytestring) # rgb
                    if self.is_aligned:
                        tinput = self.warp(tinput, gtmatrix)
                    input.append(tinput)
                
                if self.degradation == 'lr':
                    lrimages = self.lr(input)
                    lrimages = np.array(lrimages, np.float32) # [t,h,w,c]
                    lrimages = torch.from_numpy(lrimages).permute(0,3,1,2)
                    if self.mean is not None or self.std is not None:
                        lrimages = (lrimages - self.mean)/self.std
                    d['lq'] = lrimages
                    
                elif self.degradation == 'blr':
                    lqinput = []
                    d = {}
                    for i in range(start, end):
                        if i<1:
                            i=1
                        if i>(self.numdict[dirt]-2):
                            i=(self.numdict[dirt]-2)
                        pathlq = os.path.join('LR_Blind', dirt, '{:0>8d}.png'.format(i))
                        # bytestring = self.nw.get()
                        bytestring = self.file_client.get(self.path2id[pathlq])
                        tlqinput = create_opencv_image_from_stringio(bytestring) # rgb
                        if self.is_aligned:
                            lqmatrix = self.cal_warp_affine2d(landmark=landmark5, scale=4.0)
                            tlqinput = self.warp(tlqinput, lqmatrix, shape=(512,512), scale=4.0 )
                            
                            # tlqinput = cv2.resize(tlqinput, (512,512))
                            
                        
                        lqinput.append(tlqinput)
                    lqinput = np.array(np.array(lqinput)/255.0, np.float32) # [t,h,w,c]
                    lqinput = torch.from_numpy(lqinput).permute(0,3,1,2)
                    if self.mean is not None or self.std is not None:
                        lqinput = (lqinput - self.mean)/self.std
                    d['lq'] = lqinput
                        
                
                input = np.array(input,np.float32)/255.0# [t,h,w,c]
                input = torch.from_numpy(input).permute(0,3,1,2)
                if self.mean is not None or self.std is not None:
                    input = (input - self.mean)/self.std
                d['gt'] = input
                
                # mse = torch.mean((input-lrimages)*(input - lrimages))
                d['path'] = [dirt,center]
                d['gt_path'] = warppath
                if self.length == 1:
                    return {
                        'lq': d['lq'][0],
                        'gt': d['gt'][0],
                        'path': [dirt,center],
                        'gt_path':warppath
                        }
                
                return d
            except:
                print('error index',index)
                index = random.randint(2,self.__len__()-2)
    
    def __len__(self,):
        return len(self.indexs)
                


@DATASET_REGISTRY.register()
class VFHQFULLNTMEBASICV2TEST(data.Dataset):
    # 参数说明
    # r: 半径，每个时序多少帧（2*r+1)，当r设置为0时就是单帧
    # is_aligned: true or false 是否启用对齐，例如此时r=3，序列长度为7。读取第4帧的人脸5点参数，并根据第四帧人脸参数对7帧进行统一warp
    # dictpath: 人脸参数记录文件 默认位置为pad后的记录文件 's3://kepengxu/vfhqfull/pad_5landmark_matrix.txt'
    # degradation: 退化模式：可选 'blr' 和 'lr' 分别代表了盲退化和仅仅下采样
    # phase： train 或者 test
    # root: 'dataroot'
    def __init__(self, opt):
        super(VFHQFULLNTMEBASICV2TEST, self).__init__()
        print(opt)
        data.Dataset.__init__(self)
        self.length = 2 * opt['r'] + 1
        self.is_aligned  = opt['is_aligned']
        self.dictpath = opt['dictpath']
        self.save_freq = 100
        self.dictlist = {}
        self.opt = opt
        import time
        starttime = time.time()
        # with refile.smart_open(self.dictpath,'r') as f:
        #     for lin in f:
        #         line = lin[:-1]
        #         key,v = line.split('||||,')
        #         vl = v.split(',')
        #         five_points = np.reshape(np.array([float(tv) for tv in vl[:10]]),(5,2))
        #         matrix = np.reshape(np.array([float(tv) for tv in vl[10:]]),(2,3))
        #         if key in self.dictlist.keys():
        #             print('error')
        #             sys.exit()
        #         self.dictlist[key] = {
        #             '5landmark':five_points,
        #             'matrix':matrix
        #         }
        

        self.degradation = opt['degradation']
        self.phase = opt['phase1']
        self.mean = torch.Tensor(opt['mean']).view(1,3,1,1) if 'mean' in opt else None
        self.std = torch.Tensor(opt['std']).view(1,3,1,1) if 'std' in opt else None
        self.dirlist = []
        self.path2id = {}
        self.file_client = ~
        
        
        # if 'test' in self.phase:
        #     pl = ['testgt.filelist','testlr_blind.filelist']
        # else:
        #     pl = ['traingt.filelist','trainlr_blind.filelist']
        # for filep in pl[0:1]:
        #     with refile.smart_open(os.path.join(opt['root'],filep), 'r') as f:
        #         for lin in f:
        #             line = lin[:-1]
        #             path, id512 = line.split('||||||')
        #             tdir = path.split('/')[1]
        #             if (not tdir in self.dirlist) and filep == pl[0]:
        #                 self.dirlist.append(tdir)
        #             self.path2id[path] = id512
        # ttt = 0
        # tl = []
        # for key in self.path2id.keys():
        #     if key in self.dictlist.keys():
        #         continue
        #     else:
        #         ttt +=1
        #         tl.append(key)

        # for filep in pl[1:2]:
        #     with refile.smart_open(os.path.join(opt['root'],filep), 'r') as f:
        #         for lin in f:
        #             line = lin[:-1]
        #             path, id512 = line.split('||||||')
        #             tdir = path.split('/')[1]
        #             if (not tdir in self.dirlist) and filep == pl[0]:
        #                 self.dirlist.append(tdir)
        #             self.path2id[path] = id512
        # print()
        self.pickleload()
        print('inited dataset',time.time()-starttime)
        
    def pickleload(self):
        import pickle
        import refile
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dictlist.pkl','rb')
        self.dictlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'path2id.pkl','rb')
        self.path2id = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dirlist.pkl','rb')
        self.dirlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'numdict.pkl','rb')
        self.numdict = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'list.pkl','rb')
        self.indexs = pickle.load(f)
        f.close()
        
        # self.indexs = []
        

    def cal_warp_affine2d(self, landmark, scale=1.0):
        face_template=np.array([ 
                                [192.98138, 239.94708],
                                [318.90277, 240.1936],
                                [256.63416, 314.01935],
                                [201.26117, 371.41043],
                                [313.08905, 371.15118]
                                ])/scale
        landmarkt = np.array( landmark/scale)
        affine_matrix = cv2.estimateAffinePartial2D(landmarkt, face_template, method=cv2.LMEDS)[0]
        return affine_matrix

    def warp(self, input_img, affine_matrix, shape=(512,512),scale=1):
        shape = (int(shape[0]/scale),int(shape[1]/scale))
        cropped_face = cv2.warpAffine(
                input_img, affine_matrix, shape, borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray 
        return cropped_face


    def lr(self, inputs):
        lr_images = [ ]
        for v in inputs:
            t = imresize(np.array(v,np.float32)/255.0, 0.25)
            lr_images.append(t)
        return lr_images

    def __getitem__(self, index):
        # dirt = self.dirlist[index]
        # index = int(index*50+25)

        dirt,center = self.indexs[index]
        start = center-self.length//2
        end = start + self.length
        
        
        input = []
        d = {}
        warppath = os.path.join('GT', dirt, '{:0>8d}.png'.format((start + end)//2))
        landmark5 = self.dictlist[warppath]['5landmark']
        gtmatrix = self.cal_warp_affine2d(landmark=landmark5)
        for i in range(start, end):
            if i<1:
                i=1
            if i>(self.numdict[dirt]-2):
                i=(self.numdict[dirt]-2)
            path = os.path.join('GT',dirt, '{:0>8d}.png'.format(i))
            # bytestring = self.nw.get()
            bytestring = self.file_client.get(self.path2id[path])
            tinput = create_opencv_image_from_stringio(bytestring) # rgb
            if self.is_aligned:
                tinput = self.warp(tinput, gtmatrix)
            input.append(tinput)
        
        if self.degradation == 'lr':
            lrimages = self.lr(input)
            lrimages = np.array(lrimages, np.float32) # [t,h,w,c]
            lrimages = torch.from_numpy(lrimages).permute(0,3,1,2)
            if self.mean is not None or self.std is not None:
                lrimages = (lrimages - self.mean)/self.std
            d['lq'] = lrimages
            
        elif self.degradation == 'blr':
            lqinput = []
            d = {}
            for i in range(start, end):
                if i<1:
                    i=1
                if i>(self.numdict[dirt]-2):
                    i=(self.numdict[dirt]-2)
                pathlq = os.path.join('LR_Blind', dirt, '{:0>8d}.png'.format(i))
                # bytestring = self.nw.get()
                bytestring = self.file_client.get(self.path2id[pathlq])
                tlqinput = create_opencv_image_from_stringio(bytestring) # rgb
                if self.is_aligned:
                    lqmatrix = self.cal_warp_affine2d(landmark=landmark5, scale=4.0)
                    tlqinput = self.warp(tlqinput, lqmatrix, shape=(512,512), scale=4.0 )
                    
                    # tlqinput = cv2.resize(tlqinput, (512,512))
                    
                
                lqinput.append(tlqinput)
            lqinput = np.array(np.array(lqinput)/255.0, np.float32) # [t,h,w,c]
            lqinput = torch.from_numpy(lqinput).permute(0,3,1,2)
            if self.mean is not None or self.std is not None:
                lqinput = (lqinput - self.mean)/self.std
            d['lq'] = lqinput
                
        
        input = np.array(input,np.float32)/255.0# [t,h,w,c]
        input = torch.from_numpy(input).permute(0,3,1,2)
        if self.mean is not None or self.std is not None:
            input = (input - self.mean)/self.std
        d['gt'] = input
        
        # mse = torch.mean((input-lrimages)*(input - lrimages))
        d['path'] = [dirt,center]
        d['gt_path'] = warppath
        if self.length == 1:
            return {
                'lq': d['lq'][0],
                'gt': d['gt'][0],
                'path': [dirt,center],
                'gt_path':warppath
                }
        
        return d

    
    def __len__(self,):
        return len(self.indexs)
                


@DATASET_REGISTRY.register()
class VFHQFULLNTMEBASICV2TRAINUP(data.Dataset):
    # 参数说明
    # r: 半径，每个时序多少帧（2*r+1)，当r设置为0时就是单帧
    # is_aligned: true or false 是否启用对齐，例如此时r=3，序列长度为7。读取第4帧的人脸5点参数，并根据第四帧人脸参数对7帧进行统一warp
    # dictpath: 人脸参数记录文件 默认位置为pad后的记录文件 's3://kepengxu/vfhqfull/pad_5landmark_matrix.txt'
    # degradation: 退化模式：可选 'blr' 和 'lr' 分别代表了盲退化和仅仅下采样
    # phase： train 或者 test
    # root: 'dataroot'
    def __init__(self, opt):
        super(VFHQFULLNTMEBASICV2TRAINUP, self).__init__()
        print(opt)
        data.Dataset.__init__(self)
        self.length = 2 * opt['r'] + 1
        self.is_aligned  = opt['is_aligned']
        self.dictpath = opt['dictpath']
        self.dictlist = {}
        self.save_freq = 100
        self.opt = opt
        import time
        starttime = time.time()

        self.degradation = opt['degradation']
        self.phase = opt['phase1']
        self.mean = torch.Tensor(opt['mean']).view(1,3,1,1) if 'mean' in opt else None
        self.std = torch.Tensor(opt['std']).view(1,3,1,1) if 'std' in opt else None
        self.dirlist = []
        self.path2id = {}
        
        self.pickleload()
        print('inited dataset',time.time()-starttime)
        
    def pickleload(self):
        import pickle
        import refile
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dictlist.pkl','rb')
        self.dictlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'path2id.pkl','rb')
        self.path2id = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dirlist.pkl','rb')
        self.dirlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'numdict.pkl','rb')
        self.numdict = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'list.pkl','rb')
        self.indexs = pickle.load(f)
        f.close()
        
        # self.indexs = []
        

    def cal_warp_affine2d(self, landmark, scale=1.0):
        face_template=np.array([ 
                                [192.98138, 239.94708],
                                [318.90277, 240.1936],
                                [256.63416, 314.01935],
                                [201.26117, 371.41043],
                                [313.08905, 371.15118]
                                ])/scale
        landmarkt = np.array( landmark/scale)
        affine_matrix = cv2.estimateAffinePartial2D(landmarkt, face_template, method=cv2.LMEDS)[0]
        return affine_matrix

    def warp(self, input_img, affine_matrix, shape=(512,512),scale=1):
        shape = (int(shape[0]/scale),int(shape[1]/scale))
        cropped_face = cv2.warpAffine(
                input_img, affine_matrix, shape, borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray 
        return cropped_face


    def lr(self, inputs):
        lr_images = [ ]
        for v in inputs:
            t = imresize(np.array(v,np.float32)/255.0, 0.25)
            lr_images.append(t)
        return lr_images

    def __getitem__(self, index):
        # dirt = self.dirlist[index]
        while True:
            try:
                dirt,center = self.indexs[index]
                start = center-self.length//2
                end = start + self.length
                
                
                input = []
                d = {}
                warppath = os.path.join('GT', dirt, '{:0>8d}.png'.format((start + end)//2))
                landmark5 = self.dictlist[warppath]['5landmark']
                gtmatrix = self.cal_warp_affine2d(landmark=landmark5)
                for i in range(start, end):
                    if i<1:
                        i=1
                    if i>(self.numdict[dirt]-2):
                        i=(self.numdict[dirt]-2)
                    path = os.path.join('GT',dirt, '{:0>8d}.png'.format(i))
                    # bytestring = self.nw.get()
                    bytestring = self.file_client.get(self.path2id[path])
                    tinput = create_opencv_image_from_stringio(bytestring) # rgb
                    if self.is_aligned:
                        tinput = self.warp(tinput, gtmatrix)
                    input.append(tinput)
                
                if self.degradation == 'lr':
                    lrimages = self.lr(input)
                    lrimages = np.array(lrimages, np.float32) # [t,h,w,c]
                    lrimages = torch.from_numpy(lrimages).permute(0,3,1,2)
                    if self.mean is not None or self.std is not None:
                        lrimages = (lrimages - self.mean)/self.std
                    d['llq'] = lrimages
                    d['lq'] = torch.nn.functional.interpolate(lrimages,(512,512),mode = 'bilinear',align_corners=True)
                    
                elif self.degradation == 'blr':
                    lqinput = []
                    d = {}
                    for i in range(start, end):
                        if i<1:
                            i=1
                        if i>(self.numdict[dirt]-2):
                            i=(self.numdict[dirt]-2)
                        pathlq = os.path.join('LR_Blind', dirt, '{:0>8d}.png'.format(i))
                        # bytestring = self.nw.get()
                        bytestring = self.file_client.get(self.path2id[pathlq])
                        tlqinput = create_opencv_image_from_stringio(bytestring) # rgb
                        if self.is_aligned:
                            lqmatrix = self.cal_warp_affine2d(landmark=landmark5, scale=4.0)
                            tlqinput = self.warp(tlqinput, lqmatrix, shape=(512,512), scale=4.0 )
                            
                            # tlqinput = cv2.resize(tlqinput, (512,512))
                            
                        
                        lqinput.append(tlqinput)
                    lqinput = np.array(np.array(lqinput)/255.0, np.float32) # [t,h,w,c]
                    lqinput = torch.from_numpy(lqinput).permute(0,3,1,2)
                    if self.mean is not None or self.std is not None:
                        lqinput = (lqinput - self.mean)/self.std
                    d['llq'] = lqinput
                    d['lq'] = torch.nn.functional.interpolate(lqinput,(512,512),mode = 'bilinear',align_corners=True)
                        
                # print('mean',self.mean,self.std)
                # print(self.opt)
                input = np.array(input,np.float32)/255.0# [t,h,w,c]
                input = torch.from_numpy(input).permute(0,3,1,2)
                if self.mean is not None or self.std is not None:
                    input = (input - self.mean)/self.std
                d['gt'] = input
                
                # mse = torch.mean((input-lrimages)*(input - lrimages))
                d['path'] = [dirt,center]
                d['gt_path'] = warppath
                if self.length == 1:
                    return {
                        'lq': d['lq'][0],
                        'gt': d['gt'][0],
                        'path': [dirt,center],
                        'gt_path':warppath
                        }
                
                return d
            except:
                print('error index',index)
                index = random.randint(2,self.__len__()-2)
    
    def __len__(self,):
        return len(self.indexs)
                


@DATASET_REGISTRY.register()
class VFHQFULLNTMEBASICV2TESTUP(data.Dataset):
    # 参数说明
    # r: 半径，每个时序多少帧（2*r+1)，当r设置为0时就是单帧
    # is_aligned: true or false 是否启用对齐，例如此时r=3，序列长度为7。读取第4帧的人脸5点参数，并根据第四帧人脸参数对7帧进行统一warp
    # dictpath: 人脸参数记录文件 默认位置为pad后的记录文件 's3://kepengxu/vfhqfull/pad_5landmark_matrix.txt'
    # degradation: 退化模式：可选 'blr' 和 'lr' 分别代表了盲退化和仅仅下采样
    # phase： train 或者 test
    # root: 'dataroot'
    def __init__(self, opt):
        super(VFHQFULLNTMEBASICV2TESTUP, self).__init__()
        print(opt)
        data.Dataset.__init__(self)
        self.length = 2 * opt['r'] + 1
        self.is_aligned  = opt['is_aligned']
        self.dictpath = opt['dictpath']
        self.dictlist = {}
        self.save_freq = 100
        self.opt = opt
        import time
        starttime = time.time()

        self.degradation = opt['degradation']
        self.phase = opt['phase1']
        self.mean = torch.Tensor(opt['mean']).view(1,3,1,1) if 'mean' in opt else None
        self.std = torch.Tensor(opt['std']).view(1,3,1,1) if 'std' in opt else None
        self.dirlist = []
        self.path2id = {}
        self.inter_space = opt.get('inter_space', 1)
        

        self.pickleload()
        print('inited dataset',time.time()-starttime)
        
    def pickleload(self):
        import pickle
        import refile
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dictlist.pkl','rb')
        self.dictlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'path2id.pkl','rb')
        self.path2id = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dirlist.pkl','rb')
        self.dirlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'numdict.pkl','rb')
        self.numdict = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'list.pkl','rb')
        self.indexs = pickle.load(f)
        f.close()
        
        # self.indexs = []
        

    def cal_warp_affine2d(self, landmark, scale=1.0):
        face_template=np.array([ 
                                [192.98138, 239.94708],
                                [318.90277, 240.1936],
                                [256.63416, 314.01935],
                                [201.26117, 371.41043],
                                [313.08905, 371.15118]
                                ])/scale
        landmarkt = np.array( landmark/scale)
        affine_matrix = cv2.estimateAffinePartial2D(landmarkt, face_template, method=cv2.LMEDS)[0]
        return affine_matrix

    def warp(self, input_img, affine_matrix, shape=(512,512),scale=1):
        shape = (int(shape[0]/scale),int(shape[1]/scale))
        cropped_face = cv2.warpAffine(
                input_img, affine_matrix, shape, borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray 
        return cropped_face


    def lr(self, inputs):
        lr_images = [ ]
        for v in inputs:
            t = imresize(np.array(v,np.float32)/255.0, 0.25)
            lr_images.append(t)
        return lr_images

    def __getitem__(self, index0):
        # dirt = self.dirlist[index]
        # index = int(index*50+25)
        index = self.inter_space*index0 + self.inter_space // 2
        dirt,center = self.indexs[index]
        start = center-self.length//2
        end = start + self.length
        
        
        input = []
        d = {}
        warppath = os.path.join('GT', dirt, '{:0>8d}.png'.format((start + end)//2))
        landmark5 = self.dictlist[warppath]['5landmark']
        gtmatrix = self.cal_warp_affine2d(landmark=landmark5)
        for i in range(start, end):
            if i<1:
                i=1
            if i>(self.numdict[dirt]-2):
                i=(self.numdict[dirt]-2)
            path = os.path.join('GT',dirt, '{:0>8d}.png'.format(i))
            # bytestring = self.nw.get()
            bytestring = self.file_client.get(self.path2id[path])
            tinput = create_opencv_image_from_stringio(bytestring) # rgb
            if self.is_aligned:
                tinput = self.warp(tinput, gtmatrix)
            input.append(tinput)
        
        if self.degradation == 'lr':
            lrimages = self.lr(input)
            lrimages = np.array(lrimages, np.float32) # [t,h,w,c]
            lrimages = torch.from_numpy(lrimages).permute(0,3,1,2)
            if self.mean is not None or self.std is not None:
                lrimages = (lrimages - self.mean)/self.std
            d['llq'] = lrimages
            d['lq'] = torch.nn.functional.interpolate(lrimages,(512,512),mode = 'bilinear',align_corners=True)
            
        elif self.degradation == 'blr':
            lqinput = []
            d = {}
            for i in range(start, end):
                if i<1:
                    i=1
                if i>(self.numdict[dirt]-2):
                    i=(self.numdict[dirt]-2)
                pathlq = os.path.join('LR_Blind', dirt, '{:0>8d}.png'.format(i))
                # bytestring = self.nw.get()
                bytestring = self.file_client.get(self.path2id[pathlq])
                tlqinput = create_opencv_image_from_stringio(bytestring) # rgb
                if self.is_aligned:
                    lqmatrix = self.cal_warp_affine2d(landmark=landmark5, scale=4.0)
                    tlqinput = self.warp(tlqinput, lqmatrix, shape=(512,512), scale=4.0 )
                    
                    # tlqinput = cv2.resize(tlqinput, (512,512))
                    
                
                lqinput.append(tlqinput)
            lqinput = np.array(np.array(lqinput)/255.0, np.float32) # [t,h,w,c]
            lqinput = torch.from_numpy(lqinput).permute(0,3,1,2)
            if self.mean is not None or self.std is not None:
                lqinput = (lqinput - self.mean)/self.std
            d['llq'] = lqinput
            d['lq'] = torch.nn.functional.interpolate(lqinput,(512,512),mode = 'bilinear',align_corners=True)
                
        
        input = np.array(input,np.float32)/255.0# [t,h,w,c]
        input = torch.from_numpy(input).permute(0,3,1,2)
        if self.mean is not None or self.std is not None:
            input = (input - self.mean)/self.std
        d['gt'] = input
        
        # mse = torch.mean((input-lrimages)*(input - lrimages))
        d['path'] = [dirt,center]
        d['gt_path'] = warppath
        d['lq_path'] = warppath
        print(dirt,center)
        if self.length == 1:
            return {
                'lq': d['lq'][0],
                'gt': d['gt'][0],
                'path': [dirt,center],
                'gt_path':warppath,
                'lq_path':warppath
                }
        
        return d

    
    def __len__(self,):
        return len(self.indexs)//self.inter_space
                




@DATASET_REGISTRY.register()
class VFHQFULLNTMEBASICV2TESTUPROTATE(data.Dataset):
    # 参数说明
    # r: 半径，每个时序多少帧（2*r+1)，当r设置为0时就是单帧
    # is_aligned: true or false 是否启用对齐，例如此时r=3，序列长度为7。读取第4帧的人脸5点参数，并根据第四帧人脸参数对7帧进行统一warp
    # dictpath: 人脸参数记录文件 默认位置为pad后的记录文件 's3://kepengxu/vfhqfull/pad_5landmark_matrix.txt'
    # degradation: 退化模式：可选 'blr' 和 'lr' 分别代表了盲退化和仅仅下采样
    # phase： train 或者 test
    # root: 'dataroot'
    def __init__(self, opt):
        super(VFHQFULLNTMEBASICV2TESTUPROTATE, self).__init__()
        print(opt)
        data.Dataset.__init__(self)
        self.length = 2 * opt['r'] + 1
        self.is_aligned  = opt['is_aligned']
        self.dictpath = opt['dictpath']
        self.dictlist = {}
        self.save_freq = 100
        self.opt = opt
        import time
        starttime = time.time()

        self.degradation = opt['degradation']
        self.phase = opt['phase1']
        self.mean = torch.Tensor(opt['mean']).view(1,3,1,1) if 'mean' in opt else None
        self.std = torch.Tensor(opt['std']).view(1,3,1,1) if 'std' in opt else None
        self.dirlist = []
        self.path2id = {}
        self.file_client = ~
        self.inter_space = opt.get('inter_space', 1)
        
        self.pickleload()
        print('inited dataset',time.time()-starttime)
        
    def pickleload(self):
        import pickle
        import refile
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dictlist.pkl','rb')
        self.dictlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'path2id.pkl','rb')
        self.path2id = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'dirlist.pkl','rb')
        self.dirlist = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'numdict.pkl','rb')
        self.numdict = pickle.load(f)
        f.close()
        
        f = refile.smart_open('dataroot'+self.phase+'_'+'list.pkl','rb')
        self.indexs = pickle.load(f)
        f.close()
        
        # self.indexs = []
        

    def cal_warp_affine2d(self, landmark, scale=1.0):
        face_template=np.array([ 
                                [192.98138, 239.94708],
                                [318.90277, 240.1936],
                                [256.63416, 314.01935],
                                [201.26117, 371.41043],
                                [313.08905, 371.15118]
                                ])/scale
        landmarkt = np.array( landmark/scale)
        affine_matrix = cv2.estimateAffinePartial2D(landmarkt, face_template, method=cv2.LMEDS)[0]
        return affine_matrix

    def warp(self, input_img, affine_matrix, shape=(512,512),scale=1):
        shape = (int(shape[0]/scale),int(shape[1]/scale))
        cropped_face = cv2.warpAffine(
                input_img, affine_matrix, shape, borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))  # gray 
        return cropped_face


    def lr(self, inputs):
        lr_images = [ ]
        for v in inputs:
            t = imresize(np.array(v,np.float32)/255.0, 0.25)
            lr_images.append(t)
        return lr_images


    def rotate_s(self,image,sigma):
        cX = image.shape[1]//2
        cY = image.shape[0]//2
        M = cv2.getRotationMatrix2D((cX, cY), sigma, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return rotated

    def __getitem__(self, index0):
        import random
        random.seed(index0)
        sig = int(random.random()*60)-30
        if sig<0:
            sig = sig+360
        # print('rotate  ',sig)
        
        # dirt = self.dirlist[index]
        # index = int(index*50+25)
        index = self.inter_space*index0 + self.inter_space // 2
        dirt,center = self.indexs[index]
        start = center-self.length//2
        end = start + self.length
        
        
        input = []
        d = {}
        warppath = os.path.join('GT', dirt, '{:0>8d}.png'.format((start + end)//2))
        landmark5 = self.dictlist[warppath]['5landmark']
        gtmatrix = self.cal_warp_affine2d(landmark=landmark5)
        for i in range(start, end):
            if i<1:
                i=1
            if i>(self.numdict[dirt]-2):
                i=(self.numdict[dirt]-2)
            path = os.path.join('GT',dirt, '{:0>8d}.png'.format(i))
            # bytestring = self.nw.get()
            bytestring = self.file_client.get(self.path2id[path])
            tinput = create_opencv_image_from_stringio(bytestring) # rgb
            if self.is_aligned:
                tinput = self.warp(tinput, gtmatrix)
            tinput = self.rotate_s(tinput,sig)
            input.append(tinput)
        
        if self.degradation == 'lr':
            lrimages = self.lr(input)
            lrimages = np.array(lrimages, np.float32) # [t,h,w,c]
            lrimages = torch.from_numpy(lrimages).permute(0,3,1,2)
            if self.mean is not None or self.std is not None:
                lrimages = (lrimages - self.mean)/self.std
            d['llq'] = lrimages
            d['lq'] = torch.nn.functional.interpolate(lrimages,(512,512),mode = 'bilinear',align_corners=True)
            
        elif self.degradation == 'blr':
            lqinput = []
            d = {}
            for i in range(start, end):
                if i<1:
                    i=1
                if i>(self.numdict[dirt]-2):
                    i=(self.numdict[dirt]-2)
                pathlq = os.path.join('LR_Blind', dirt, '{:0>8d}.png'.format(i))
                # bytestring = self.nw.get()
                bytestring = self.file_client.get(self.path2id[pathlq])
                tlqinput = create_opencv_image_from_stringio(bytestring) # rgb
                if self.is_aligned:
                    lqmatrix = self.cal_warp_affine2d(landmark=landmark5, scale=4.0)
                    tlqinput = self.warp(tlqinput, lqmatrix, shape=(512,512), scale=4.0 )
                    
                    # tlqinput = cv2.resize(tlqinput, (512,512))
                    
                tlqinput = self.rotate_s(tlqinput,sig)
                lqinput.append(tlqinput)
            lqinput = np.array(np.array(lqinput)/255.0, np.float32) # [t,h,w,c]
            lqinput = torch.from_numpy(lqinput).permute(0,3,1,2)
            if self.mean is not None or self.std is not None:
                lqinput = (lqinput - self.mean)/self.std
            d['llq'] = lqinput
            d['lq'] = torch.nn.functional.interpolate(lqinput,(512,512),mode = 'bilinear',align_corners=True)
                
        
        input = np.array(input,np.float32)/255.0# [t,h,w,c]
        input = torch.from_numpy(input).permute(0,3,1,2)
        if self.mean is not None or self.std is not None:
            input = (input - self.mean)/self.std
        d['gt'] = input
        
        # mse = torch.mean((input-lrimages)*(input - lrimages))
        d['path'] = [dirt,center]
        d['gt_path'] = warppath
        d['lq_path'] = warppath
        print(dirt,center)
        if self.length == 1:
            return {
                'lq': d['lq'][0],
                'gt': d['gt'][0],
                'path': [dirt,center],
                'gt_path':warppath,
                'lq_path':warppath
                }
        
        return d

    
    def __len__(self,):
        return len(self.indexs)//self.inter_space
                






from collections import OrderedDict
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

if __name__ == '__main__':
    import yaml
    with open('options/train/TDRQVAE/TDRQVAE.yml', mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    dataset_opt = opt['datasets']['val']
    dataset = VFHQFULLNTMEBASICV2TESTUP(dataset_opt)
    
    for i in range(dataset.__len__()):
        d = dataset.__getitem__(i)
        print(d.keys())
    