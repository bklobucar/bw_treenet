#!/bin/env python3

import os
import numpy as np
import torch
import time
from osgeo import gdal
import util.util as u

import torch.nn.functional as F
from torch.autograd import Variable
from model import BWTreeNet
from util.EnhanceImage import EnhanceImage
from SwissTestParameters import img_rows, img_cols, window_shape, source_img_band, classnum, result_img_band, model_path, gpu, model_name,result_path , project_name,model_path,test_path,delete_edge_value
from SwissTestParameters import normalization_strategy,overlap_strategy,stride,abandon_strategy,delete_edge_strategy,flip_strategy,ratio_output_strategy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

class mainClass(object):
    def __init__(self):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.window_shape = window_shape
        self.stride = stride
        self.source_img_band = source_img_band
        self.classnum = classnum
        self.result_img_band = result_img_band
        if ratio_output_strategy:
            self.result_img_band = 2
        self.model_path = model_path
        self.gpu = gpu
        self.model_name = model_name
    

    def pred_patch(self, model, patch_list):
        
        model.eval()
        img_length = patch_list.shape[0]
        pred_list = []
        for i in range(img_length):
            images = patch_list[i]
            pred_result =[]
            if flip_strategy:
                image_data = images[0]
                image_flip1 = np.flip(image_data,axis=0)[np.newaxis,:,:]
                image_flip2 = np.flip(image_data,axis=1)[np.newaxis,:,:]
                image_flip3 = image_data
                image_flip3 = image_flip3[::-1].T[np.newaxis,:,:]
                image_flip4 = image_data
                image_flip4 = image_flip4[::-1][:,::-1][np.newaxis,:,:]
                images = np.concatenate((images,image_flip1,image_flip2,image_flip3,image_flip4),axis=0)

            for i in range(images.shape[0]):
                test_image = images[i]
                test_image = test_image[np.newaxis,np.newaxis,:]
                test_image = torch.tensor(test_image)
                test_image = Variable(test_image)
                test_image = test_image.cuda(self.gpu)
                test_image = test_image.float()
            
                pred = model(test_image)
                if ratio_output_strategy ==True:
                    
                    pred = pred*255
                    
                    pred[pred<0] = 0
                    pred[pred>255] = 255
                    pred = pred.detach()
                    pred = pred.cpu()
                    pred = pred.numpy()
                    pred_result.append(pred)
                else:
                    pred = pred.argmax(1)
                    pred = pred.cpu()
                    pred = pred.numpy()
                    pred_result.append(pred)
            
            pred_result = np.asarray(pred_result, dtype='uint8')
            
            if len(pred_result.shape)==4:
                pred_result = pred_result[:,:,np.newaxis,:,:]
            cls_num=pred_result.shape[2]
            
            for c in range(cls_num):
                if flip_strategy:
                    image_ori = pred_result[0][0][c]
                    image_ori = np.asarray(image_data,dtype='int32')
                    image_flip1 = pred_result[1][0][c]

                    image_flip1 = np.flip(image_flip1,axis=0)
                    image_flip2 = pred_result[2][0][c]
                    image_flip2 = np.flip(image_flip2,axis=1)
                    
                    image_flip3 = pred_result[3][0][c]
                    image_flip3 = image_flip3[:,::-1].T
                    image_flip4 = pred_result[4][0][c]
                    image_flip4 = image_flip4[::-1][:,::-1]
                    
                    image_ori = image_ori+image_flip1+image_flip2+image_flip3+image_flip4
                    
                    
                    if ratio_output_strategy:
                        image_ori = image_ori/5
                    else:
                        image_ori[image_ori>2]=1
                    pred_result[0][0][c]=image_ori
            pred_list.append(pred_result[0][0])
            
        pred_list = np.asarray(pred_list, dtype='uint8')
        return pred_list
    
    def patch_process(self,patch):
        if normalization_strategy:
            patch[patch > 255] = 255
            patch = np.nan_to_num(patch,nan=0)
            eh = u.EqualizeHist(patch,bins=255)
            patch = eh.operation()
            patch = patch.astype(np.float64)
        return patch
    
        
    def is_edge(self,write_w,write_h,result_w,result_h,window_shape):
        edge_flag = False
        if write_w == 0 or write_h == 0:
            edge_flag = True
        if write_w+window_shape>=result_w or write_h+window_shape>=result_h:
            edge_flag =True
        return edge_flag
        
    def write_result(self,result_img,pred_result_list,present_h,result_w,result_h):
        pred_num = pred_result_list.shape[0]
        for n in range(0, pred_num, 1):
            write_w = n*self.stride
            write_h = present_h
            
            patch_wid = self.window_shape
            patch_hei = self.window_shape
            
            if write_w > result_w or write_h > result_h:
                continue
            if delete_edge_strategy:
                if self.is_edge(write_w,write_h,result_w,result_h,window_shape):
                    patch = pred_result_list[n]
                else:
                    img_pred = pred_result_list[n]
                    patch = img_pred[:,delete_edge_value:patch_hei-delete_edge_value,delete_edge_value:patch_wid-delete_edge_value]
            else:
                img_pred = pred_result_list[n]
            
            for c in range(pred_result_list.shape[1]):
                if not ratio_output_strategy:
                    patch[patch>0] = 255
            
                if delete_edge_strategy and (self.is_edge(write_w,write_h,result_w,result_h,window_shape)==False):
                    edge_value = delete_edge_value
                else:
                    edge_value = 0
                
                write_start_w = write_w+edge_value
                write_start_h = write_h+edge_value
                if write_h + window_shape > result_h:
                    write_start_h = result_h-window_shape
                if write_w + window_shape > result_w:
                    write_start_w = result_w-window_shape
                result_img.GetRasterBand(c+1).WriteArray(
                patch[c], write_start_w, write_start_h)
            
        return 1

    def load_model(self,model_name,model_path):
        print('Load model......',model_name)
        model = eval(self.model_name)(n_class=self.classnum)
        if self.gpu >= 0:
            model.cuda(self.gpu)
        model.load_state_dict(torch.load(
            model_path, map_location='cuda:0'))
        return model
    
    def create_result_img(self,source_path,target_path):
        if not os.path.exists(source_path):
            print('no file')
        img = gdal.Open(source_path)
        tiffwid = img.RasterXSize
        tiffhei = img.RasterYSize
        
        tiffwid = img.RasterXSize
        tiffhei = img.RasterYSize
        gdal.SetConfigOption("GDAL_FILENAME_IF_UTF8", "NO")
        gdal.SetConfigOption("SHAPE_ENCODING", "")
        tiff_geotransform = img.GetGeoTransform()
        tiff_proj = img.GetProjection()
        if tiffwid == 0:
            print("read fail")
            return False
        exist_file = os.path.exists(target_path)
        if exist_file == True:
            os.remove(target_path)
        result_tiff_datatype = gdal.GDT_Byte
        result_tiff_width = tiffwid
        result_tiff_height = tiffhei
        result_tiff_bands = self.result_img_band
        result_tiff_geotransform = tiff_geotransform
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(target_path, result_tiff_width,
                                result_tiff_height, result_tiff_bands, result_tiff_datatype)
        if(dataset != None):
            dataset.SetGeoTransform(result_tiff_geotransform)
            dataset.SetProjection(tiff_proj)
        else:
            print('generate fail')
            return False
        return img,dataset
    
    def predict_WFV_process(self,test_img,result_img,model):
        tiffwid = test_img.RasterXSize
        tiffhei = test_img.RasterYSize
        
        for h in range(0, tiffhei, self.stride):
            if h+self.window_shape > tiffhei:
                heigh_patch = test_img.ReadAsArray(
                    0, tiffhei-self.window_shape, tiffwid, self.window_shape)
            else:
                heigh_patch = test_img.ReadAsArray(0, h, tiffwid, self.window_shape)
            
            img_patch_list = []
            patch_w = heigh_patch.shape[1]

            for w in range(0, patch_w, self.stride):
                if w+self.window_shape > patch_w:
                    patch = heigh_patch[0:self.window_shape, patch_w-window_shape:patch_w]
                else:
                    patch = heigh_patch[0:self.window_shape, w:w+self.window_shape]
                patch = self.patch_process(patch)
                patch = patch[np.newaxis,:]
                img_patch_list.append(patch)
                
            img_patch_list = np.asarray(img_patch_list, dtype='float')
            
            pred_patch_list = self.pred_patch(model, img_patch_list)
            self.write_result(result_img,pred_patch_list,h,tiffwid,tiffhei)

        return result_img
    
    def test_WFV_img(self, source_path, target_path):
        print('testing......')
        
        time_start = time.time()
        
        model = self.load_model(self.model_name,self.model_path)

        test_img,result_img = self.create_result_img(source_path,target_path)

        result_img = self.predict_WFV_process(test_img,result_img,model)
        
        time_end = time.time()
        print('process time： ',str('%.2f'%(time_end-time_start)),' s')

        del result_img

def main():
    for i in test_path:
        source_path = i
        target_path = result_path+project_name+'/'+i.split('/')[-2]+'/'
        os.makedirs(target_path, exist_ok=True)
        img_list = os.listdir(source_path)
        for img_name in img_list:
            source_main_path = source_path+img_name
            target_main_path = target_path + img_name
            if  img_name[-3:]!='tif':
                continue
            print(source_main_path)
            worker = mainClass()
            worker.test_WFV_img(source_main_path, target_main_path)


if __name__ == '__main__':
    main()
