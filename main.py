import os
import cv2
import json
import retinex
from dehaze import dehaze
from tqdm import tqdm



d = False # 去雾
e = True # 图像增强
# 图片路径
data_path = 'input/set3'
save_path = 'result/set3'
img_list = os.listdir(data_path)

if len(img_list) == 0:
    print('Data directory is empty.')
    exit()

with open('config.json', 'r') as f:
    config = json.load(f)
# img_list = ['1 (5).tif']
for img_name in tqdm(img_list):
    if img_name == '.gitkeep':
        continue
    img = cv2.imread(os.path.join(data_path, img_name))
    #img = cv2.imread(r'./person/input.jpg')
    # 保存原图
    cv2.imwrite('./{}/{}'.format(save_path,img_name),img)  


    ###########################去雾###############################################
    if d:
        #--------------------去雾-------------------------------
        dehaze_image = dehaze(img)
        cv2.imwrite('./{}/{}_dehaze.{}'.format(save_path,
                                                img_name.split('.')[0],
                                                img_name.split('.')[1]),dehaze_image) 
        #----------------------------------------------------------------
        

    ############################增强retinex#########################################
    if e:
        #--------------retinex增强 MSRCR------------------------------------------
        img_msrcr = retinex.MSRCR(
            img,
            config['sigma_list'],
            config['G'],
            config['b'],
            config['alpha'],
            config['beta'],
            config['low_clip'],
            config['high_clip']
        )
        cv2.imwrite('./{}/{}_MSRCR.{}'.format(save_path,
                                            img_name.split('.')[0],
                                            img_name.split('.')[1]),img_msrcr)  
        #----------------------------------------------------------------------------


        #--------------retinex增强 AUTOMSRCR------------------------------------------
        img_amsrcr = retinex.automatedMSRCR(
            img,
            config['sigma_list']
        )
        cv2.imwrite('./{}/{}_automatedMSRCR.{}'.format(save_path,
                                                    img_name.split('.')[0],
                                                    img_name.split('.')[1]),img_amsrcr)    
        #-----------------------------------------------------------------------------
        
        #--------------retinex增强 MSRCP-----------------------------------------------
        img_msrcp = retinex.MSRCP(
            img,
            config['sigma_list'],
            config['low_clip'],
            config['high_clip']        
        )
        cv2.imwrite('./{}/{}_MSRCP.{}'.format(save_path,
                                              img_name.split('.')[0],
                                              img_name.split('.')[1]
                                              ),img_msrcp)
        #------------------------------------------------------------------------------


    # show image    
    # cv2.imshow('Image', img)
    # cv2.imshow('retinex', img_msrcr)
    # cv2.imshow('Automated retinex', img_amsrcr)
    # cv2.imshow('MSRCP', img_msrcp)





