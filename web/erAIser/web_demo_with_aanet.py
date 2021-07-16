import glob
from vos.test import *

import os, sys
import os.path
import numpy as np
import cv2
import pickle

from vos.vos_models.custom import Custom

from vi.model import generate_model
from vi.vi_inference import VIInference

from AANet.aa_inference import *


def assign_siammask(cfg, device):     
    siammask = Custom(anchors=cfg['anchors'])
    if cfg['resume']:
        assert isfile(cfg['resume']), 'Please download {} first.'.format(cfg['resume'])
        siammask = load_pretrain(siammask, cfg['resume'])
    siammask.eval().to(device)
    
    return siammask
    
def track_mask(ims, bbox, AAInf, siammask, cfg, device, method='origin video', toc=0):
    """method : origin video(1), source image(2), driving video(3)"""
    
    x,y,w,h=bbox

    if method=='driving video':
        target_poses=[]
        target_sizes=[]
        ims=[(im*255).astype(int) for im in ims] # SIAM : 0~255, AANet: 0~1
    toc_sub=0
    for f, im in enumerate(ims):        

        tic = cv2.getTickCount()
        
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker

        # 첫번째 frame부터 tracking
        state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)
        mask = state['mask'] > state['p'].seg_thr

        if method=='origin video':
            AAInf.origin_video_pos.append(state['target_pos'].astype(int))
            AAInf.origin_video_sz.append(state['target_sz'].astype(int))
            AAInf.origin_video_mask.append(mask.astype(int))
        elif method=='source image':
            AAInf.source_image_mask=mask.astype(int)
            AAInf.source_image=mask_image(AAInf.source_image, AAInf.source_image_mask)
        elif method=='driving video':
            target_sizes.append(state['target_sz'].astype(int))
            target_poses.append(state['target_pos'].astype(int))
            
        toc_sub += cv2.getTickCount() - tic
    
        
    toc_sub /= cv2.getTickFrequency()
     
    
    toc+=toc_sub
    
    if method=='driving video':
        return target_sizes, target_poses, toc
    
    return toc   

def aanet_main(AAInf, vi_result, origin_video, cfg):
    
    #Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    
    AAInf.vi_result=[cv2.cvtColor(im, cv2.COLOR_BGR2RGB)/255 for im in vi_result]
    print("vi_result: ",len(AAInf.vi_result))
    # save original video
    ims = origin_video['ims']
    AAInf.origin_video=[(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255) for img in ims]
    
    
    ### track origin video ### 
    
    # siam mask 할당.
    siammask=assign_siammask(cfg, device)
    
    # bbox of object of origin video
    bbox=origin_video['coordinates_origin']
    
    # track origin video 
    toc=track_mask(ims, bbox, AAInf, siammask, cfg, device, method='origin video', toc=0)

    ### crop video to (384, 384) ###
    AAInf.driving_video_384, AAInf.driving_video_384_mask, _=AAInf.generate_driving_video_384(
        AAInf.origin_video, AAInf.origin_video_mask, AAInf.origin_video_pos,
                            AAInf.origin_video_sz)

    ### track source image ### 
    siammask = assign_siammask(cfg, device)

    
    # take bbox of source image
    #     cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    #     # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #     try:
    #         init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
    #         bbox = init_rect
    #     except:
    #         exit()
    

    assert AAInf.source_image_bbox
    
    bbox=AAInf.source_image_bbox # source image에 대한 bbox
    
    toc=track_mask([AAInf.source_image], bbox, AAInf, siammask,cfg,device, method='source image', toc=toc)
    
    ### refine source image and driving video to (384, 384) shape 
    # 동시에 AAInf.source_image_bbox_384도 할당됨.
    if AAInf.driving_video_384:
        AAInf.source_image_384, AAInf.driving_video_384=AAInf.resize_for_generating_animation(
  AAInf.source_image, AAInf.driving_video_384, AAInf.source_image_bbox) 
    else:
        raise NameError('use driving_video_384')
    
    ### generate animation ###

    toc_sub=0
    tic=cv2.getTickCount()
    
    AAInf.source_animation=AAInf.generate_animation(
        AAInf.source_image_384, AAInf.driving_video_384, cfg['aanet_ani_mode'])

    toc_sub=cv2.getTickCount()-tic
    toc_sub /= cv2.getTickFrequency()
    
    toc+=toc_sub
    ### track driving video(animation video) ### 

    AAInf.source_animation, AAInf.source_image_384_bbox=AAInf.resize_animation(AAInf.source_animation, AAInf.origin_video, AAInf.source_image_384_bbox)

    siammask = assign_siammask(cfg, device)

    bbox=AAInf.source_image_384_bbox

    target_sizes, target_poses, toc= track_mask(AAInf.source_animation, bbox, AAInf, siammask,cfg, device, method='driving video', toc=toc)
    
    target_sizes, AAInf.source_animation=AAInf.resize_target_size(target_sizes, AAInf.origin_video_sz_512, AAInf.source_animation) #*
    
    ### decouple object from background ###
    masks=AAInf.decouple_background(AAInf.source_animation)
    
    ### synthesize object to inpainting video ### 
    video=AAInf.synthesize_object(masks, target_sizes, target_poses)
    
    # 비디오 합성 (0710 메서드화.)  
    return video, toc

    
        
