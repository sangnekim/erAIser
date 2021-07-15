import glob
import os, sys
import os.path
import numpy as np
import cv2
import pickle
import json
from os.path import exists
import imageio

from vos.test import *
from vos.vos_models.custom import Custom

from vi.model import generate_model
from vi.vi_inference import VIInference

from AANet.preprocessing_aanet import resize_bbox
from AANet.aa_inference import AAInference

from .web_demo_with_aanet import aanet_main


def WebDemo(videos: dict, cfg: json, source=None):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    from vos.vos_models.custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if cfg["resume"]:
        assert isfile(cfg['resume']), 'Please download {} first.'.format(cfg['resume'])
        siammask = load_pretrain(siammask, cfg['resume'])
    
    siammask.eval().to(device)
    vinet, _ = generate_model(cfg['opt'])
    vinet.eval()
    inf = VIInference(vinet)

    # Parse Image file and coordinates of bounding box
    ims = videos['ims']  # list object
    ims = [cv2.resize(im, (512,512)) for im in ims]  # list comprehension for resizing -> it can be change like 'map'
    videos['coordinates'] = resize_bbox(videos['coordinates_origin'],
                                        videos['ims'][0].shape[:2][::-1],
                                        (512,512))
    x, y, w, h = videos['coordinates']  # tuple or list object
    

    toc = 0

    if cfg['using_aanet']:
        # set aanet
        AAInf=AAInference(cfg['aanet_config_path'], cfg['aanet_model_path'])
        
        # save information of source image(origin) and video(512,512)
        AAInf.source_image=source['image']
        AAInf.source_image_bbox=source['coordinates']
        
        AAInf.origin_video_512=[(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255) for img in ims]

    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        
        if f == 0:  # init vos 
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        
        # tracking
        state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        if cfg['using_aanet']:
            # save information of object in origin video(512,512)
            AAInf.origin_video_pos_512.append(state['target_pos'].astype(int))
            AAInf.origin_video_mask_512.append(mask.astype(int))

        inf.inference(im, mask)

        toc += cv2.getTickCount() - tic
    
    toc /= cv2.getTickFrequency()  # inference time
    fps = f / toc  # speed
    inpainted = inf.inpainted_imgs
    inpainted = [cv2.cvtColor(inpainted_one, cv2.COLOR_BGR2RGB) for inpainted_one in inpainted]

    if cfg['using_aanet']:
        # check source image in AANet class
        inpainting_time={'toc':toc, 'fps':fps}
        
        assert type(AAInf.source_image) is imageio.core.util.Array
        
        # generate new video
        aa_result, toc=aanet_main(AAInf, inpainted, videos, cfg)
        aa_result=[(f*255).astype(np.uint8) for f in aa_result]
        aa_result = [cv2.cvtColor(aa_result_one, cv2.COLOR_BGR2RGB) for aa_result_one in aa_result]
        #*
        toc+=inpainting_time['toc']
        fps= (f*4)/toc # vinet에서 512 video에 한 번, aanet origin_video, generating_video, driving_video에 3번, 총 4번
        return aa_result, toc, fps
    
    else:
        return inpainted, toc, fps  # retrun inference time & speed


if __name__=="__main__":
    config_path = './config_web.json'
    config = json.load(open(config_path))
    videos = {}
    config['base_path'] = '../vos/data/tennis'
    img_files = sorted(glob.glob(join(config['base_path'], '*.jp*')))  # base_path: base video directory path
    videos['ims'] = [cv2.imread(imf) for imf in img_files]
    videos['coordinates_origin'] = (300, 110, 165, 250)
    inf_time, speed = WebDemo(videos=videos, cfg=config)
    print('Inference Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(inf_time, speed))