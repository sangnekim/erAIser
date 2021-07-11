import glob
import os, sys
import os.path
import numpy as np
import cv2
import pickle
import json
from os.path import exists

from vos.test import *
from vi.model import generate_model
from vi.vi_inference import VIInference


def WebDemo(videos: dict, cfg: json):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    from vos.vos_models.custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if cfg.resume:
        assert isfile(cfg.resume), 'Please download {} first.'.format(cfg.resume)
        siammask = load_pretrain(siammask, cfg.resume)
    
    siammask.eval().to(device)
    vinet, _ = generate_model(cfg['opt'])
    vinet.eval()
    inf = VIInference(vinet)

    # Parse Image file and coordinates of bounding box
    ims = videos['ims']  # list object
    ims = [cv2.resize(im, (512,512)) for im in ims]  # list comprehension for resizing -> it can be change like 'map'
    x, y, w, h = videos['coordinates']  # tuple or list object

    toc = 0

    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init vos 
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        
        # tracking
        state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)
        location = state['polygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        inf.inference(im, mask)

        toc += cv2.getTickCount() - tic
    
    toc /= cv2.getTickFrequency()  # inference time
    fps = f / toc  # speed
    inf.to_video("test", cfg.save_path)  # save video in media directory to show on web browser
    
    return toc, fps  # retrun inference time & speed


if __name__=="__main__":
    config_path = '../config_inference.json'
    config = json.load(open(config_path))
    videos = {}
    img_files = sorted(glob.glob(join(config['base_path'], '*.jp*')))  # base_path: base video directory path
    videos['ims'] = [cv2.imread(imf) for imf in img_files]
    inf_time, speed = WebDemo(videos=videos, cfg=config)
    print('Inference Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(inf_time, speed))
