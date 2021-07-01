import glob
from vos.test import *

import os, sys
import os.path
import numpy as np
import cv2
import pickle

from vi.model import generate_model
from vi.vi_inference import VIInference


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_inference.json',
                    help='hyper-parameter of SiamMask and VINet in json format')
parser.add_argument('--base_path', default='./vos/data/tennis', help='datasets')
parser.add_argument('--save_path', default='./results', help='save path for modified video')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()


if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from vos.models.custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)
    vinet, _ = generate_model(cfg['opt'])
    vinet.eval()
    inf = VIInference(vinet)


    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    # VI쪽으로 넘겨주기 위해 이미지 사이즈 변환 (w,h) = (512,512)
    ims = [cv2.resize(cv2.imread(imf), (512,512)) for imf in img_files]
    
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
       init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
       x, y, w, h = init_rect
    except:
       exit()

    # x, y, w, h = 300, 110, 165, 250
    toc = 0

    for f, im in enumerate(ims):        
        # VI 쪽으로 이미지 넘겨주기 위해 저장하는 코드
        im_name = 'images/' + str(f).zfill(5) + '.jpg'
        cv2.imwrite(im_name, im)

        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            
        # 첫번째 frame부터 tracking
        state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        inf.inference(im, mask)
            
        toc += cv2.getTickCount() - tic

    

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    inf.to_video("test", args.save_path)
    


