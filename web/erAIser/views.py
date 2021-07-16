from pathlib import WindowsPath
from django.shortcuts import render
from numpy.core.fromnumeric import resize
from .models import ImageTarget, Video, Image
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import HttpResponse, response

from .web_demo import WebDemo # 추가
from os.path import exists # 추가
import json
import shutil 
import imageio
from pathlib import Path
import cv2
import PIL
import numpy as np
import time
from moviepy.editor import *
from AANet.preprocessing_aanet import resize_bbox

# Create your views here.
@method_decorator(csrf_exempt, name="dispatch")
def home(request):
    return render(request, 'index.html')

def result(request):
    if request.POST:


        if request.POST["resume"] == "resume1":

            title = request.POST["filename"]
            video = request.FILES["uploadImages"]
            x = int(request.POST['x'])
            y = int(request.POST['y'])
            w = int(request.POST['w'])
            h = int(request.POST['h'])

            vids = Video(title = title, videofile = video)
            vids.save()
            
            vidobj = cv2.VideoCapture(vids.videofile.path)
            origin_fps = vidobj.get(cv2.CAP_PROP_FPS)
            frame_list = []

            while True:
                ret, frame = vidobj.read() # 프레임 정보, 프레임
                frame_list.append(frame)


                if not ret:
                    break
            
            frame_list = frame_list[:-1]
            videos = {'ims':frame_list, 'coordinates':(x, y, w, h)} # _origin
            config_path = 'erAIser/config_web.json' # 기존 erAIser -> classifier
            config = json.load(open(config_path))
            config['using_aanet'] = False
            inpainted, _, _ = WebDemo(videos=videos, cfg=config)

            for i, im in enumerate(inpainted):
                cv2.imwrite('media/src_img/{}.png'.format(i), im)
            
            ims_list = ['media/src_img/{}.png'.format(i) for i in range(len(inpainted))]

            clip = ImageSequenceClip(ims_list, fps = origin_fps)
            clip.write_videofile("media/rst/{}".format(title), fps = 50)

        
            if vidobj.isOpened():
                vidobj.release()
            
            vid_path = "/media/rst/{}".format(title)
            #동영상 save
            #vidswriter = cv2.VideoWriter_fourcc('X','2','6','4')



            response_data = {
                'video_path' : vid_path
            }

            return HttpResponse(json.dumps(response_data), content_type = "application/json")
            #return render(request, 'index.html', response_data)
        elif request.POST["resume"] == "resume2":

 

            videotitle = request.POST["videoname"]
            video = request.FILES["video"]
            imagetitle = request.POST["imagename"]
            image = request.FILES["image"]
            x = int(request.POST['x'])
            y = int(request.POST['y'])
            w = int(request.POST['w'])
            h = int(request.POST['h'])
            img_x = int(request.POST['img_x'])
            img_y = int(request.POST['img_y'])
            img_w = int(request.POST['img_w'])
            img_h = int(request.POST['img_h'])

            print("***********************************")
            print(videotitle)
            print(video)
            print(imagetitle)
            print(image)
            print("***********************************")
            print("video 좌표 : ", x, y, w, h)
            print("image 좌표 : ", img_x, img_y, img_w, img_h)

            imgs = Image(title = imagetitle, imagefile = image)
            imgs.save()
            vids = Video(title = videotitle, videofile = video)
            vids.save()
            

            print(vids.videofile.url)
            
            vidobj = cv2.VideoCapture(vids.videofile.path)
            origin_fps = vidobj.get(cv2.CAP_PROP_FPS)
            frame_list = []

            while True:
                ret, frame = vidobj.read() # 프레임 정보, 프레임
                frame_list.append(frame)


                if not ret:
                    break
            
            frame_list = frame_list[:-1]
            videos = {'ims':frame_list, 'coordinates':(x, y, w, h)}
            config_path = 'erAIser/config_web.json' # 기존 erAIser -> classifier
            config = json.load(open(config_path))
            config['using_aanet'] = True
            source = {}
            source['image'] = imageio.imread('media/image/{}'.format(imagetitle))
            source['coordinates'] = (img_x, img_y, img_w, img_h)
            source['coordinates'] = resize_bbox(source['coordinates'], (512,512),
                                                source['image'].shape[:2][::-1])
            inpainted, _, _ = WebDemo(videos=videos, cfg=config, source=source)

            for i, im in enumerate(inpainted):
                cv2.imwrite('media/src_img/{}.png'.format(i), im)
            
            ims_list = ['media/src_img/{}.png'.format(i) for i in range(len(inpainted))]

            clip = ImageSequenceClip(ims_list, fps = origin_fps)
            clip.write_videofile("media/rst/{}".format(videotitle), fps = origin_fps)

        
            if vidobj.isOpened():
                vidobj.release()
            
            vid_path = "/media/rst/{}".format(videotitle)
            #동영상 save
            #vidswriter = cv2.VideoWriter_fourcc('X','2','6','4')



            response_data = {
                'video_path' : vid_path
            }

            return HttpResponse(json.dumps(response_data), content_type = "application/json")
            #return render(request, 'index.html', json.dumps(response_data))