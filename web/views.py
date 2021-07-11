from pathlib import WindowsPath
from django.shortcuts import render
from .models import ImageTarget, Video

from web_demo import WebDemo

import cv2
import PIL
import numpy as np
import time

import json
from os.path import exists

# Create your views here.
def home(request):
    return render(request, 'home.html')

def result(request):
    if request.POST:

        title = request.POST["title"]
        video = request.FILES["videofile"]

        vids = Video(title = title, videofile = video)
        vids.save()


        vidobj = cv2.VideoCapture(vids.videofile.path)
        frame_list = []

        while True:
            ret, frame = vidobj.read() # 프레임 정보, 프레임
            frame_list.append(frame)

            if not ret:
                break
        if vidobj.isOpened():
            vidobj.release()
        

        # WebDemo
        videos = {'ims':frame_list, 'coordinates':(300, 110, 165, 250)}  # 나중에 coordinates 변수로 바꾸기 (x, y, w, h)
        config_path = '../config_inference.json'  # json file for model implement
        config = json.load(open(config_path))
        inf_time, speed = WebDemo(videos=videos, cfg=config)


        context = {
            'vids': vids,
            'frame_length': len(frame_list)
        }

        return render(request, 'result.html', context)