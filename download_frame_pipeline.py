#!/usr/local/bin/python3
import sys
import logging
import json
import requests
import re
import os
from tempfile import NamedTemporaryFile
import cv2

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-20s: %(levelname)-8s %(message)s')
logging.getLogger('').addHandler(console)

AVGLE_SEARCH_VIDEOS_API_URL = 'https://api.avgle.com/v1/search/{}/0'

def get_video_list(name):
    name_logger = logging.getLogger('FaceSearch.video_list')
    resp = json.loads(urllib.request.urlopen(AVGLE_SEARCH_VIDEOS_API_URL.format(urllib.parse.quote_plus(name))).read().decode())
    if resp['success']:
        videos = resp['response']['videos']
        if len(videos) <= 0:
            name_logger.info('there is no video match key word {}'.format(name))
        video_tasks = []
        for v in videos:
            video_tasks = video_tasks + get_video_seg(v['embedded_url'], name, int(v['vid']))
        return video_tasks

def download_video(v_src, temp_file):
    try:
        r = requests.get(v_src, stream=True)
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                temp_file.write(chunk)
        return temp_file
    except requests.exceptions.SSLError:
        return None

def extract_frame(v_file, frame_interval, seg):
    v_file.seek(0)
    video = cv2.VideoCapture(v_file.name)
    cnt = 0
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if cnt % frame_interval == 0:
                frames.append(frame)
                cv2.imwrite('./frame/{}/{}.jpg'.format(seg, cnt), frame)
    return frames

def get_video(v_code):
    seg = 1
    while True:
        video_src = "https://condombaby.com/{v_code:s}/1080p/{seg:05d}.ts".format(v_code=v_code, seg=seg)
        ntf = NamedTemporaryFile()
        ntf = download_video(video_src, ntf)
        extract_frame(ntf, 10, seg)
        ntf.close()
        if not ntf:
            print("the last segment is {}".format(seg))
        break
        seg = seg+1


if __name__ == '__main__':
    with open('porn_star.json') as jsonfile:
        star2video = json.load(jsonfile)
    for name, videos in star2video.items():
        for v in videos:
            get_video(v)
