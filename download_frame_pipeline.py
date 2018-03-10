#!/usr/local/bin/python3
import sys
import logging
import json
import requests
import re
import os
from tempfile import NamedTemporaryFile
import cv2
from server_config import detect_server, gender_server
import os

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

def detect_face(frame):
    ntf = NamedTemporaryFile(suffix='.jpg')
    cv2.imwrite(ntf.name, frame)
    r = requests.post(detect_server, files={'image': open(ntf.name, 'rb')})
    ntf.close()
    return json.loads(r.text)['bounding_boxes']

def query_gender(bb, frame):
    ntf = NamedTemporaryFile(suffix='.jpg')
    cv2.imwrite(ntf.name, frame[ bb[1]:bb[3], bb[0]:bb[2], : ])
    r = requests.post(gender_server, files={'image': open(ntf.name, 'rb')})
    ntf.close()
    return json.loads(r.text)['gender']

def extract_frame(v_file, frame_interval, seg):
    v_file.seek(0)
    video = cv2.VideoCapture(v_file.name)
    cnt = 0
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if cnt % frame_interval == 0:
                bbs = detect_face(frame)
                if not all(bbs):
                    continue
                for bb in bbs:
                    gender = query_gender(bb, frame)
                    if gender is 0:
                        cv2.imwrite('./thumbnails/{}_{}.jpg'.format(seg, cnt), frame[ bb[1]:bb[3], bb[0]:bb[2], :])
            cnt = cnt + 1
        else:
            break
    return frames

def get_video(v_code):
    seg = 8
    while True:
        video_src = "https://condombaby.com/{v_code:s}/1080p/{seg:05d}.ts".format(v_code=v_code, seg=seg)
        print("start to process {} segment.".format(seg))
        ntf = NamedTemporaryFile()
        ntf = download_video(video_src, ntf)
        thumbnails = extract_frame(ntf, 10, seg)
        if os.path.getsize(ntf.name) < 80000:
            print("video processing complete")
            ntf.close()
            break
        ntf.close()
        seg = seg+1


if __name__ == '__main__':
    with open('porn_star.json') as jsonfile:
        star2video = json.load(jsonfile)
    for name, videos in star2video.items():
        for v in videos:
            get_video(v)
