#!/usr/local/bin/python3
import sys
import glob
import os
import cv2

star_list = ['三上悠亜', '吉川愛美', 'RION', '波多野結衣']
star_list = ['大橋未久']

for s in star_list:
    for i, name in enumerate(glob.glob('./thumbnails/{}/*.jpg'.format(s))):
        cropped = cv2.imread(name)
        scaled = cv2.resize(cropped, (182, 182))
        cv2.imwrite(name, scaled)
        os.rename(name, './thumbnails/{star:s}/{star:s}_{num:04d}.jpg'.format(star=s, num=i+1))

