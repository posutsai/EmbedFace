#!/usr/local/bin/python3
import sys
import json
config = {
    'detect_server': {
        'ip': '0.0.0.0',
        'port': 80
    },
    'gender_server': {
        'ip': '0.0.0.0',
        'port': 9000
    }
}
with open('server_config.json', 'w') as outfile:
    json.dump(porn_star_video_list, outfile)

