#!/usr/bin/env python3

import os
import sys
import json
import base64
import requests
import logging
import simplejson as json
import struct

DB = 1
DIM = 512


class Fawn:
    def __init__(self, url='http://127.0.0.1:8888'):
        self.session = requests.Session()
        self.url = url

    def search(self, feature, **kwargs):
        # feature is list of 512 float numbers 
        buf = struct.pack('%df' % DIM, *feature)

        K = 100
        if 'K' in kwargs:
            K = kwargs['K']
            pass

        req = {
            'db': DB,
            'raw': False,
            'type': '',
            'K': K,
            'hint_K': K,  # 10-NN for each face
            'R': 1e38,
            'hint_R': 1e38,
            'url': '',
            'content': base64.b64encode(buf),
        }
        # server will return at most hint_K * faces in the image
        # print('request:')
        # print(json.dumps(req))
        respo = self.session.post(self.url + '/search', json=req)
        # print resp.text
        resp = json.loads(respo.text)
        candidates = {}
        if not 'hits' in resp:
            print(respo.text)
            raise Exception('fail to search')
        # print('resp:')
        # print(respo.text)
        hits = resp['hits']
        return hits

    def insert(self, key, feature, meta):
        buf = struct.pack('%df' % DIM, *feature)
        req = {
            'db': DB,
            'key': key,
            'raw': False,
            'content': base64.b64encode(buf),
            'meta': meta,
        }
        resp = self.session.post(self.url + '/insert', json=req)
        if resp.status_code != 200:
            print(resp.status_code)
            raise Exception(resp.headers.get("Error", "failed to insert"))
        pass

    def sync(self):
        resp = self.session.post(self.url + '/misc', json={'method': 'sync'})
        return

    def reindex(self):
        resp = self.session.post(self.url + '/misc', json={'method': 'reindex', 'db': DB})
        return

    def stats(self):
        resp = self.session.post(self.url + '/stats', json={})
        return json.loads(resp.text)

    pass


    # for i in range(100):
    #     feature = dummy_feature(i)
    #     xxx = client.search(feature, K=2)
    #     print("QUERY %d" % i)
    #     print(xxx)
    #     print("----------")
    # pass
