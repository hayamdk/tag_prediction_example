#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import niconico_chainer_models
import pickle
import argparse
import urllib.request
import numpy
import PIL.Image
import chainer

def fix_b_to_str(model) :
	if hasattr(model, "__dict__"):
		list = model.__dict__.copy()
		for d in list:
			d_str = d.decode('utf8')
			model.__dict__[d_str] = model.__dict__[d]
			del model.__dict__[d]
			fix_b_to_str(model.__dict__[d_str])

def fetch_image(url):
    response = urllib.request.urlopen(url)
    image = numpy.asarray(PIL.Image.open(response).resize((224,224)), dtype=numpy.float32)
    if (not len(image.shape)==3): # not RGB
        image = numpy.dstack((image, image, image))
    if (image.shape[2]==4): # RGBA
        image = image[:,:,:3]
    return image

def to_bgr(image):
    return image[:,:,[2,1,0]]
    return numpy.roll(image, 1, axis=-1)

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("mean")
parser.add_argument("tags")
parser.add_argument("image_url")
parser.add_argument("--gpu", type=int, default=-1)
args = parser.parse_args()

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

f = open(args.model, 'rb')
model = pickle.load(f, fix_imports=True, encoding='bytes')
fix_b_to_str(model)

if args.gpu >= 0:
    model.to_gpu()

mean_image = numpy.load(open(args.mean, 'rb'), encoding='bytes')
tags = [line.rstrip() for line in open(args.tags, encoding='utf-8')]
tag_dict = dict((i,tag) for i, tag in enumerate(tags))

img_preprocessed = (to_bgr(fetch_image(args.image_url)) - mean_image).transpose((2, 0, 1))

predicted = model.predict(xp.array([img_preprocessed]))[0]
#predicted = model.predict_all(xp.array([img_preprocessed]))

top_10 = sorted(enumerate(predicted), key=lambda index_value: -index_value[1])[:30]
top_10_tag = [
    (tag_dict[key], float(value))
    for key, value in top_10 if value > 0
]
for tag, score in top_10_tag:
    print("tag: {} / score: {}".format(tag, score))
