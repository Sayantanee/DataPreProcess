import os
import fnmatch
import numpy as np
import shutil
import random

os.chdir('../')

content_dct={}
with open('config.txt') as f:
    content=f.readlines()
    for i in content:
        var,val = i.split('=')
        content_dct[var.strip()]=val.strip()

root_dir = content_dct['image_path']

allFileNames=[]

test_ratio = 0.15


allImgFileNames = fnmatch.filter(os.listdir(content_dct['image_path']), '*.png')
for img in allImgFileNames:
    allFileNames.append(os.path.splitext(img)[0])

np.random.shuffle(allFileNames)
train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - test_ratio))])

if os.path.isdir(content_dct['train_test']+'/train/'):
    shutil.rmtree(content_dct['train_test']+'/train/')
os.makedirs(content_dct['train_test']+'/train/')
if os.path.isdir(content_dct['train_test']+'/test/'):
    shutil.rmtree(content_dct['train_test']+'/test/')
os.makedirs(content_dct['train_test']+'/test/')


for name in train_FileNames:
    shutil.copy(root_dir+name+'.png', content_dct['train_test']+'/train/')
    shutil.copy(root_dir+name + '.txt', content_dct['train_test']+'/train/')

for name in test_FileNames:
    shutil.copy(root_dir+name + '.png', content_dct['train_test']+'/test/')
    shutil.copy(root_dir+name + '.txt', content_dct['train_test']+'/test/')




