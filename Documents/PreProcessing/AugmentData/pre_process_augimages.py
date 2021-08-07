
import os
import cv2
import pandas as pd
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import re
os.chdir('../')
content_dct={}
with open('config.txt') as f:
    content=f.readlines()
    for i in content:
        var,val = i.split('=')
        content_dct[var.strip()]=val.strip()

#load the labels as dictionary
label_dct={}
for ind,lab in enumerate(content_dct['class_name'].split(',')):
    label_dct[lab]=ind

try:
    int(content_dct['width'])
    int(content_dct['height'])
    df= pd.read_csv(content_dct['aug_label_path']+content_dct['aug_file_name'], sep=",",dtype={content_dct['xmin']:float,
                                                             content_dct['ymin']:float,content_dct['xmax']:float,
                                                             content_dct['ymax']:float})
    flag=1
except:
    df= pd.read_csv(content_dct['aug_label_path']+content_dct['aug_file_name'], sep=",",dtype={content_dct['width']:int,content_dct['height']:int,
                                              content_dct['xmin']:float,content_dct['ymin']:float,
                                              content_dct['xmax']:float,content_dct['ymax']:float})

new_df=df.copy(deep=True)

#Map the labels to integer
try:
    new_df=new_df.loc[new_df['class'].isin(label_dct.keys())]
    new_df['class']=new_df[content_dct['label']].map(label_dct)
except:
    print('The column label mentioned in the config file doesn\'t exist in the csv file')
    raise

#converting the VOC points into yolo points
if(flag==0):
    try:
        new_df['x']=((df[content_dct['xmin']]+df[content_dct['xmax']])/2)/df[content_dct['width']]
        new_df['y']=((df[content_dct['ymin']]+df[content_dct['ymax']])/2)/df[content_dct['height']]
        new_df['w']=(df[content_dct['xmax']]-df[content_dct['xmin']])/df[content_dct['width']]
        new_df['h']=(df[content_dct['ymax']]-df[content_dct['ymin']])/df[content_dct['height']]
    except:
        print('The column mentioned in the config file doesn\'t exist in the csv file')
        raise
else:
    new_df['x']=((df[content_dct['xmin']]+df[content_dct['xmax']])/2)/int(content_dct['width'])
    new_df['y']=((df[content_dct['ymin']]+df[content_dct['ymax']])/2)/int(content_dct['height'])
    new_df['w']=(df[content_dct['xmax']]-df[content_dct['xmin']])/int(content_dct['width'])
    new_df['h']=(df[content_dct['ymax']]-df[content_dct['ymin']])/int(content_dct['height'])

final_df=pd.DataFrame({'filename':new_df['filename'],'label':new_df['class'],'x':new_df['x'],'y':new_df['y'],'w':new_df['w'],'h':new_df['h']})
unique_image=final_df['filename'].unique()

#writting the txt file with the images of images in the specified folder
for i in unique_image:
    aug_file_name=i.split('.')[0]+'.txt'
    row_series=final_df.loc[final_df['filename']==i,'label':'h']
    if not os.path.isfile(content_dct['aug_images_path'] +'Aug_images/'+ aug_file_name):
        try:
            with open(content_dct['aug_images_path']+'Aug_images/'+aug_file_name,'w') as f:
                row_series.to_string(f,header=False,index=False)
        except:
            print('please check the txt file path')
            raise