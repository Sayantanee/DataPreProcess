#pip3 install opencv-python

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


xml_path=content_dct['xml_path']
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[5][0].text),
                         int(member[5][1].text),
                         int(member[5][2].text),
                         int(member[5][3].text)
                         )
                xml_list.append(value)
            except Exception as e:
                pass

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
xml_df=xml_to_csv(xml_path)
xml_df.to_csv(content_dct['label_path']+'labels.csv', index=None)


#load the labels as dictionary
label_dct={}
for ind,lab in enumerate(content_dct['class_name'].split(',')):
    label_dct[lab]=ind

try:
    int(content_dct['width'])
    int(content_dct['height'])
    df= pd.read_csv(content_dct['label_path']+content_dct['file_name'], sep=",",dtype={content_dct['xmin']:float,
                                                             content_dct['ymin']:float,content_dct['xmax']:float,
                                                             content_dct['ymax']:float})
    flag=1
except:
    df= pd.read_csv(content_dct['label_path']+content_dct['file_name'], sep=",",dtype={content_dct['width']:int,content_dct['height']:int,
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
    file_name=i.split('.')[0]+'.txt'
    row_series=final_df.loc[final_df['filename']==i,'label':'h']
    if not os.path.isfile(content_dct['image_path'] + file_name):
        try:
            with open(content_dct['image_path']+file_name,'w') as f:
                row_series.to_string(f,header=False,index=False)
        except:
            print('please check the txt file path')
            raise