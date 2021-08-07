import os
import cv2
import pandas as pd
import os
import glob
import pandas as pd
import numpy as np
import argparse
import xml.etree.ElementTree as ET

os.chdir('../')

content_dct={}
with open('config.txt') as f:
    content=f.readlines()
    for i in content:
        var,val = i.split('=')
        content_dct[var.strip()]=val.strip()



df=pd.read_csv(content_dct['label_path']+'labels.csv')


cols = df.select_dtypes(include=[np.float64]).columns
df[cols] = df[cols].astype(np.float32)

#Pickup a random image number
img_num = np.random.randint(0, df.shape[0])

#Read the image
img_file = df.loc[img_num,'filename']
img = cv2.imread(content_dct['image_path'] + '/' + img_file)



#Find all rows which have same file name
rows_with_file = df[df['filename'] == img_file].index.tolist()

#Draw rectangle(s) as per bounding box information
for i in rows_with_file:

    #Get bounding box
    xmin, ymin, xmax, ymax = df.loc[i, ['xmin', 'ymin', 'xmax', 'ymax']]

    #Get Label
    label = df.loc[i, 'class']
    #Add bounding box
    cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0,255,0), 2)
    # Add text
    cv2.putText(img, label, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

cv2.imshow('Image', img)
cv2.waitKey(0)