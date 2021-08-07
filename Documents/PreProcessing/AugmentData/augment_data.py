import imgaug as ia
ia.seed(1)
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
import xml.etree.ElementTree as ET
import shutil

os.chdir('../')

content_dct={}
with open('config.txt') as f:
    content=f.readlines()
    for i in content:
        var,val = i.split('=')
        content_dct[var.strip()]=val.strip()


all_img_path=content_dct['image_path']
os.chdir(all_img_path)

images = []
for index, file in enumerate(glob.glob('*.png')):
    images.append(imageio.imread(file))

all_xml_path=content_dct['xml_path']
os.chdir(all_xml_path)

# Function that will extract column data for our CSV file
def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
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
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
labels_df = xml_to_csv('./')

#Now only select images where head class is more than 90%
grouped = labels_df.groupby('filename')
head_imgs=[]
# we can pull each group with get_group() using the filename
for img in labels_df['filename'].unique():
    group_df=pd.DataFrame()
    group_df = grouped.get_group(img)
    group_df = group_df.reset_index()
    group_df = group_df.drop(['index'], axis=1)
    if 'head' in group_df['class'].value_counts():
        if group_df['class'].value_counts(normalize=True)['head']>0.9:
            head_imgs.append(img)
aug_df=labels_df[labels_df['filename'].isin(head_imgs)]
# get bounding boxes coordinates from grouped data frame and write into array
bb_array = aug_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
# pass the array of bounding boxes coordinates to the imgaug library
bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=images[0].shape)


def resize_imgaug(df, images_path, aug_images_path, image_prefix):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    grouped = df.groupby('filename')

    for filename in df['filename'].unique():
        #   Get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)

        aug_bbs_xy = pd.concat([aug_bbs_xy, group_df])
    # return dataframe with updated images and bounding boxes annotations
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy


# function to convert BoundingBoxesOnImage object into DataFrame
def bbs_obj_to_df(bbs_object):
    #     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
    #     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    return df_bbs

# overwrite the labels.csv with updated info
resized_images_df = resize_imgaug(aug_df, all_img_path, 'aug', '')


# This setup of augmentation parameters will pick two of four given augmenters and apply them in random order
aug = iaa.SomeOf(2, [
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.Affine(rotate=(-60, 60)),
    iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
    iaa.Fliplr(1),
    iaa.Multiply((0.5, 1.5))
])


def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
                              )
    grouped = df.groupby('filename')

    for filename in df['filename'].unique():
        #   get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        #   read the image
        image = imageio.imread(images_path + filename)
        #   get bounding boxes coordinates and write into array
        bb_array = group_df.drop(['filename', 'width', 'height', 'class'], axis=1).values
        #   pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        #   apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
        #   disregard bounding boxes which have fallen out of image pane
        bbs_aug = bbs_aug.remove_out_of_image()
        #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()

        #   don't perform any actions with the image if there are no bounding boxes left in it
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass

        #   otherwise continue
        else:
            #   write augmented image to a file
            imageio.imwrite(aug_images_path + image_prefix + filename, image_aug)
            #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
            for index, _ in info_df.iterrows():
                info_df.at[index, 'width'] = image_aug.shape[1]
                info_df.at[index, 'height'] = image_aug.shape[0]
            #   rename filenames by adding the predifined prefix
            info_df['filename'] = info_df['filename'].apply(lambda x: image_prefix + x)
            #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
            #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
            #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])

            # return dataframe with updated images and bounding boxes annotations
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy
aug_images_path=content_dct['aug_images_path']
os.chdir(aug_images_path)
if os.path.isdir('./Aug_images/'):
    shutil.rmtree('./Aug_images/')
os.makedirs('./Aug_images/')

augmented_images_df = image_aug(resized_images_df, all_img_path, aug_images_path+'Aug_images/', 'aug', aug)
# Concat resized_images_df and augmented_images_df together and save in a new all_labels.csv file
#all_labels_df = pd.concat([resized_images_df, augmented_images_df])
augmented_images_df=augmented_images_df.dropna(axis=0, subset=['xmin', 'ymin', 'xmax', 'ymax'])
augmented_images_df.to_csv(aug_images_path+'all_labels.csv', index=False)