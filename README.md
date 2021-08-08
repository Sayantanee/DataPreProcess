1. Place all the images in .\PreProcessing\01_data\images
2. Place all the xmls in .\PreProcessing\01_data\annotations
Navigate to PreProcessing\DataPreProcessing\
======================================================
3. "date_processing.py" converts xml annotations to csv format and writes "labels.csv" in the "PreProcessing\DataPreProcessing" location. 
   It also converts each annotation.xml file into the corresponding Yolo txt format and writes in ".\PreProcessing\01_data\images"
4. "train_test_split" splits the images into train and test sets and create and populate new "train" and "test" directories in " .\PreProcessing\"

Navigate to PreProcessing\AugmentData\
======================================================

5. "augment_data" detects all images with 90% head class and augments those images and stores them in ".\PreProcessing\AugmentData\Aug_images"
6. "pre_process_augimages" writes yolo txt annotations in ".\PreProcessing\AugmentData\Aug_images" for the augmented images

Navigate to PreProcessing\DataVisualize\
======================================================
Randomly picks images and visualize with BB

config.txt meanings
======================================================


file_name = labels.csv //DO NOT CHANGE//

aug_file_name=all_labels.csv //DO NOT CHANGE//

width = 416 //DO NOT CHANGE//

height = 416 //DO NOT CHANGE//

xmin = xmin //DO NOT CHANGE//

ymin = ymin //DO NOT CHANGE//

xmax = xmax //DO NOT CHANGE//

ymax = ymax //DO NOT CHANGE//

label = class //DO NOT CHANGE//

class_name = head,helmet,person //DO NOT CHANGE//

output_csv_name = yolo_output_voc //DO NOT CHANGE//

image_path=C:\Users\sense\Documents\Final_Capstone_Project\01_data\images\  //SHOULD BE THE COMPLETE PATH OF THE IMAGE DIRECTORY//

xml_path=C:\Users\sense\Documents\Final_Capstone_Project\01_data\annotations\ //SHOULD BE THE COMPLETE PATH OF THE ANNOTATION DIRECTORY//

label_path=C:\Users\sense\Documents\Final_Capstone_Project\DataPreProcessing\ //SHOULD BE THE COMPLETE PATH OF THE "PreProcessing\DataPreProcessing\" DIRECTORY//

aug_label_path=C:\Users\sense\Documents\Final_Capstone_Project\AugmentData\ //SHOULD BE THE COMPLETE PATH OF THE "PreProcessing\AugmentData\" DIRECTORY//

train_test=C:\Users\sense\Documents\Final_Capstone_Project\ //SHOULD BE THE COMPLETE PATH OF THE "PreProcessing\" DIRECTORY (ROOT)//

aug_images_path=C:\Users\sense\Documents\Final_Capstone_Project\AugmentData\ //SHOULD BE THE COMPLETE PATH OF THE "PreProcessing\AugmentData\" DIRECTORY//
