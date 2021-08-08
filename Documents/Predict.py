# Importing required libraries, obviously
import cv2
from PIL import Image
import numpy as np
import os
import pandas as pd
from pathlib import Path


content_dct={}
with open('config_predict.txt') as f:
    content=f.readlines()
    for i in content:
        var,val = i.split('=')
        content_dct[var.strip()]=val.strip()


confidenceThreshold = 0.2
NMSThreshold = 0.4

modelConfiguration = 'yolov3_testing.cfg'
modelWeights = 'yolov3_training_final.weights'


labelsPath = 'classes.txt'
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

os.chdir(content_dct['test_img_path'])
image = Image.open('hard_hat_workers20.png')
img_name=image.filename

image = np.array(image.convert('RGB'))
#image = cv2.imread(image)
(H, W) = image.shape[:2]
layerName = net.getLayerNames()
layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#layerName : ['yolo_82', 'yolo_94', 'yolo_106']

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layersOutputs = net.forward(layerName)

boxes = []
confidences = []
classIDs = []

for output in layersOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > confidenceThreshold:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype('int')
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)


outputs = {}
# Apply Non Maxima Suppression
detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

df=pd.read_csv(content_dct['label_path']+'labels.csv')

cols = df.select_dtypes(include=[np.float64]).columns
df[cols] = df[cols].astype(np.float32)

#Find all rows which have same file name
rows_with_file = df[df['filename'] == img_name].index.tolist()

print("Prediction --------------------->")
if len(detectionNMS) > 0:
    for i in detectionNMS.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        print("Class:", labels[classIDs[i]])
        print("Coordinates:", x, y, x + w, y + h)
        cv2.rectangle(image, (x, y), (x + w, y + h),  (255,0,0), 2)
        text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

print("Ground Truth------------------->")
#Draw rectangle(s) as per bounding box information
for i in rows_with_file:

    #Get bounding box
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = df.loc[i, ['xmin', 'ymin', 'xmax', 'ymax']]
    # Get Label
    label = df.loc[i, 'class']
    print("Class:",label)
    print("Coordinates:",xmin_gt, ymin_gt, xmax_gt, ymax_gt)

    #Add bounding box
    cv2.rectangle(image, (xmin_gt,ymin_gt), (int(xmax_gt), int(ymax_gt)), (0,0,255), 2)
    # Add text
    cv2.putText(image, label, (xmin_gt, ymin_gt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

cv2.imshow('Image', image)
cv2.waitKey(0)