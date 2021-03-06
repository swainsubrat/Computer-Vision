# -*- coding: utf-8 -*-
"""Assign_3_problem2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rbiH-gxi98aBJ-DwEUqU9L7zCh5lHElf
"""

import torch
from torchvision import datasets
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2


print('Is CUDA available', torch.cuda.is_available())
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'))
#https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#https://www.kaggle.com/abhishek/training-fast-rcnn-using-torchvision

from google.colab import drive
drive.mount('/content/gdrive')

#https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=-WXLwePV5ieP
!git clone https://github.com/pytorch/vision.git
!cd vision
!git checkout v0.3.0

!cp /content/vision/references/detection/utils.py ../usr/lib/python3.6
!cp /content/vision/references/detection/transforms.py ../usr/lib/python3.6
!cp /content/vision/references/detection/coco_eval.py ../usr/lib/python3.6
!cp /content/vision/references/detection/engine.py ../usr/lib/python3.6
!cp /content/vision/references/detection/coco_utils.py ../usr/lib/python3.6

dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#https://github.com/pytorch/vision/blob/master/references/detection/engine.py
#https://github.com/pytorch/vision/blob/master/references/detection/utils.py
#from engine import evaluate,train_one_epoch
from torch import utils

def return_class(class_name):
    classes=['background','person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    return classes.index(class_name)

def return_class_name(class_name):
    classes=['background','person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
    return classes[class_name]

class VOC_Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset, transforms=None):
    self.transforms = transforms
    self.dataset=dataset
  
  def __getitem__(self,ind):
    boxes=[]
    names=[]
    img,dic =self.dataset[ind]
    for i in range(len(dic['annotation']['object'])):
      xmax=int(dic['annotation']['object'][i]['bndbox']['xmax'])
      xmin=int(dic['annotation']['object'][i]['bndbox']['xmin'])
      ymax=int(dic['annotation']['object'][i]['bndbox']['ymax'])
      ymin=int(dic['annotation']['object'][i]['bndbox']['ymin'])
      boxes.append([xmin, ymin, xmax, ymax])
      names.append(return_class(dic['annotation']['object'][i]['name']))
    
    target={}
    target['boxes']=torch.as_tensor(boxes,dtype=torch.float32)
    target['labels']=torch.as_tensor(names,dtype=torch.int64)
    target['image_id'] = torch.tensor(int(dic['annotation']['source']['flickrid']),dtype=torch.int64)
    target['area']=(target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
    target['iscrowd']=torch.zeros((len(dic['annotation']['object']),), dtype=torch.int64)
    return img,target

  def __len__(self):
    return len(self.dataset)

data_path='../data/'
voc_train_tem=datasets.VOCDetection(data_path, year='2007', image_set='train', download=True, transform=transforms.ToTensor())
voc_test_tem=datasets.VOCDetection(data_path, year='2007', image_set='trainval',download=True, transform=transforms.ToTensor())

voc_train=VOC_Dataset(dataset=voc_train_tem)
voc_test=VOC_Dataset(dataset=voc_test_tem)

print(len(voc_test))
img,lab=voc_test_tem[3]
plt.axis('off')
plt.imshow(img.permute(1,2,0))
#plt.savefig('voc6.jpg',bbox_inches='tight',pad_inches=0, dpi=100)

train_loader=torch.utils.data.DataLoader(voc_train,batch_size=2,shuffle=True, num_workers=0)
test_loader=torch.utils.data.DataLoader(voc_test,batch_size=2,shuffle=False, num_workers=0)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_fasterRCNN(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False,progress=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model=get_fasterRCNN(20+1)
model=model.to(dev)

optimizer= torch.optim.Adam(model.parameters(),lr=0.00001)
num_epochs=20

print_freq=1000

for i in range(num_epochs):
  j=0
  for images, targets in train_loader:
    j=j+train_loader.batch_size
    images = list(image.to(dev) for image in images)
    targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
    model=model.train()
    out= model(images,targets)
    l1,l2,l3,l4=out['loss_classifier'],out['loss_box_reg'],out['loss_objectness'],out['loss_rpn_box_reg']
    loss=l1+l2+l3+l4
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if j%print_freq==0:
      print("Epoch: %d, Progress: %d/%d, Loss: %f, loss_classifier: %f, loss_box_reg: %f, loss_objectness: %f, loss_rpn_box_reg: %f" % (i, j,len(train_loader.dataset),float(loss),float(l1),float(l2),float(l3),float(l4)))
  #train_one_epoch(model, optimizer, train_loader, dev, i, print_freq=20)

 # evaluate(model,test_loader,dev)
  torch.save(model, '/content/gdrive/MyDrive/FasterRCNN/model'+str(i)+'.pth')

model=torch.load('/content/gdrive/MyDrive/FasterRCNN/model19.pth')
model=model.eval()
model=model.to(dev)

j=1017
img,lab=voc_test[j]
plt.imshow(img.permute(1,2,0))

img=img.to(dev)
out=model(img.unsqueeze(0))

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (0, 225, 0)
thickness = 2

sample=img.permute(1,2,0).cpu().numpy()
for i in range(len(out[0]['scores'])):
  if out[0]['scores'][i]>0.5:
    sample=cv2.rectangle(sample,(out[0]['boxes'][i][0],out[0]['boxes'][i][1]),(out[0]['boxes'][i][2],out[0]['boxes'][i][3]),color, thickness=2)
    sample=cv2.putText(sample, return_class_name(int(out[0]['labels'][i].cpu())), ((out[0]['boxes'][i][2]+out[0]['boxes'][i][0])/2-15,out[0]['boxes'][i][3]-15), font, fontScale, color, thickness, cv2.LINE_AA, False) 
plt.axis('off')
plt.imshow(cv2.UMat.get(sample))
plt.savefig(str(j)+'.png',bbox_inches='tight',dpi=200,pad_inches=0)

!pip install gluoncv
!pip install mxnet

"""02. Predict with pre-trained Faster RCNN models
==============================================

This article shows how to play with pre-trained Faster RCNN model.

First let's import some necessary libraries:
"""

from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an Faster RCNN model trained on Pascal VOC
# dataset with ResNet-50 backbone. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.
#
# The returned model is a HybridBlock :py:class:`gluoncv.model_zoo.FasterRCNN`
# with a default context of `cpu(0)`.

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms.
# The default behavior is to resize the short edge of the image to 600px.
# But you can feed an arbitrarily sized image.
#
# You can provide a list of image file names, such as ``[im_fname1, im_fname2,
# ...]`` to :py:func:`gluoncv.data.transforms.presets.rcnn.load_test` if you
# want to load multiple image together.
#
# This function returns two results. The first is a NDArray with shape
# `(batch_size, RGB_channels, height, width)`. It can be fed into the
# model directly. The second one contains the images in numpy format to
# easy to be plotted. Since we only loaded a single image, the first dimension
# of `x` is 1.
#
# Please beware that `orig_img` is resized to short edge 600px.

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='biking.jpg')
x, orig_img = data.transforms.presets.rcnn.load_test(im_fname)

######################################################################
# Inference and display
# ---------------------
#
# The Faster RCNN model returns predicted class IDs, confidence scores,
# bounding boxes coordinates. Their shape are (batch_size, num_bboxes, 1),
# (batch_size, num_bboxes, 1) and (batch_size, num_bboxes, 4), respectively.
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:

box_ids, scores, bboxes = net(x)
ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

plt.show()

print(voc_train)

from gluoncv import data, utils
from matplotlib import pyplot as plt


print('Num of training images:', len(voc_train))
print('Num of validation images:', len(voc_test))

train_image, train_label = voc_train[5]
print('Image size (RGB, width, height):', train_image.shape)
print(len(train_label))

bounding_boxes = train_label[:,:4]
print('Num of objects:', bounding_boxes.shape[0])
print('Bounding boxes (num_boxes, x_min, y_min, x_max, y_max):\n',
      bounding_boxes)
