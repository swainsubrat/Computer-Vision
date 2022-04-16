## Faster R-CNN object Detection
In this problem, we are using Pascal VOC 2007 dataset to train our faster rcnn model and detect objects and classifies them into different classes like, persons, vehicles, indoors, etc.

Faster R-CNN achieved much better speeds and a state-of-the-art accuracy. It is worth noting that although future models did a lot to increase detection speeds, few models managed to outperform Faster R-CNN by a significant margin. In other words, Faster R-CNN may not be the simplest or fastest method for object detection, but it is still one of the best performing. Case in point, Tensorflow’s Faster R-CNN with Inception ResNet is their slowest but most accurate model.
At the end of the day, Faster R-CNN may look complicated, but its core design is the same as the original R-CNN: hypothesize object regions and then classify them. This is now the predominant pipeline for many object detection models, including our next one.

![image](https://user-images.githubusercontent.com/47186039/163661062-db588eeb-f35d-45c0-89ef-3623d00cf72c.png)



### Pascal VOC 2007 Datasets

In this dataset, There are 20 classes for classification/detections. The data has been split into 50% for training/validation and 50% for testing. The distributions of images and objects by class are approximately equal across the training/validation and test sets. In total there are 9,963 images, containing 24,640 annotated objects. The datasets is divided into training and validation sets, trainig and validation consists of total of 5011 images, and 4952 are used as testing image dataset.


| **Classes**  | **Images** | **Objects** |
| -------- | ---- | --- |
| Aeroplane |	238 |	306	| 
| Bicycle | 243	| 353	| 
| Bird	| 330	| 486	|
| Boat	|	181	| 290	|
| Bottle |	244 |	505	|
| Bus	|	186	 | 229 |
| Car	|	713	 | 1250	|
| Cat	|	337	 | 376	|
| Chair	|	445	| 798	|
| Cow		| 141	| 259	|
| Diningtable |	200	| 215 |
| Dog	| 421	| 510	|
| Horse	| 287	| 362	|
| Motorbike	| 245	| 339	|
| Person | 2008	| 4690	|
| Pottedplant	|	245 |	514	|
| Sheep	| 96	| 257	|
| Sofa | 229	| 248	|
| Train	|	261	| 297	|
| Tvmonitor	|	256	| 324	|
| **Total**	| **5011**	| **12608**	|

### Results

The best result of our trained model are as follows:
<br>**mAP** score is 69.9%.
<br>**Time/image** is around 198 ms.
<br>
|Classes    |All Points (IOU Threshold=0.5)|All points (IOU Threshold =0.75)|All points (IOU Threshold = 0.9)|
|-----------|------------------------------|--------------------------------|--------------------------------|
|Aeroplanes |0.77                          |0.58                            |0.51                            |
|bicycle    |0.41                          |0.31                            |0.26                            |
|bird       |0.66                          |0.5                             |0.43                            |
|boat       |0.47                          |0.29                            |0.23                            |
|bottle     |0.45                          |0.33                            |0.25                            |
|bus        |0.8                           |0.74                            |0.66                            |
|car        |0.53                          |0.39                            |0.32                            |
|cat        |0.73                          |0.54                            |0.45                            |
|chair      |0.41                          |0.3                             |0.24                            |
|cow        |0.74                          |0.6                             |0.52                            |
|diningtable|0.44                          |0.28                            |0.23                            |
|dog        |0.65                          |0.53                            |0.45                            |
|horse      |0.43                          |0.35                            |0.26                            |
|motorbike  |0.51                          |0.38                            |0.33                            |
|person     |0.68                          |0.53                            |0.47                            |
|pottedplant|0.37                          |0.22                            |0.19                            |
|sheep      |0.68                          |0.57                            |0.5                             |
|sofa       |0.44                          |0.37                            |0.31                            |
|train      |0.76                          |0.65                            |0.58                            |
|tvmonitor  |0.54                          |0.43                            |0.37                            |
|Average    |0.57                          |0.44                            |0.38                            |
<br>
<br>
With a less restrictive IOU threshold (t = 0.5), higher recall values can be obtained with the highest precision. In other words, the detector can retrieve about 66.5% of the total ground truths without any miss detection.
Using t = 0.9, the detector is more sensitive to different confidence values τ. This is
explained by the more accentuated monotonic behavior for this IOU threshold.
Regardless the IOU threshold applied, this detector can never retrieve 100% of the
ground truths (Pr(τ) = 1) for any confidence value τ. This is due to the fact that the
algorithm failed to output any bounding box for one of the ground truths.
When a lower IOU threshold t was considered (t = 0.5 as opposed to t = 0.9), the
AP was considerably increased in both interpolation approaches. This is caused by the
increase in the TP detections, due to lower threshold IOU.
<br>

![Average Precisions (mAP) vs IOU Threshold](https://user-images.githubusercontent.com/47186039/163677748-90314e9f-a2dc-474a-86cf-2243ace693a4.png)


![download](https://user-images.githubusercontent.com/47186039/163685298-49a3d44d-c2e9-4f75-9c88-b1dc0648acd0.png)
