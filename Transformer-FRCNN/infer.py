import os
import gluoncv
from gluoncv import model_zoo, data, utils

from matplotlib import pyplot as plt

net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
base_dir = "./JPEGImages/"
images = os.listdir(base_dir)

images = [base_dir + image for image in images]

lst_val = []

class_names=['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
def solve(labels, scores, bboxes, class_names, thresh=0.5):
    scores = scores.asnumpy()
    labels = labels.asnumpy()
    bboxes = bboxes.asnumpy()
    items = []
    for i, bbox in enumerate(bboxes):
        # print(i)
        if scores is not None and scores.flat[i] < thresh:
            continue
        if labels is not None and labels.flat[i] < 0:
            continue
        cls_id = int(labels.flat[i]) if labels is not None else -1
        # if cls_id not in colors:
        #     if class_names is not None:
        #         colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
        #     else:
        #         colors[cls_id] = (random.random(), random.random(), random.random())
        xmin, ymin, xmax, ymax = [int(x) for x in bbox]
        # rect = plt.Rectangle((xmin, ymin), xmax - xmin,
        #                      ymax - ymin, fill=False,
        #                      edgecolor=colors[cls_id],
        #                      linewidth=linewidth)
        # ax.add_patch(rect)
        if class_names is not None and cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = str(cls_id) if cls_id >= 0 else ''
        score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''
        # if class_name or score:
        #     ax.text(xmin, ymin - 2,
        #             '{:s} {:s}'.format(class_name, score),
        #             bbox=dict(facecolor=colors[cls_id], alpha=0.5),
        #             fontsize=fontsize, color='white')
        items.append((score, cls_id, class_name, xmin, ymin, xmax, ymax))
    
    return items

final_lst = []
iter = 0
from tqdm import tqdm 
for image in tqdm(images):
    x, orig_img = data.transforms.presets.rcnn.load_test(image)
    box_ids, scores, bboxes = net(x)
    items = solve(box_ids[0], scores[0], bboxes[0], net.classes)

    ax = utils.viz.plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)
    plt.show()

    # name = image.split("/")[-1]

    # final_lst.append((name, items))
    # iter += 1


from pprint import pprint

pprint(final_lst)
