
import torch
import pickle
import torchvision

import numpy as np

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataloader import load_pascal

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_classes = len(('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'))

# help(FasterRCNN)
def create_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def save_checkpoint(epoch, model, optimizer, path='./saved_models/fast_rcnn.pth.tar'):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}

    filename = path
    torch.save(state, filename)

model = create_model(num_classes=num_classes)
model = model.to(device)
# model.eval()
# _, test_loader = load_pascal(batch_size=1)
# for i, (image, boxes, labels, difficulties) in enumerate(test_loader):
#     images = image.to(device)
#     o = model(images)
#     print(o)
#     break

lr = 0.00001
momentum = 0.9
weight_decay = 0.0005
batch_size = 32
epochs     = 10
patience   = 5
min_valid_loss = np.inf
train_los = []
valid_los = []
train_acc  = []
valid_acc  = []
epoch_since_last_improve = 0

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

train_loader, _ = load_pascal(batch_size=batch_size)

for epoch in range(epochs):
    train_loss = 0.0
    for i, (images, targets) in enumerate(train_loader):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        losses.backward()
        optimizer.step()
        
        train_los.append(loss_value)
        if i % 10 == 0 and i != 0:
            print(f"Epoch: [{epoch+1}][{i}/{len(train_loader)}] Loss: {loss_value: .4f}")

        if loss_value < min_valid_loss:
            min_valid_loss = loss_value
            save_checkpoint(epoch, model, optimizer)

with open("./objects/losses", 'wb') as f:
    pickle.dump({"losses": train_los}, f)
