from dis import dis
from doctest import DocFileSuite
import re
from turtle import back
import torch
import cv2
import torch.nn as nn
import torchvision.models as models
from moco.builder import MoCo
from scipy.misc import imread
import numpy as np
from ipdb import set_trace
from torchvision import transforms
import torchvision.datasets as datasets
from collections import defaultdict


def read_img(img_path, query_size=128):
    im = cv2.resize(
        imread(img_path), (query_size, query_size), interpolation=cv2.INTER_LINEAR
    )

    im = im.astype(np.float32, copy=False)
    im /= 255.0  # Convert range to [0,1]
    pixel_means = [0.485, 0.456, 0.406]
    pixel_stdens = [0.229, 0.224, 0.225]

    # normalize manual
    im -= pixel_means  # Minus mean
    im /= pixel_stdens  # divide by stddev

    query = torch.from_numpy(im)
    query = query.unsqueeze(0)
    query = query.permute(0, 3, 1, 2).contiguous()
    return query


def eval(model, val_loader, classes):
    cls_vec = defaultdict(list)
    for i, (images, cls) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        cls = int(cls)
        query = model(images)
        query = query.mean(-1).mean(-1)
        cls_vec[cls].append(query)
        # if i > 5:
        #     break
    intra_dist = 0.0
    inter_sim = 0.0
    cls_means = []
    for cls in cls_vec:
        vec = torch.cat(cls_vec[cls])
        cls_mean = vec.mean(0)
        cls_means.append(cls_mean)
        cls_mean = cls_mean.unsqueeze(0).repeat(len(cls_vec[cls]), 1)
        dist = 1 - torch.cosine_similarity(vec, cls_mean)
        dist = float(dist.mean())
        print("%10s" % classes[cls], "%5f" % dist)
        intra_dist += dist
    cls_num = len(cls_means)
    for i in range(cls_num):
        for j in range(i + 1, cls_num):
            sim = float(
                torch.cosine_similarity(
                    cls_means[i].unsqueeze(0), cls_means[j].unsqueeze(0)
                )
            )
            inter_sim += sim
            # print("%10s and %10s similarity is :%5f" % (classes[i], classes[j], sim))

    inter_sim /= (cls_num * (cls_num - 1)) / 2
    return intra_dist, inter_sim


def main():
    model = MoCo(models.__dict__["resnet50"], mlp=True, K=8192)
    checkpoint = torch.load("models/resnet50/xjb_video/checkpoint_0199.pth.tar")
    model.load_state_dict(checkpoint["state_dict"])

    backbone = nn.Sequential(
        model.encoder_q.conv1,
        model.encoder_q.bn1,
        model.encoder_q.relu,
        model.encoder_q.maxpool,
        model.encoder_q.layer1,
        model.encoder_q.layer2,
        model.encoder_q.layer3,
        model.encoder_q.layer4,
    )
    backbone.cuda()
    backbone.eval()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augmentation = [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize,
    ]
    traindir = "data/xjb/bbox"
    val_dataset = datasets.ImageFolder(traindir, transforms.Compose(augmentation))

    train_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    intra_dist, inter_dist = eval(backbone, val_loader, val_dataset.classes)
    print("intra_dist", intra_dist)
    print("inter_dist", inter_dist)


if __name__ == "__main__":
    # path = "door1.jpg"
    # read_img(path)
    main()
