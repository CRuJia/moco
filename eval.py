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


def main():
    model = MoCo(models.__dict__["resnet50"], mlp=True, K=8192)
    checkpoint = torch.load("models/resnet50/xjb_video/checkpoint_0100.pth.tar")
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
    cls_vec = defaultdict(list)
    for i, (images, cls) in enumerate(val_loader):
        cls = int(cls)
        query = backbone(images)
        query = query.mean(-1).mean(-1)
        cls_vec[cls].append(query)
        # if i > 5:
        #     break
    dist_sum = 0.0
    for cls in cls_vec:
        vec = torch.cat(cls_vec[cls])
        cls_mean = vec.mean(0)
        cls_mean = cls_mean.unsqueeze(0).repeat(len(cls_vec[cls]), 1)
        dist = 1 - torch.cosine_similarity(vec, cls_mean)
        dist = float(dist.mean())
        print(val_dataset.classes[cls], dist)
        dist_sum += dist
    print("dist_sum: ", dist_sum)

    # set_trace()

    # path = "door1.jpg"
    # query = read_img(path)
    # query = query.mean(-1).mean(-1)
    # set_trace()


if __name__ == "__main__":
    # path = "door1.jpg"
    # read_img(path)
    main()

    """
    199
    cabinet 0.06566110998392105
chair 0.07336632162332535
desk 0.07596726715564728
door 0.100099578499794
handrail 0.09504374116659164
panel 0.06813354045152664
washbasin 0.08650251477956772
window 0.10193048417568207
dist_sum:  0.6667045578360558
    """
