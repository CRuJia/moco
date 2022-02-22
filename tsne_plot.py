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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# def read_img(img_path, query_size=128):
#     im = cv2.resize(
#         imread(img_path), (query_size, query_size), interpolation=cv2.INTER_LINEAR
#     )

#     im = im.astype(np.float32, copy=False)
#     im /= 255.0  # Convert range to [0,1]
#     pixel_means = [0.485, 0.456, 0.406]
#     pixel_stdens = [0.229, 0.224, 0.225]

#     # normalize manual
#     im -= pixel_means  # Minus mean
#     im /= pixel_stdens  # divide by stddev

#     query = torch.from_numpy(im)
#     query = query.unsqueeze(0)
#     query = query.permute(0, 3, 1, 2).contiguous()
#     return query

def tsne_plot(feature, label, classes, title ='tsne', file_name = 'tmp.jpg'):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(feature)
    # set_trace()
    data = result
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i,0], data[i,1], str(classes[label[i]]),
            color=plt.cm.Set1(label[i]/10.),
            fontdict={'weight':'bold','size':9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig(file_name)


def compute_query_feature(model, val_loader, classes):
    cls_vec = defaultdict(list)
    features = []
    labels = []
    ims = []
    pixel_means = [0.485, 0.456, 0.406]
    pixel_stdens = [0.229, 0.224, 0.225]
    for i, (images, cls) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        query = model(images)
        query = query.mean(-1).mean(-1).squeeze(0).detach().cpu().numpy()
        features.append(query)
        labels.append(int(cls.detach().cpu()))

        im = images.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        im *= pixel_stdens # divide by stddev
        im += pixel_means # Minus mean    
        im*=255
        im = im.astype(np.int16)
        ims.append(im)
    
    feature = np.array(features)
    label = np.array(labels)
    ims = np.array(ims)

    return feature, label, ims

def main():
    model = MoCo(models.__dict__["resnet50"], mlp=True, K=8192)
    checkpoint = torch.load("models/resnet50/xjb_video2/checkpoint_0199_1.378461.pth.tar")
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
    feature, label, ims = compute_query_feature(backbone, val_loader, val_dataset.classes)
    print(len(feature))
    print(len(label))
    print(len(ims))
    # set_trace()

    tsne_plot(feature, label, val_dataset.classes,title="video3", file_name="video3.jpg")

if __name__ == "__main__":
    main()
