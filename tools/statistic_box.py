import os
import os.path as osp
import glob
import cv2
import xml.etree.ElementTree as ET
from ipdb import set_trace


def main():
    ann_root = "/home/cairujia1/Projects/SimDet/data/xjb/VOC2007/Annotations"
    save_root = "/home/cairujia1/Projects/SimDet/data/xjb/bbox"
    train_path = (
        "/home/cairujia1/data/xinjinbo/VOCdevkit/VOC2007/ImageSets/Main/train.txt"
    )
    train_files = []
    with open(train_path, "r") as f:
        for line in f.readlines():
            train_files.append(line.split("\n")[0])

    # images = glob.glob(img_root + "/*.jpg")
    anns = glob.glob(ann_root + "/*.xml")
    if not osp.exists(save_root):
        os.makedirs(save_root)
    cnt_map = {}

    for file_name in anns:
        # load img
        name = file_name.split("/")[-1].split(".")[0]
        if name in train_files:
            continue
        tree = ET.parse(file_name)
        objs = tree.findall("object")
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            cls = obj.find("name").text.lower().strip()
            if cls not in cnt_map:
                cnt_map[cls] = 0
            else:
                cnt_map[cls] += 1
    print("train", cnt_map)


if __name__ == "__main__":
    main()
