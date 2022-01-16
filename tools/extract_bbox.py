import os
import os.path as osp
import glob
import cv2
import xml.etree.ElementTree as ET
from ipdb import set_trace


def main():
    ann_root = "/home/cairujia1/Projects/SimDet/data/xjb/VOC2007/Annotations"
    img_root = "/home/cairujia1/Projects/SimDet/data/xjb/VOC2007/JPEGImages"
    save_root = "/home/cairujia1/Projects/SimDet/data/xjb/bbox"

    images = glob.glob(img_root + "/*.jpg")
    anns = glob.glob(ann_root + "/*.xml")
    if not osp.exists(save_root):
        os.makedirs(save_root)
    cnt_map = {}

    for file_name in anns:
        # load img
        name = file_name.split("/")[-1].split(".")[0]
        img_file = osp.join(img_root, "%s.jpg" % name)
        im = cv2.imread(img_file)
        # set_trace()
        tree = ET.parse(file_name)
        objs = tree.findall("object")
        for ix, obj in enumerate(objs):
            bbox = obj.find("bndbox")
            # Make pixel indexes 0-based
            x1 = int(bbox.find("xmin").text)
            y1 = int(bbox.find("ymin").text)
            x2 = int(bbox.find("xmax").text)
            y2 = int(bbox.find("ymax").text)
            cls = obj.find("name").text.lower().strip()
            if cls not in cnt_map:
                cnt_map[cls] = 0
            else:
                cnt_map[cls] += 1
            im_tmp = im[y1:y2, x1:x2, :]
            if not osp.exists(osp.join(save_root, cls)):
                os.makedirs(osp.join(save_root, cls))
            cv2.imwrite(osp.join(save_root, cls, "%05d.jpg" % cnt_map[cls]), im_tmp)


if __name__ == "__main__":
    main()
