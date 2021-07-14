import os
import sys
import json
import time
import cv2
import argparse
import detectron2
import numpy as np

from tqdm import tqdm
from glob import glob
from PIL import Image
from os.path import join

from detectron2.utils.visualizer import Visualizer


class DataTransformer:
    def __init__(self, mask_path, img_path, cls_dict):
        self.data = {
            "version": "4.5.7",
            "flags": {},
            "shapes": [],
            "imagePath": img_path,
            "imageData": None,
            "imageHeight": 0,
            "imageWidth": 0,
        }
        self.cls_dict = cls_dict
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


    def _getShapes(self):
        shapes = []
        poly_point_dict = self._mask2Poly()
        for cls, poly_points in poly_point_dict.items():
            for poly_point in poly_points:
                shapes.append({
                    "label": cls,
                    "points": poly_point,
                    "ground_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                })
        self.data["shapes"] = shapes

    def _getWH(self):
        h, w = self.mask.shape[:2]
        self.data["imageHeight"] = h
        self.data["imageWidth"] = w

    def _mask2Poly(self):
        """
        :param mask_path: mask_path mask format like [[0, 0, 0, 1, 1],
                                                        [0, 0, 0, 1, 1],
                                                        [2, 2, 0, 0, 0],
                                                        [2, 2, 0, 0, 0]]
        :return: polygon points dict {"cls1": [[x1, y1,], [x2, y2], ...](point1),
                                            [[x1, y1,], [x2, y2], ...](point2), ...
                                     "cls2": ...,
                                    ...}
        """

        ## init poly point dict
        poly_point_dict = {}
        for k, v in self.cls_dict.items():
            if k == "__background__" or (self.mask != v).all():
                continue
            poly_point_dict[k] = []
            cls_mask = (self.mask == v).astype(np.uint8)
            ## add points in poly_point_dict
            poly_points, _ = cv2.findContours(cls_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for points in poly_points:
                ##  points length <= 2 should be removed
                if len(points) <= 2:
                    continue
                poly_point_dict[k].append(points.squeeze().tolist())
        return poly_point_dict

    def mask2Json(self, output_path):
        self._getWH()
        self._getShapes()
        with open(output_path, "w") as f:
            context = json.dumps(self.data, indent=4)
            f.write(context)
        return self.data


def main(args):
    mask_root = args.mask_root
    output_root = args.output_root
    os.makedirs(output_root, exist_ok=True)
    category_ids = {
        "__background__": 0,
        "jt": 1,
        "qp": 2
    }
    mask_paths = glob(join(mask_root, "*.png"))
    for mask_path in tqdm(mask_paths):
        json_name = mask_path.split("/")[-1].replace(".png", ".json")
        output_path = join(join(output_root, json_name))
        img_path = mask_path.replace("tujiao_train_mask", "tujiao_train_pic")
        img_path = img_path.replace(".png", ".jpg")
        img_path = img_path.replace("_json_label", "")
        data_transformer = DataTransformer(mask_path, img_path, category_ids)
        data = data_transformer.mask2Json(output_path)
        """check
        img = cv2.imread(img_path)
        for x in data["shapes"]:
            v = Visualizer(img)
            poly = np.asarray(x["points"])
            color = detectron2.utils.colormap.random_color()
            color = color / 255
            img = v.draw_polygon(poly, color, alpha=0.3)
            img = img.get_image()
        """



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_root", default="/zhuangjf/gitae/Lightweight-Segmentation/tujiao_copy_paste/masks", type=str,
                        help="input image directory")
    parser.add_argument("--output_root", default="/zhuangjf/gitae/Lightweight-Segmentation/tujiao_copy_paste/labelme_json", type=str,
                        help="output dataset directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)