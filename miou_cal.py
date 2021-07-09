import json
import cv2
import os
import argparse
import numpy as np

from PIL import Image
from os.path import join
from glob import glob



class Evaluator(object):
    def __init__(self, class_txt, pred_root, label_root):
        with open(class_txt, "r") as f:
            lines = f.readlines()
        self.classes = ['back_ground'] + [x.rstrip() for x in lines]
        self.num_class = len(self.classes)
        self.confusion_matrix = np.zeros((self.num_class,)*2)  # shape:(num_class, num_class)
        self.pred_img_paths = sorted(glob(join(pred_root, "*.jpg")))
        self.label_img_paths = sorted(glob(join(label_root, "*.png")))

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        print('-----------Acc of each classes-----------')
        for i,x in enumerate(self.classes):
            print(x+ "          : %.6f" % (Acc[i] * 100.0), "%\t")

        Acc = np.nanmean(Acc)
        print("mean Pixel Acc: {:.2f}".format(Acc * 100))
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        # print MIoU of each class
        print('-----------IoU of each classes-----------')
        for i, x in enumerate(self.classes):
            print(x+"          : %.6f" % (MIoU[i] * 100.0), "%\t")
        MIoU = np.nanmean(MIoU)
        print("mIOU: {:.2f}".format(MIoU * 100))
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >=0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def show_results(self):
        self.reset()
        for label_img_path, pred_img_path in zip(self.label_img_paths, self.pred_img_paths):
            label = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
            pred_img_path = label_img_path.replace("_json_label.png", ".jpg")
            pred_img_path = pred_img_path.replace("_val_", "_pred_")
            pred = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
            self.add_batch(label, pred)
        self.Pixel_Accuracy_Class()
        self.Mean_Intersection_over_Union()

def main(args):
    evalater = Evaluator(args.info_txt, args.seg_res_root, args.gt_root)
    evalater.show_results()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root",
                        default="/zhuangjf/gitae/Lightweight-Segmentation/tujiao_val_mask",
                        type=str,
                        help="ground truth mask root")
    parser.add_argument("--seg-res-root",
                        default="/zhuangjf/gitae/Lightweight-Segmentation/tujiao_pred_mask",
                        type=str,
                        help="segmentation mask root")
    parser.add_argument("--info-txt",
                        default="/zhuangjf/gitae/Lightweight-Segmentation/info.txt",
                        type=str,
                        help="infomation txt")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
