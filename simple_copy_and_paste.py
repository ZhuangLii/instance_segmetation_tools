"""
copy from https://github.com/qq995431104/Copy-Paste-for-Semantic-Segmentation
"""
import imgviz
import cv2
import argparse
import os
import numpy as np
import tqdm

from PIL import Image
from os.path import join
from glob import glob



def save_colored_mask(mask, img, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(".tmp.png")
    lbl_pil = cv2.imread(".tmp.png")
    lbl_pil = cv2.cvtColor(lbl_pil, cv2.COLOR_BGR2RGB)
    out = np.hstack((img, lbl_pil))
    cv2.imwrite(save_path, out)

def save_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    lbl_pil.save(save_path)

def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img


def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)
    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)
    img_main = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
    return img_main


def rescale_src(mask_src, img_src, h, w):
    if len(mask_src.shape) == 3:
        h_src, w_src, c = mask_src.shape
    elif len(mask_src.shape) == 2:
        h_src, w_src = mask_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w), dtype=np.uint8)
    img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad


def Large_Scale_Jittering(mask, img, min_scale=0.1, max_scale=2.0):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask_pad = np.zeros((h, w), dtype=np.uint8)
        img_pad[y:y+h_new, x:x+w_new, :] = img
        mask_pad[y:y+h_new, x:x+w_new] = mask
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y+h, x:x+w, :]
        mask_crop = mask[y:y+h, x:x+w]
        return mask_crop, img_crop


def copy_paste(mask_src, img_src, mask_main, img_main):
    mask_src, img_src = random_flip_horizontal(mask_src, img_src)
    mask_main, img_main = random_flip_horizontal(mask_main, img_main)

    # LSJ， Large_Scale_Jittering
    if args.lsj:
        mask_src, img_src = Large_Scale_Jittering(mask_src, img_src)
        mask_main, img_main = Large_Scale_Jittering(mask_main, img_main)
    else:
        # rescale mask_src/img_src to less than mask_main/img_main's size
        h, w, _ = img_main.shape
        mask_src, img_src = rescale_src(mask_src, img_src, h, w)

    img = img_add(img_src, img_main, mask_src)
    mask = img_add(mask_src, mask_main, mask_src)

    return mask, img


def main(args):
    # input path
    # segclass = os.path.join(args.input_dir, 'SegmentationClass')
    # JPEGs = os.path.join(args.input_dir, 'JPEGImages')
    img_root = args.img_root
    mask_root = args.mask_root
    output_mask_root = join(args.output_dir,"masks")
    output_img_root = join(args.output_dir, "images")
    output_color_mask_root = join(args.output_dir, "color")
    img_names = os.listdir(args.img_root)


    # create output path
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(output_mask_root, exist_ok=True)
    os.makedirs(output_img_root, exist_ok=True)
    os.makedirs(output_color_mask_root, exist_ok=True)

    for _ in tqdm.tqdm(range(args.generate_num)):
        while True:
            img_name = np.random.choice(img_names)
            tmp_img_name = np.random.choice(img_names)
            if img_name != tmp_img_name:
                break
        # get source mask and img
        mask_name = img_name.replace(".jpg", "_json_label.png")
        mask_src = np.asarray(Image.open(join(mask_root, mask_name)), dtype=np.uint8)
        img_src = cv2.imread(join(img_root, img_name))

        # random choice main mask/img
        tmp_img_name = np.random.choice(img_names)
        tmp_mask_name = tmp_img_name.replace(".jpg", "_json_label.png")
        tmp_mask_path = join(mask_root, tmp_mask_name)
        mask_main = np.asarray(Image.open(tmp_mask_path), dtype=np.uint8)
        img_main = cv2.imread(join(img_root, tmp_img_name))

        # Copy-Paste data augmentation
        mask, img = copy_paste(mask_src, img_src, mask_main, img_main)
        mask_filename = "copy_paste_" + tmp_mask_name
        img_filename = mask_filename.replace('.png', '.jpg')
        save_colored_mask(mask, img, join(output_color_mask_root, mask_filename.replace(".png", "_color.png")))
        save_mask(mask, join(output_mask_root, mask_filename))
        cv2.imwrite(os.path.join(output_img_root, img_filename), img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", default="/zhuangjf/gitae/Lightweight-Segmentation/tujiao_train_pic", type=str,
                        help="input image directory")
    parser.add_argument("--mask_root", default="/zhuangjf/gitae/Lightweight-Segmentation/tujiao_train_mask", type=str,
                        help="input image directory")
    parser.add_argument("--output_dir", default="/zhuangjf/gitae/Lightweight-Segmentation/tujiao_copy_paste", type=str,
                        help="output dataset directory")
    parser.add_argument("--generate_num", default= 150, type=int,
                        help="generate image number")
    parser.add_argument("--lsj", default=True, type=bool, help="if use Large Scale Jittering")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)