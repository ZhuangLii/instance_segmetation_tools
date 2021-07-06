import numpy as np
import argparse
import json
import os
import os.path as osp
import PIL.Image
import yaml
from labelme import utils
import cv2
from skimage import img_as_ubyte
from os.path import join

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    list_path = os.listdir(json_file)
    print('json_file:', json_file)
    for i in range(0, len(list_path)):
        if ".DS_Store" == list_path[i]:
            continue
        path = os.path.join(json_file, list_path[i])
        if os.path.isfile(path):
            print(path)
            data = json.load(open(path))
            img = utils.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

            captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            # lbl_viz = utils.draw_label(lbl, img, captions)

            out_dir = osp.basename(path).replace('.', '_')
            save_file_name = out_dir

            # ------------------------保存从json中解析出来的图像、label、图像+label-------------------
            if not osp.exists( join( json_file, 'labelme_json')):
                os.mkdir(join(json_file,'labelme_json'))
            labelme_json = join(json_file, 'labelme_json')

            out_dir1 = join(labelme_json, save_file_name)
            if not osp.exists(out_dir1):
                os.mkdir(out_dir1)
            # PIL.Image.fromarray(img).save(join(out_dir1, save_file_name + '_img.png'))
            # PIL.Image.fromarray(lbl).save(join(out_dir1,save_file_name + '_label.png'))
            # PIL.Image.fromarray(lbl_viz).save(join(out_dir1, save_file_name + '_label_viz.png'))
            # ---------------------------------保存label的mask（0 1 2 3）----------------------------
            if not osp.exists(join(json_file, 'mask_png')):
                os.mkdir(join(json_file, 'mask_png'))
            mask_save2png_path = join(json_file, 'mask_png')

            mask_dst = img_as_ubyte(lbl)  # mask_pic
            # print(mask_dst.shape)
            # mask_dst = cv2.cvtColor(mask_dst, cv2.COLOR_BGR2GRAY)
            print('pic2_deep:', mask_dst.dtype)
            cv2.imwrite(join(mask_save2png_path, save_file_name + '_label.png'), mask_dst)
            mask_dst = np.repeat(mask_dst[...,np.newaxis],3,2) 
            a = np.hstack((mask_dst*50, img))
            cv2.imwrite(join(mask_save2png_path, save_file_name + '_label_vis.png'),a )
            # with open(osp.join(out_dir1, 'label_names.txt'), 'w') as f:
            #     for lbl_name in lbl_names:
            #         f.write(lbl_name + '\n')

            # info = dict(label_names=lbl_names)
            # with open(osp.join(out_dir1, 'info.yaml'), 'w') as f:
            #     yaml.safe_dump(info, f, default_flow_style=False)

            # print('Saved to: %s' % out_dir1)


if __name__ == '__main__':
    main()
