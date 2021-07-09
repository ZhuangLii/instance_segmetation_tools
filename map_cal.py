import cv2
import os
import math
import numpy as np
import json
import argparse
import shutil
import operator

from tqdm import tqdm
from labelme import utils
from os.path import join
from collections import defaultdict
from multiprocessing import Pool
from glob import glob
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

white = (255,255,255)
light_blue = (255,200,100)
green = (0,255,0)
light_red = (30,30,255)
pink = (203,192,255)
black = [0, 0, 0]
bottom_border = 60
margin = 10
font = cv2.FONT_HERSHEY_PLAIN
fontScale = 1
lineType = 1

class mAPevaluater:
    def __init__(self, gt_root, det_res_root, image_root, ignore_list=[], min_overlap=0.5, quiet=False, draw_plot=False, show_animation=False):
        self.gt_root = gt_root
        self.det_res_root = det_res_root
        self.image_root = image_root
        self.gt_json_files = []
        self.ignore_list = ignore_list
        self.min_overlap = min_overlap
        self.gt_counter_per_class = defaultdict(int)
        self.counter_images_per_class = defaultdict(int)
        self.tmp_file_path = ".temp_files"
        if not os.path.exists(self.tmp_file_path):
            os.makedirs(self.tmp_file_path)
        self.output_files_path = "output"
        if not os.path.exists(self.output_files_path):
            os.makedirs(self.output_files_path)
        self.show_animation = show_animation
        self.sum_AP = 0.0
        self.quiet = quiet
        self.draw_plot = draw_plot
        self.ap_dictionary = {}
        self.lamr_dictionary = {}


    def _log_average_miss_rate(self, prec, rec, num_images):
        """
            log-average miss rate:
                Calculated by averaging miss rates at 9 evenly spaced FPPI points
                between 10e-2 and 10e0, in log-space.

            output:
                    lamr | log-average miss rate
                    mr | miss rate
                    fppi | false positives per image
        """
        # if there were no detections of that class
        if prec.size == 0:
            lamr = 0
            mr = 1
            fppi = 0
            return lamr, mr, fppi
        fppi = (1 - prec)
        mr = (1 - rec)
        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp = np.insert(mr, 0, 1.0)
        # Use 9 evenly spaced reference points in log-space
        ref = np.logspace(-2.0, 0.0, num=9)
        for i, ref_i in enumerate(ref):
            # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
            j = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i] = mr_tmp[j]
        # log(0) is undefined, so we use the np.maximum(1e-10, ref)
        lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
        return lamr, mr, fppi

    def _voc_ap(self, rec, prec):
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre


    def _txt_file_to_list(self, path):
        with open(path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        return content


    def _dump_gt_json(self):
        print("ground truth json dumping ...")
        gt_txt_files = sorted(glob(join(self.gt_root, "*.txt")))
        for txt_file in tqdm(gt_txt_files):
            file_name = txt_file.split("/")[-1][:-4]
            bounding_boxes = []
            already_seen_classes = set()
            line_list = self._txt_file_to_list(txt_file)

            for line in line_list:
                class_name, left, top, right, bottom, _difficult = line.split()
                is_difficult = False if _difficult == '0' else True
                if class_name in self.ignore_list:
                    continue
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": is_difficult})
                if not is_difficult:
                    self.gt_counter_per_class[class_name] += 1
                    if class_name not in already_seen_classes:
                        self.counter_images_per_class[class_name] += 1
                    already_seen_classes.add(class_name)

            new_tmp_json_file = join(self.tmp_file_path, file_name + "_ground_truth.json")
            self.gt_json_files.append(new_tmp_json_file)
            with open(new_tmp_json_file, "w") as f:
                json.dump(bounding_boxes, f)
        self.gt_classes = sorted(list(self.gt_counter_per_class.keys()))
        self.n_classes = len(self.gt_classes)


    def _dump_det_res_json(self):
        print("detection results dumping ...")
        det_res_txt_files = sorted(glob(join(self.det_res_root, "*.txt")))
        for class_index, class_name in tqdm(enumerate(self.gt_classes)):
            bounding_boxes = []
            for txt_file in det_res_txt_files:
                file_name = txt_file.split("/")[-1][:-4]
                line_list = self._txt_file_to_list(txt_file)
                for line in line_list:
                    det_class_name, confidence, left, top, right, bottom = line.split()
                    if det_class_name == class_name:
                        bbox = left + " " + top + " " + right + " " + bottom
                        bounding_boxes.append({"confidence": confidence, "file_name": file_name, "bbox":bbox})
            bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
            with open(join(self.tmp_file_path, class_name + "_dr.json"), 'w') as outfile:
                json.dump(bounding_boxes, outfile)

    def _draw_text_in_image(self, img, text, pos, color, line_width):
        bottomLeftCornerOfText = pos
        cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
        text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
        return img, (line_width + text_width)

    def _adjust_axes(self, r, t, fig, axes):
        # get text width for re-scaling
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        # get axis width in inches
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        # get axis limit
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1] * propotion])

    def _draw_false_negatives(self):
        count_false_negtives = defaultdict(int)

        for tmp_file in self.gt_json_files:
            ground_truth_data = json.load(open(tmp_file))
            # print(ground_truth_data)
            # get name of corresponding image
            start = self.tmp_file_path + '/'
            img_id = tmp_file[tmp_file.find(start) + len(start):tmp_file.rfind('_ground_truth.json')]
            img_cumulative_path = self.output_files_path + "/images/" + img_id + ".jpg"
            img = cv2.imread(img_cumulative_path)
            if img is None:
                img_path = self.image_root + '/' + img_id + ".jpg"
                img = cv2.imread(img_path)
            # draw false negatives
            for obj in ground_truth_data:
                if not obj['used']:
                    bbgt = [int(round(float(x))) for x in obj["bbox"].split()]
                    cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), pink, 2)
                    count_false_negtives[obj["class_name"]] += 1
            if self.show_animation:
                cv2.imwrite(img_cumulative_path, img)
        return count_false_negtives


    def _calculate_ap_each_class(self):
        print("calculate ap for each class ...")
        if not os.path.exists(join(self.output_files_path, "images","detections_one_by_one")):
            os.mkdir(join(self.output_files_path, "images","detections_one_by_one"))
        with open(join(self.output_files_path, "output.txt") , "w") as output_file:
            output_file.write("# AP and precision/recall per class\n")
            count_true_positives = {}
            count_false_positives = {}
            for class_index, class_name in enumerate(self.gt_classes):
                count_true_positives[class_name] = 0
                count_false_positives[class_name] = 0
                dr_file = join(self.tmp_file_path, class_name + "_dr.json")
                dr_data = json.load(open(dr_file))
                nd = len(dr_data)
                tp = [0] * nd  # creates an array of zeros of size nd
                fp = [0] * nd
                print(class_name + " calculating...")
                for idx, detection in tqdm(enumerate(dr_data)):
                    file_name = detection["file_name"]
                    if self.show_animation:
                        img_path = join(self.image_root, file_name + ".jpg")
                        img = cv2.imread(img_path)
                        img_cumulative_path = join(self.output_files_path, "images", file_name + ".jpg")
                        img_cumulative = img.copy() if not os.path.isfile(img_cumulative_path) else cv2.imread(img_cumulative_path)
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=black)
                    gt_file_path = join(self.tmp_file_path, file_name + "_ground_truth.json")
                    ground_truth_data = json.load(open(gt_file_path))
                    ovmax = -1
                    gt_match = -1
                    bb = [float(x) for x in detection["bbox"].split()]
                    for obj in ground_truth_data:
                        if obj["class_name"] == class_name:
                            bbgt = [float(x) for x in obj["bbox"].split()]
                            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                                     (bbgt[2] - bbgt[0]+ 1) * (bbgt[3] - bbgt[1] + 1) - \
                                     iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj
                    status = "NO MATCH FOUND"
                    if ovmax >= self.min_overlap:
                        if not gt_match["difficult"]:
                            if not bool(gt_match["used"]):
                                # true positive
                                tp[idx] = 1
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                # update the ".json" file
                                with open(gt_file_path, 'w') as f:
                                    f.write(json.dumps(ground_truth_data))
                                    status = "MATCH!"
                            else:
                                # false positive (multiple detection)
                                fp[idx] = 1
                                count_false_positives[class_name] += 1
                                status = "REPEATED MATCH!"
                    else:
                        # false positive
                        count_false_positives[class_name] += 1
                        fp[idx] = 1
                        if ovmax > 0:
                            status = "INSUFFICIENT OVERLAP"

                    """
                     Draw image to show animation
                    """
                    if self.show_animation:
                        height, widht = img.shape[:2]
                        v_pos = int(height - margin - (bottom_border / 2.0))
                        text = "Image: " + file_name + " "
                        img, line_width = self._draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        text = "Class [" + str(class_index) + "/" + str(self.n_classes) + "]: " + class_name + " "
                        img, line_width = self._draw_text_in_image(img, text,
                                                             (margin + line_width, v_pos),
                                                             light_blue,
                                                             line_width)

                        if ovmax != -1:
                            color = light_red
                            if status == "INSUFFICIENT OVERLAP":
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(self.min_overlap * 100)
                            else:
                                text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(self.min_overlap * 100)
                                color = green
                            img, _ = self._draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                        v_pos += int(bottom_border / 2.0)
                        rank_pos = str(idx + 1)  # rank position (idx starts at 0)
                        text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                            float(detection["confidence"]) * 100)
                        img, line_width = self._draw_text_in_image(img, text, (margin, v_pos), white, 0)
                        color = light_red
                        if status == "MATCH!":
                            color = green
                        text = "Result: " + status + " "
                        img, line_width = self._draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                        if ovmax > 0:  # if there is intersections between the bounding-boxes
                            bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                            cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                            cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                            cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                        cv2.LINE_AA)
                        bb = [int(i) for i in bb]
                        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                        cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                        cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                        output_img_path = self.output_files_path + "/images/detections_one_by_one/" + class_name + "_detection" + str(
                            idx) + ".jpg"
                        cv2.imwrite(output_img_path, img)
                        # save the image with all the objects drawn to it
                        cv2.imwrite(img_cumulative_path, img_cumulative)
                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / self.gt_counter_per_class[class_name]
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
                ap, mrec, mprec = self._voc_ap(rec[:], prec[:])
                self.sum_AP += ap
                text = "{0:.2f}%".format(
                    ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
                """
                 Write to output.txt
                """
                rounded_prec = ['%.2f' % elem for elem in prec]
                rounded_rec = ['%.2f' % elem for elem in rec]
                output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
                if not self.quiet:
                    print(text)
                self.ap_dictionary[class_name] = ap
                n_images = self.counter_images_per_class[class_name]
                lamr, mr, fppi = self._log_average_miss_rate(np.array(prec), np.array(rec), n_images)
                self.lamr_dictionary[class_name] = lamr
                """
                Draw plot
                """
                if self.draw_plot:
                    plt.plot(rec, prec, '-o')
                    # add a new penultimate point to the list (mrec[-2], 0.0)
                    # since the last line segment (and respective area) do not affect the AP value
                    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                    # set window title
                    fig = plt.gcf()  # gcf - get current figure
                    fig.canvas.set_window_title('AP ' + class_name)
                    # set plot title
                    plt.title('class: ' + text)
                    # plt.suptitle('This is a somewhat long figure title', fontsize=16)
                    # set axis titles
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    # optional - set axes
                    axes = plt.gca()  # gca - get current axes
                    axes.set_xlim([0.0, 1.0])
                    axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
                    # Alternative option -> wait for button to be pressed
                    # while not plt.waitforbuttonpress(): pass # wait for key display
                    # Alternative option -> normal display
                    # plt.show()
                    # save the plot
                    if not os.path.exists(join(self.output_files_path, "classes")):
                        os.makedirs(join(self.output_files_path, "classes"))
                    fig.savefig(join(self.output_files_path, "classes", class_name+".png"))
                    plt.cla()  # clear axes for next plot
            output_file.write("\n# mAP of all classes\n")
            mAP = self.sum_AP / self.n_classes
            text = "mAP = {0:.2f}%".format(mAP * 100)
            output_file.write(text + "\n")
            print(text)
        count_false_negtives = self._draw_false_negatives()
        with open(join(self.output_files_path, "output.txt"), "a") as output_file:
            tp_sum_num, fp_sum_num, fn_sum_num = 0, 0, 0
            for i in range(self.n_classes):
                tmp_tp = count_true_positives[self.gt_classes[i]]
                tmp_fp = count_false_positives[self.gt_classes[i]]
                tmp_fn = count_false_negtives[self.gt_classes[i]]
                tmp_text = self.gt_classes[i] + ": tp {:d}, fp {:d}, fn {:d}, percision {:.3f} recall {:.3f}\n".format(tmp_tp, tmp_fp, tmp_fn, tmp_tp/ (tmp_tp+tmp_fp), tmp_tp/ (tmp_tp+tmp_fn))
                output_file.write(tmp_text)
                tp_sum_num += tmp_tp
                fp_sum_num += tmp_fp
                fn_sum_num += tmp_fn
            tmp_text = "all classes: percision {:.3f}, recall {:.3f}".format(tp_sum_num/ (tp_sum_num + fp_sum_num), tp_sum_num/ (tp_sum_num + fn_sum_num))
            output_file.write(tmp_text)
        shutil.rmtree(self.tmp_file_path)

    def _count_det_results(self):
        self.det_counter_per_class = defaultdict(int)
        dr_files_list = sorted(glob(join(self.det_res_root, "*.txt")))
        for txt_file in dr_files_list:
            # get lines to list
            lines_list = self._txt_file_to_list(txt_file)
            for line in lines_list:
                class_name = line.split()[0]
                # check if class is in the ignore list, if yes skip
                if class_name in self.ignore_list:
                    continue
                # count that object
                self.det_counter_per_class[class_name] += 1
        self.dr_classes = list(self.det_counter_per_class.keys())

    def _draw_plot_func(self, dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                       true_p_bar):
        # sort the dictionary by decreasing value, into a list of tuples
        sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
        # unpacking the list of tuples into two lists
        sorted_keys, sorted_values = zip(*sorted_dic_by_value)
        #
        if true_p_bar != "":
            """
             Special case to draw in:
                - green -> TP: True Positives (object detected and matches ground-truth)
                - red -> FP: False Positives (object detected but does not match ground-truth)
                - pink -> FN: False Negatives (object not detected but present in the ground-truth)
            """
            fp_sorted = []
            tp_sorted = []
            for key in sorted_keys:
                fp_sorted.append(dictionary[key] - true_p_bar[key])
                tp_sorted.append(true_p_bar[key])
            plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
            plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                     left=fp_sorted)
            # add legend
            plt.legend(loc='lower right')
            """
             Write number on side of bar
            """
            fig = plt.gcf()  # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                fp_val = fp_sorted[i]
                tp_val = tp_sorted[i]
                fp_str_val = " " + str(fp_val)
                tp_str_val = fp_str_val + " " + str(tp_val)
                # trick to paint multicolor with offset:
                # first paint everything and then repaint the first number
                t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
                plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
                if i == (len(sorted_values) - 1):  # largest bar
                    self._adjust_axes(r, t, fig, axes)
        else:
            plt.barh(range(n_classes), sorted_values, color=plot_color)
            """
             Write number on side of bar
            """
            fig = plt.gcf()  # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                str_val = " " + str(val)  # add a space before
                if val < 1.0:
                    str_val = " {0:.2f}".format(val)
                t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
                # re-set axes to show number inside the figure
                if i == (len(sorted_values) - 1):  # largest bar
                    self._adjust_axes(r, t, fig, axes)
        # set window title
        fig.canvas.set_window_title(window_title)
        # write classes in y axis
        tick_font_size = 12
        plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
        """
         Re-scale height accordingly
        """
        init_height = fig.get_figheight()
        # comput the matrix height in points and inches
        dpi = fig.dpi
        height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
        height_in = height_pt / dpi
        # compute the required figure height
        top_margin = 0.15  # in percentage of the figure height
        bottom_margin = 0.05  # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        # set new height
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        # set plot title
        plt.title(plot_title, fontsize=14)
        # set axis titles
        # plt.xlabel('classes')
        plt.xlabel(x_label, fontsize='large')
        # adjust size of window
        fig.tight_layout()
        # save the plot
        fig.savefig(output_path)
        # close the plot
        plt.close()


    def _plot_static_info(self):
        if self.draw_plot:
            window_title = "ground-truth-info"
            plot_title = "ground-truth\n"
            plot_title += "(" + str(len(self.gt_json_files)) + " files and " + str(self.n_classes) + " classes)"
            x_label = "Number of objects per class"
            output_path = self.output_files_path + "/ground-truth-info.png"
            to_show = False
            plot_color = 'forestgreen'
            self._draw_plot_func(
                self.gt_counter_per_class,
                self.n_classes,
                window_title,
                plot_title,
                x_label,
                output_path,
                to_show,
                plot_color,
                '',
            )

    def _write_down_ground_truth_objects(self):
        with open(self.output_files_path + "/output.txt", 'a') as output_file:
            output_file.write("\n# Number of ground-truth objects per class\n")
            for class_name in sorted(self.gt_counter_per_class):
                output_file.write(class_name + ": " + str(self.gt_counter_per_class[class_name]) + "\n")

        for class_name in self.dr_classes:
            # if class exists in detection-result but not in ground-truth then there are no true positives in that class
            if class_name not in self.gt_classes:
                count_true_positives[class_name] = 0

    def eval(self):
        self._dump_gt_json()
        self._dump_det_res_json()
        self._calculate_ap_each_class()
        self._count_det_results()
        self._plot_static_info()
        self._write_down_ground_truth_objects()



def main(args):
    gt_root = args.gt_root
    det_res_root = args.det_res_root
    image_root = args.img_root
    ignore_list = args.ignore_list
    if ignore_list is None:
        ignore_list = []
    min_overlap = args.min_overlap
    quiet = args.quiet
    draw_plot = args.draw_plot
    show_animation = args.show_animation
    map_eval = mAPevaluater(gt_root, det_res_root, image_root,
                            ignore_list,
                            min_overlap=min_overlap,
                            quiet=quiet,
                            draw_plot=draw_plot,
                            show_animation=show_animation)
    map_eval.eval()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root",
                        default="/zhuangjf/gitae/Lightweight-Segmentation/runs/gt_txt",
                        type=str,
                        help="ground truth root")
    parser.add_argument("--det-res-root",
                        default="/zhuangjf/gitae/Lightweight-Segmentation/runs/pred_txt",
                        type=str,
                        help="detection results root")
    parser.add_argument("--img-root",
                        default="/zhuangjf/gitae/Lightweight-Segmentation/tujiao_val_pic",
                        type=str,
                        help="image root")
    parser.add_argument("--show-animation",
                        default=False, type=bool,
                        help="draw plot results")
    parser.add_argument("--min-overlap",
                        default=0.5, type=float,
                        help="defined in the PASCAL VOC2012 challenge")
    parser.add_argument("--quiet",
                        default=True, type=bool,
                        help="show the results on screen")
    parser.add_argument("--draw_plot",
                        default=False, type=bool,
                        help="draw plot")
    parser.add_argument("--ignore-list", nargs='+')
    parser.add_argument("--output-root",
                        default="/zhuangjf/gitae/Lightweight-Segmentation/runs/output_results",
                        type=str,
                        help="output image directory")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)

