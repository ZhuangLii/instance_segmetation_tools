
# instance_segmetation_tools
## Requirement
```
labelme==3.8.0
```
## Support
- [x] trans labelme to mask (0 for background)
- [x] trans labelme to COCO format data
- [x] trans labelme to VOC format data
- [x] MAP for detection 
- [x] MIoU for instance segmentation
- [x] Simple copy and paste for instance segmentation
- [x] mask png to labelme format json data


### MIoU for instance segmentation
***Pred mask & Label mask***
```
[ 0, 0, 1, ....
  1, 1, 1, ....
  0, 2, 2, ....
  0, 0 ,0, ....
  ...
  ...]
```
0 for backdound,  >= 1 for classes
***Info.txt***
```
jt
qp
```
***Results***
```
-----------Acc of each classes-----------
back_ground          : 97.387168 %	
jt          : 97.197456 %	
qp          : 0.000141 %	
mean Pixel Acc: 64.86
-----------IoU of each classes-----------
back_ground          : 93.342430 %	
jt          : 95.128109 %	
qp          : 0.000140 %	
mIOU: 62.82
```


### MAP for detection

copy from https://github.com/Cartucho/mAP

***What's New***
- [x] We add precision and recall statistics in output.
```
# <a name="6">mAP of all classes</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>
mAP = 74.78%
jt: tp 106, fp 18, fn 7, percision 0.855 recall 0.938
qp: tp 51, fp 25, fn 29, percision 0.671 recall 0.637
all classes: percision 0.785, recall 0.813
# <a name="7">Number of ground-truth objects per class</a><a style="float:right;text-decoration:none;" href="#index">[Top]</a>
jt: 113
qp: 80
```

***Create the ground-truth files***

    ```
    <class_name> <left> <top> <right> <bottom> 1 or 0
    ```
- The `difficult` parameter is 1 otherwise 0, use it if you want the calculation to ignore a specific detection.
- E.g. "image_1.txt":
    ```
    tvmonitor 2 10 173 238 0
    book 439 157 556 241 0
    book 437 246 518 351 1
    pottedplant 272 190 316 259 0
    ```

***Create the detection-results files***

- Create a separate detection-results text file for each image.
- Use **matching names** for the files (e.g. image: "image_1.jpg", detection-results: "image_1.txt").
- In these files, each line should be in the following format:
    ```
    <class_name> <confidence> <left> <top> <right> <bottom>
    ```
- E.g. "image_1.txt":
    ```
    tvmonitor 0.471781 0 13 174 244
    cup 0.414941 274 226 301 265
    book 0.460851 429 219 528 247
    chair 0.292345 0 199 88 436
    book 0.269833 433 260 506 336
    ```

***color***
```
- green -> TP: True Positives (object detected and matches ground-truth)
- red -> FP: False Positives (object detected but does not match ground-truth)
- pink -> FN: False Negatives (object not detected but present in the ground-truth)
```
