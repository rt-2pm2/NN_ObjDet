#!/bin/zsh

executable="./nn_objdet/main.py"
path2graph="model/ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb"
#path2graph="model/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb"

path2labels="model/mscoco_label_map.pbtxt"


python3 $executable -d 0 -o 1 -op "./outputs/out.avi" -i "./inputs/cutcut.mp4" -pg $path2graph -pl $path2labels -w 2 -q-size 200 
