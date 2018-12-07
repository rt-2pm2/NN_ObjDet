# Object Detection using Tensorflow.
This python application performs object detection using Tensorflow. 

## Getting Started
0. Virtual Environment(Optional) 
python3 -m venv [target_folder]
source ./[target_folder]/bin/activate

1. Requirements
pip install -r requirements.txt
 
2. Usage
> python3 ./nn_objdet/main.py [args] 

List of arguments are (with default value []):

"-d", "--display", [0] Whether or not frames should be displayed
"-o", "--output", [0] Whether or not modified videos shall be writen
"-op", "--output-path", ["./output"] Name of the output video file
"-i", "--input-source", "./" Path to videos input, overwrite device input if used
'-w', '--num-workers', [2], Number of workers
'-q-size', '--queue-size', [5] Size of the queue.
'-l', '--logger-debug', [0], Print logger debug
"-pg", "--graph_path",["./model"] Path to the frozen graph
"-pl", "--label_path", ["./model"],Path to the object labels

Work in progress...

# Application structure
The structure of the application is shown in the following Figure.
![alt text](https://github.com/rt-2pm2/NN_ObjDet/blob/master/doc/app_scheme.gif)

The aim is to take advantage of the concurrent execution to speed up the object detection routine. I still have to check the assumption, but the idea is that the data flow is more I/O bound, whilst the NN jobs are CPU bound.
Due to the GIL problem under python, CPU bound activities won't get the expected speed up if managed with multithreading. A possible solution is to deploy the routines in separate processes, with separate memory spaces, to avoid the GIL issue. 


