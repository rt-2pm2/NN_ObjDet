# @file main.py
#
#
import argparse
from multiprocessing import Queue, Pool, Process, Value
from threading import Thread
import queue
import cv2
import os, sys
import ctypes

# My Library
from classes.nn_objdetector import *
from classes.timemeas import *

TIME_TO_EXIT = Value(ctypes.c_bool, False)

#### WORKING THREAD
def work(input_q, output_q, path2fg, path2lab, TIME_TO_EXIT):
    """
    Function for the processing of the frames

    Args:
        input_q (Queue): Input queue for the input frames
        output_q (Queue): Output queue for the processed frames
        path2fg (Str): Path to the Frozen Graph file
        path2lab (Str): Path to the Labels file

    Returns:
        (void)

    """

    # Instantiate the Object Detector class
    nn_od = NN_ObjDetector(path2fg, path2lab)

    tm = TimeMeas()
    while (not TIME_TO_EXIT.value):
        try:
            frame = input_q.get(block=True, timeout=1) # Blocking until data is available
        except queue.Empty:
            if (TIME_TO_EXIT.value):
                # No more frame to process
                break
            else:
                # Maybe a delay in the inflow?
                continue

        if (len(frame) == 2):
            # frame[0] is the index |  frame[1] is the frame data
            frame_rgb = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)

            # Process the frame
            tm.tick()
            tm.start()
            outframe = nn_od.detect_objects(frame_rgb)
            tm.stop()

            # Put it in the outqueue
            output_q.put((frame[0], outframe))
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            tm.start()
            outframe = nn_od.detect_objects(frame_rgb)
            tm.stop()
            tm.tick()

            output_q.put(outframe)

    print(f"NN Process[{os.getpid():4}] | " + 
            f"Avg Period = {tm.getPeriod():3.6} s " +
            f"Avg Comp. Time = {tm._avg_elapsed:3.6} s")

    nn_od.close_session()


def data_flow(source, input_q, output_q):
    
    vs = cv2.VideoCapture(source)

    if (not vs.isOpened()):
        print("Problem opening the file!")
    else:
        print("Loaded file with " + \
                str(int(vs.get(cv2.CAP_PROP_FRAME_COUNT))) + " frames")

    ## OUTPUT
    if args["output"]:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = vs.get(cv2.CAP_PROP_FPS)
        fwidth= int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        fheight = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args["output_path"],
                fourcc, fps, (fwidth, fheight))


    # Read the number of frames in the source
    nFrame = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    if (nFrame <= 0):
        print("No frame to process!", file=sys.stderr)
        sys.exit()

    p_in = Thread(target=inflow_thread, args=(input_q, vs))
    p_in.start()

    p_out = Thread(target=outflow_thread, args=(True, nFrame, True, output_q, out))
    p_out.start()

    p_in.join()
    p_out.join()
   
    print("Terminating Data Flow Process")

    # Cleaning up
    vs.release()
    if (out):
        out.release()


def inflow_thread(input_q, vs):
    firstReadFrame = True

    # Initialize the counter
    countReadFrame = 0

    print("Inflow Process started!\n")
    
    tm = TimeMeas() 
    tm.start() 
    while (not TIME_TO_EXIT.value):
        # If there is space in the feeding queue 
        (ret, frame) = vs.read()
        if ret:
            tm.tick()  
            # Get the index of the next frame
            frameindex = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
            # Add the tuple (index, frame) to the input queue
            input_q.put((frameindex, frame), block=True, timeout=None) # Blocking insertion
            #print("Input queue = " + str(input_q.qsize()))

            countReadFrame = countReadFrame + 1
            if (firstReadFrame):
                print("Reading data started...\n")
                firstReadFrame = False
        else:
            print("End of file: " + str(frameindex))
            break
    tm.stop()
    print("Terminating Inflow Thread...")

    in_freq = tm.getfreq()
    print(f"Input processing rate = {in_freq:6.3}" + 
            f" | Ticks = {tm._nTicks:3}" +
            f" in {tm._elapsed:0.3} s")

def outflow_thread(disp, dim, outen, output_q, out):
    firstUsedFrame = True
    firstTreatedFrame = True
    countWriteFrame = 1

    output_pq = queue.PriorityQueue(maxsize=3*args["queue_size"])

    print("Outflow Thread started!\n")

    tm = TimeMeas()
    tm.start()
    while (not TIME_TO_EXIT.value):
        # If there are processed frames, otherwise block
        outframe = output_q.get(block=True, timeout=None)
        #print("Output queue = " + str(output_q.qsize()))
        output_pq.put(outframe)

        if firstTreatedFrame:
            print("Retrieving processed data...\n")
            firstTreatedFrame = False

        # Check the priority queue
        if not output_pq.empty():
            # Extract the first tuple
            (prior, output_frame) = output_pq.get()
            # Check whether it is the next (maintain the order)
            if (prior > countWriteFrame):
                # It is out of order, put it back
                output_pq.put((prior, output_frame))
            else:
                tm.tick()

                countWriteFrame = countWriteFrame + 1
                output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

                # If it was requested an output file
                if outen:
                    out.write(output_rgb)
    
                if (disp):
                    cv2.imshow('frame', output_rgb)
                    key = cv2.waitKey(1) & 0xFF

                if firstUsedFrame:
                    print("Started\n")
                    firstUsedFrame = False
                
                if (countWriteFrame >= dim):
                    break

    print("Terminating Outflow Thread...")   
    out_freq = tm.getfreq()
    print("Output processing rate = " + str(out_freq))

#### START
def start(args):
    """
    Start the application
    """

    ## INIT
    # Set the multiprocessing logger to debug if required
    if args["logger_debug"]:
        logger = multiprocessing.log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)
    
    ## DATA STRUCTURES
    # Define the shared data structures (Input Queues)
    input_q = Queue(maxsize=args["queue_size"])
    output_q = Queue(maxsize=args["queue_size"])


    path_to_graph = args["path2graph"]
    path_to_labels = args["path2labels"]

    ## WORKING PROCESSES
    # Creates the a pool of working processes
    pool = Pool(args["num_workers"], work, \
            (input_q, output_q, path_to_graph, path_to_labels, TIME_TO_EXIT))

    ## INPUT PROCESS
    source = args["input_source"]
    print("Source = " + source + "\n")

    data_process = Process(target=data_flow, args=(source, input_q, output_q))
    data_process.start()
    

    
    ### MAIN LOOP
    data_process.join()

    TIME_TO_EXIT.value = True

    pool.close()
    pool.join()

     ## TERMINATE
    print("Terminating Main...\n")

    cv2.destroyAllWindows()


if __name__ == '__main__':

    # Construct the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-", "--num-frames", type=int, default=0,
            help="# of frames to loop over for FPS test")

    ap.add_argument("-d", "--display", type=int, default=0,
            help="Whether or not frames should be displayed")
    ap.add_argument("-o", "--output", type=int, default=0,
            help="Whether or not modified videos shall be writen")
    ap.add_argument("-op", "--output-path", type=str, default="output",
            help="Name of the output video file")
    ap.add_argument("-I", "--input-device", type=int, default=0,
            help="Device number input")
    ap.add_argument("-i", "--input-source", type=str, default="",
            help="Path to videos input, overwrite device input if used")
    ap.add_argument('-w', '--num-workers', dest='num_workers', type=int,
            default=2, help='Number of workers.')
    ap.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
            default=5, help='Size of the queue.')
    ap.add_argument('-l', '--logger-debug', dest='logger_debug', type=int,
            default=0, help='Print logger debug')
    ap.add_argument("-pg", "--graph_path", type=str, dest='path2graph', default="model",
            help="Path to the frozen graph")
    ap.add_argument("-pl", "--label_path", type=str, dest='path2labels', default="model",
            help="Path to the object labels")

    args = vars(ap.parse_args())

    start(args)
