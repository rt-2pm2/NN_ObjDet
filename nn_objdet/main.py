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
import heapq

# My Library
from classes.nn_objdetector import *
from classes.timemeas import *

TIME_TO_EXIT = Value(ctypes.c_bool, False)

#### WORKING THREAD
def work(input_q, processed_q, path2fg, path2lab, TIME_TO_EXIT):
    """
    Function for the processing of the frames

    Args:
        input_q (Queue): Input queue for the input frames
        processed_q (Queue): Output queue for the processed frames
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
            processed_q.put((frame[0], outframe))
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            tm.start()
            outframe = nn_od.detect_objects(frame_rgb)
            tm.stop()
            tm.tick()

            processed_q.put(outframe)

    print(f"NN Process[{os.getpid():4}] | " + 
            f"Avg Period = {tm.getPeriod():3.6} s " +
            f"Avg Comp. Time = {tm._avg_elapsed:3.6} s")

    nn_od.close_session()


def data_flow(source, input_q, processed_q):
    """
    Function for the processing of the data streams 

    Args:
        source (Str): Path to the input file
        input_q (Queue): Input queue for the input frames
        processed_q (Queue): Output queue for the processed frames

    Returns:
        (void)

    """   
    vs = cv2.VideoCapture(source)
 
    if (not vs.isOpened()):
        print("Problem opening the file!")
        return
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
    p_out = Thread(target=outflow_thread, args=(True, nFrame, True, processed_q, out))
    
    p_in.start()
    p_out.start()

    p_in.join()
    print("Inflow thread terminated")
    p_out.join()
    print("Outflow thread terminated")
   
    print("Terminating Data Flow Process...")

    # Cleaning up
    vs.release()
    if (out):
        out.release()


def inflow_thread(input_q, vs):
    """
    Function to process the input stream

    Args: 
        input_q (Queue): Input queue for the input frames
        vs (VideoCapture): Object to capture the frames

    Returns:
        void

    """
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


def outflow_thread(disp, dim, outen, processed_q, out):
    """
    Function to process the input stream

    Args: 
        disp (Bool): Flag to activate the visualization
        dim (int): Length of the output stream
        outen (Bool): Flag to enable the write to file
        processed_q (Queue): Output queue for the output frames
        out: Object to write the frames

    Returns:
        void
    """
    countWriteFrame = 1
    firstUsedFrame = True
    firstTreatedFrame = True

    output_pq = []

    print("Outflow Thread started!\n")

    tm = TimeMeas()
    tm.start()
    while (not TIME_TO_EXIT.value):
        # If there are processed frames, otherwise block
        try:
            #print(f"Reading queue: {processed_q.qsize()}")
            (prior, outframe) = processed_q.get(block=True, timeout=1)
        except queue.Empty:
            if ((countWriteFrame <= dim) and (not TIME_TO_EXIT.value)):
                continue
            else:
                # No more frames either something got stuck 
                break
            
        # Extract the next element
        (prior, outframe) = heapq.heappushpop(output_pq, (prior, outframe))
        if (prior > countWriteFrame):
            heapq.heappush(output_pq, (prior, outframe))
            continue

        # Start putting the frames in the output file
        while (prior == countWriteFrame):
            tm.tick()
            output_rgb = cv2.cvtColor(outframe, cv2.COLOR_RGB2BGR)
            # If it was requested an output file
            if outen:
                out.write(output_rgb)
            # If it was requested video output
            if (disp):
                cv2.imshow('frame', output_rgb)
                key = cv2.waitKey(1) & 0xFF

            countWriteFrame = countWriteFrame + 1

            try:
                (prior, outframe) = heapq.heappop(output_pq)
            except IndexError:
                break

            if (prior > countWriteFrame):
                heapq.heappush(output_pq, (prior, outframe))
            
        if firstTreatedFrame:
            print("Retrieving processed data...\n")
            firstTreatedFrame = False

        if firstUsedFrame:
            print("Started\n")
            firstUsedFrame = False
                
    print("Terminating Outflow Thread...")   
    out_freq = tm.getfreq()
    print(f"Output processing rate = {out_freq:3.2} Hz")


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
    processed_q = Queue(maxsize=args["queue_size"])

    path_to_graph = args["path2graph"]
    path_to_labels = args["path2labels"]

    ## INPUT PROCESS
    source = args["input_source"]
    print("Source = " + source + "\n")

    data_process = Process(target=data_flow, args=(source, input_q, processed_q))
    data_process.start()
    
    ## WORKING PROCESSES
    # Creates the a pool of working processes
    pool = Pool(args["num_workers"], work, \
            (input_q, processed_q, path_to_graph, path_to_labels, TIME_TO_EXIT))

     
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
