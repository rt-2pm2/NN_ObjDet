# @file: nn_objdetector.py
#
#
import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util


class NN_ObjDetector:
    """ This class warps the NN structure to perform Object detections """

    def __init__(self, path2fg, path2lab):
        # Path to frozen detection graph. This is the actual model that is used
        # for the object detection.
        self.PATH_TO_CKPT = path2fg

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = path2lab

        # Loading label map
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
                label_map, use_display_name=True, max_num_classes=90)

        # This is what is used in other methods
        self.category_index = label_map_util.create_category_index(
                categories)

        # Start the TF environment
        self.detection_graph = tf.Graph()  # TF graph
        with self.detection_graph.as_default(): # Configure the graph as default
            od_graph_def = tf.GraphDef()

            # Initialization of the graph
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                # Import the graph def into the default graph
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


            ## Get the tensor from the graph
            self.image_tensor = \
                    self.detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was 
            # detected.
            self.boxes_tens = \
                    self.detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores_tens = \
                    self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes_tens = \
                    self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections_tens = \
                    self.detection_graph.get_tensor_by_name('num_detections:0')

            # Define the Session 
            self.sess = tf.Session(graph=self.detection_graph)



    def detect_objects(self, image_np):
        # Expand dimensions since the model expects images to have shape: 
        # [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes_tens, self.scores_tens, \
                        self.classes_tens, self.num_detections_tens],
                feed_dict={self.image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, np.squeeze(boxes), 
                np.squeeze(classes).astype(np.int32), 
                np.squeeze(scores), self.category_index,
                use_normalized_coordinates=True, line_thickness=4)

        return image_np



    def close_session(self):
        self.sess.close()


