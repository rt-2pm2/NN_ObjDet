# Generate the TFrecord data for the NN training 
#from __future__ import division
#from __future__ import print_function
#from __future__ import absolute_import

import sys, getopt
from glob import glob
import xml.etree.ElementTree as ET
from os import path
import base64

import tensorflow as tf

tf.enable_eager_execution()

#from object_detection.utils import import dataset_util

# Helper
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if (type(value) is list):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
  
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  if (type(value) is float):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
  if (type(value) is list):
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  if (type(value) is list):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def extract_xml(xml_filename, label_map):
    tree = ET.parse(xml_filename)
    root = tree.getroot()
    
    # The bounding box information is contained in the object Elemenent of the tree
    #
    # object
    #   |
    #   |-->  <name> name </name>
    #   |-->  bndbox
    #           |--> <xmin> xm </xmin>
    #           |--> <ymin> ym </ymin>
    #   ...     ...
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes_text = []
    classes = []
    for obj in root.iter('object'):
        bndbox = obj.find('bndbox')
        nameel = obj.find('name')
        xmin.append(int(bndbox.find('xmin').text))
        ymin.append(int(bndbox.find('ymin').text))
        xmax.append(int(bndbox.find('xmax').text))
        ymax.append(int(bndbox.find('ymax').text))
        classes_text.append(nameel.text.encode('utf8'))
        classes.append(label_map[nameel.text])

    return (xmin, ymin, xmax, ymax, classes_text, classes)

def normalize_coord(xmin, ymin, xmax, ymax, img_height, img_width):
    for i in range(len(xmin)):
        xmin[i] /= img_width
        xmax[i] /= img_width

        ymin[i] /= img_height
        ymax[i] /= img_height        

def usage():
    print("Generate the TFRecords from input data")
    print(f"Usage: python3 {__file__} -i <path2inputfiles> -o <path2outputfiles>") 

def create_tf_example(image_filename, xml_filename):
    print(f"Elaborating {image_filename}...")
    file = open(image_filename, 'rb').read()

    # Image data information
    image_shape = tf.image.decode_jpeg(file).shape
    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    image_depth = image_shape[2]
    image_encoded_data = base64.b64encode(file)
    image_format = b'jpg'
    
    # Extract the information from the XML
    (xmins, ymins, xmaxs, ymaxs, classes_text, classes) = extract_xml(xml_filename, label_map)
    
    # Normalize the coordinates
    normalize_coord(xmins, ymins, xmaxs, ymaxs, image_height, image_width)

    # Create the dict containing the data
    dict_height = _int64_feature(image_height)
    dict_width = _int64_feature(image_width)
    dict_filename = _bytes_feature(image_filename.encode('utf8'))
    dict_encoded = _bytes_feature(image_encoded_data)
    dict_format = _bytes_feature(image_format)
    # Boxes and labels
    dict_source_id = _bytes_feature(image_filename.encode('utf8'))
    dict_xmin = _float_feature(xmins)
    dict_xmax = _float_feature(xmaxs)
    dict_ymin = _float_feature(ymins)
    dict_ymax = _float_feature(ymaxs)
    dict_text = _bytes_feature(classes_text)
    dict_label = _int64_feature(classes)

    data_dict = {
            # Image Data
            'image/height': dict_height,
            'image/width': dict_width,
            'image/filename': dict_filename, 
            'image/encoded': dict_encoded, 
            'image/format': dict_format, 
            # Boxes and labels
            'image/source_id': dict_source_id, 
            'image/object/bbox/xmin': dict_xmin, 
            'image/object/bbox/xmax': dict_xmax,
            'image/object/bbox/ymin': dict_ymin, 
            'image/object/bbox/ymax': dict_ymax, 
            'image/object/class/text': dict_text, 
            'image/object/class/label': dict_label,
            }

    feat_dict = tf.train.Features(feature=data_dict)

    # Create the tf_example data
    tf_example = tf.train.Example(features=feat_dict)

    return tf_example

label_map = {
        "soldier": 1,
        "rifle": 2
        }


def main(argv):
    input_val = './'
    output_val = './'

    try:
        (opts, _) = getopt.getopt(argv, "i:o:h",
                ['input-path=', 'output-path=', 'help'])
    except getopt.GetoptError as err:
        # Something went wrong during parsing
        print(f"In {__file__}: {str(err)}")
        usage()
        sys.exit(2)
    for (opt, val) in opts:
        if (opt in ("-i", "--input-path")):
            input_path = val
        if (opt in ("-o", "--output-pth")):
            output_path = val
        if (opt in ("-h", "--help")):
            usage()


    writer = tf.python_io.TFRecordWriter(output_path + 'images.tfrecords')

    ## Creatign the list of training files
    ##
    print(f"Looking for input data in {input_path}...")
    input_file_list = glob(input_path + "*.jpg")
    print(f"Found {len(input_file_list)} files!\n")

    if (len(input_file_list) == 0):
        print("Error: no files found!")
        writer.close()
        sys.exit(2)

    ## Processing Files
    for filename in input_file_list:
        # For each file I expect to find a xml file containing
        # information about the boxes and relative labels
        xml_filename = path.splitext(filename)[0] + ".xml"
        if (not path.isfile(xml_filename)):
            print("Error: Missing XML file")
            continue

        # Create the tf_example
        tf_example = create_tf_example(filename, xml_filename)           

        print(f"Adding {filename} to TFRecord")
        writer.write(tf_example.SerializeToString())

    writer.close() 

if(__name__ == "__main__"):
    main(sys.argv[1:]) # The 0 element is the name of the script
