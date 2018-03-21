from __future__ import print_function
from __future__ import absolute_import

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
import tensorflow as tf
import numpy as np
from PIL import Image
# TensorFlow serving stuff to send messages
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.saved_model import signature_constants
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
import scipy

# Command line arguments
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                       'PredictionService host:port')
tf.app.flags.DEFINE_string('image', 'image5.jpg', '/home/raina25d/Downloads/')
FLAGS = tf.app.flags.FLAGS


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)

def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Send request
    request = predict_pb2.PredictRequest()
    image = Image.open("/home/raina25d/Downloads/image5.jpg")
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Call my model to make prediction on the image
    request.model_spec.name = 'exp_model'
    request.model_spec.signature_name = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.inputs['inputs'].CopyFrom(
    tf.contrib.util.make_tensor_proto(image_np_expanded))

    result = stub.Predict(request, 60.0)  # 60 secs timeout
    print(result)
    # Plot boxes on the input image
    label_map = label_map_util.load_labelmap('od.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    boxes = result.outputs['detection_boxes'].float_val
    classes = result.outputs['detection_classes'].float_val
    scores = result.outputs['detection_scores'].float_val
    image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
    	image_np,
    	np.reshape(boxes,[100,4]),
    	np.squeeze(classes).astype(np.int32),
    	np.squeeze(scores),
    	category_index,
    	use_normalized_coordinates=True,
    	line_thickness=8)
    scipy.misc.imsave('response.jpg', image_vis)


if __name__ == '__main__':
    tf.app.run()
