import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)      
    graph = tf.get_default_graph()
    vgg_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep= graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3= graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7= graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return vgg_input, vgg_keep, vgg_layer3, vgg_layer4, vgg_layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 convolution of VGG layer 7
    vgg_layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#    vgg_layer7_1x1 = tf.Print(vgg_layer7_1x1, [tf.shape(vgg_layer7_1x1)])

    # 1x1 convolution of VGG layer 4
    vgg_layer4_scaled = tf.multiply(vgg_layer4_out, 0.01)
    vgg_layer4_1x1 = tf.layers.conv2d(vgg_layer4_scaled, num_classes, 1, padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#    vgg_layer4_1x1 = tf.Print(vgg_layer4_1x1, [tf.shape(vgg_layer4_1x1)])

    # 1x1 convolution of VGG layer 3
    vgg_layer3_scaled = tf.multiply(vgg_layer3_out, 0.0001)
    vgg_layer3_1x1 = tf.layers.conv2d(vgg_layer3_scaled, num_classes, 1, padding='same',
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#    vgg_layer3_1x1 = tf.Print(vgg_layer3_1x1, [tf.shape(vgg_layer3_1x1)])

    # upsample layer 7
    layer4 = tf.layers.conv2d_transpose(vgg_layer7_1x1, num_classes, 4, 2, padding = 'same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#    layer4 = tf.Print(layer4, [tf.shape(layer4)])

    # skip connection with vgg layer 4 1x1
    layer4 = tf.add(layer4, vgg_layer4_1x1)
#    layer4 = tf.Print(layer4, [tf.shape(layer4)])

    # upsample layer 4
    layer3 = tf.layers.conv2d_transpose(layer4, num_classes, 4, 2, padding = 'same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#    layer3 = tf.Print(layer3, [tf.shape(layer3)])

    # skip connection with vgg layer 3 1x1
    layer3 = tf.add(layer3, vgg_layer3_1x1)
#    layer3 = tf.Print(layer3, [tf.shape(layer3)])

    # upsample
    layer_final = tf.layers.conv2d_transpose(layer3, num_classes, 16, 8, padding = 'same',
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#    layer_final = tf.Print(layer_final, [tf.shape(layer_final)])
    
    return layer_final

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # cross entropy loss plus the regularization loss terms
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_const = 1.0
    loss = cross_entropy_loss + reg_const * sum(reg_loss)

    # training operation      
    training_operation = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
        
    return logits, training_operation, loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss_op, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param loss_op: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())
    print("training - epochs: ", epochs, " batch_size: ", batch_size)
    for epoch in range(1, epochs+1):
        print("epoch: ", epoch)
        for (images, labels) in get_batches_fn(batch_size):
            # train batch
            junk, loss = sess.run([train_op, loss_op], feed_dict={input_image: images, correct_label: labels, keep_prob: 0.5, learning_rate: 0.0001})
            print("  batch loss:", loss)

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 25
    # Larger batch sizes might speed up the training but can degrade the quality of the model at the same time.
    # Good results obtained using small batch sizes of 2 or 4.
    # It important to note that batch size and learning rate are linked. If the batch size is too small then
    # the gradients will become more unstable and learning rate would need to be reduced (~1e-4 or 1e-5).
    batch_size = 4
    
    lr = 0.0001
    kp = 0.6

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)
        
        # import vgg model, add FCN decoder layers, define optimizer
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        full_cnn = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, training_op, loss_op = optimize(full_cnn, correct_label, learning_rate, num_classes)
        
        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, training_op, loss_op, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        print("Saving")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
