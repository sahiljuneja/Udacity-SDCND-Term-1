import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time

# TODO: Load traffic signs data.
training_file = "train.p"
with open(training_file, mode='rb') as f:
	train = pickle.load(f)

X_train, y_train = train['features'], train['labels']

# TODO: Split data into training and validation sets.
train_features, valid_features, train_labels, valid_labels = train_test_split(X_train, y_train, test_size = 0.15, random_state = 43242)

# TODO: Define placeholders and resize operation.
x = tf.placeholder("float", [None, 32, 32, 3])
y = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(x, [227,227])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
shape = (fc7.get_shape().as_list()[-1], 43)
w = tf.Variable(tf.truncated_normal(shape, stddev = 0.01))
b = tf.Variable(tf.zeros(43))
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
logits = tf.nn.xw_plus_b(fc7, w, b)
probs = tf.nn.softmax(logits)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
correct_prediction = tf.equal(tf.arg_max(probs, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# TODO: Train and evaluate the feature extraction model.
epochs = 10
batch_size = 128
init_op = tf.initialize_all_variables()

def eval_on_data(X1, y1, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X1.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X1[offset:end]
        y_batch = y1[offset:end]

        loss, acc = sess.run([cost, accuracy], feed_dict={x: X_batch, y: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X1.shape[0], total_acc/X1.shape[0]

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        # training
        train_features, train_labels = shuffle(train_features, train_labels)
        t0 = time.time()
        for offset in range(0, train_features.shape[0], batch_size):
            end = offset + batch_size
            sess.run(optimizer, feed_dict={x: train_features[offset:end], y: train_labels[offset:end]})

        val_loss, val_acc = eval_on_data(valid_features, valid_labels, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")
