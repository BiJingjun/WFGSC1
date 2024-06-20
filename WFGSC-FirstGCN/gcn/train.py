from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

from sklearn.metrics import f1_score

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', '1', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.') # 0.05
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.') # 200
flags.DEFINE_integer('hidden1', 34, 'Number of units in hidden layer 1.') # 16
flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).') # 0.5
flags.DEFINE_float('weight_decay', 0.0001, 'Weight for L2 loss on embedding matrix.') # 5e-4
flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).') # 10
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') # 3
flags.DEFINE_integer('pre', 1, 'Percentage of training set')
flags.DEFINE_float('lamb', 1, 'lamb.')
# flags.DEFINE_float('lambS', 10, 'lambS.')
# flags.DEFINE_float('lamb3', 0.000000001, 'lamb3.')
# flags.DEFINE_integer('hidden_mlp', 40, 'Number of MLP hidden layer.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
n=adj.get_shape()[0]
flags.DEFINE_integer('N', n, 'N.')
# Some preprocessing
features = preprocess_features(features)
L = preprocess_laplacian(adj)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32) # helper variable for sparse dropout
}

# Create model
model = model_func(L, adj, placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []



# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break




print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

y_label = y_test.argmax(axis=1)
out = tf.argmax(tf.nn.softmax(model.outputs), 1)
pres = sess.run(out, feed_dict=feed_dict)

preout = sess.run(tf.nn.softmax(model.outputs), feed_dict=feed_dict)
sio.savemat("handF.mat", {'pre': preout})

pretest = pres[test_mask]
ylabeltest = y_label[test_mask]

w_f1 = f1_score(ylabeltest, pretest, average='weighted')
m_f1 = f1_score(ylabeltest, pretest, average='macro')


file_handle = open('handwriting-1.txt', mode='a')
file_handle.write(FLAGS.dataset+"\n"
                  +"{:.9f}".format(FLAGS.learning_rate)+"\n"
 #                 +"{:.9f}".format(FLAGS.hidden_mlp)+"\n"
                  +"{:.9f}".format(FLAGS.hidden1)+"\n"
                  +"{:.9f}".format(FLAGS.weight_decay)+"\n"
                  +"{:.9f}".format(FLAGS.lamb)+"\n"
                  # +"{:.11f}".format(FLAGS.lambS)+"\n"
                  # +"{:.11f}".format(FLAGS.lamb3)+"\n"
                  +"test_acc="+"{:.5f}".format(test_acc)+"\n"
                  +"w_f1="+"{:.9f}".format(w_f1)+"\n"+"m_f1c="+"{:.9f}".format(m_f1)+"\n")
file_handle.close()
