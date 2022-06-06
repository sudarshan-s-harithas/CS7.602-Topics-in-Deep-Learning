import tensorflow.compat.v1 as tf
import numpy as np 
from utils import * 
from Models_Q3 import *
import time 
import matplotlib.pyplot as plt
import sys
from keras.datasets import imdb
import pandas as pd 

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


vocabulary_size = 5000


arguments_passed = sys.argv

method = arguments_passed[1]

# normalization_type = "symmetric"  ## "row" , "col" , "symmetric"  
#for optimal accuracy in gcn and graphsage we use symmetric normalizartion
# aggregation_type = "mean" , "sum" for optimal accuracy we use mean 

if ( method == "gcn" ):
    aggregation_type = "mean"
    normalization_type = "symmetric"
    combination_type = None

if( method == "GraphSage"):

    aggregation_type = "mean"
    normalization_type = "symmetric"
    combination_type = "concat"


MAXLEN = 25



if(method == "gcn" or method == "GraphSage" ):
    tf.compat.v1.disable_eager_execution()
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


flags = tf.app.flags
FLAGS = flags.FLAGS


if(method == "gcn"):

    flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
    flags.DEFINE_integer('Rnnepochs', 15 , 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.4, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 300, 'Tolerance for early stopping (# of epochs).')


if(method =="GraphSage"):

    flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.27, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 150, 'Tolerance for early stopping (# of epochs).')

if(method =="rnn"):

    flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
    flags.DEFINE_integer('Rnnepochs', 5 , 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 8, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.4, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 300, 'Tolerance for early stopping (# of epochs).')
 


val_acc =[]
# train_acc = []
test_acc = []
iters = []
cost_res = [] 

epsilon = 0.01


def Sparse_To_Tuple_Converter(sparse_mx , convSparse ):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx



def PreprocessFeatures(features  ,  norm_features ):

    if( norm_features):

        rowsum = np.array(features.sum(1)) + epsilon
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        convSparse = True 

    else:

        rowsum = np.ones( np.shape( features)[0] )
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        convSparse = True

    return Sparse_To_Tuple_Converter(features, convSparse)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def normalize_adj(adj , getNorm , normalization_type , aggregation_type ):



    if( aggregation_type == "sum"):

        adj = sp.coo_matrix(adj)
        AdjMat = adj

        return   AdjMat     

    if( normalization_type == "symmetric" ):
        if( aggregation_type =="mean" ):


            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            AdjMat = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    if(normalization_type == "row" ):

        if( aggregation_type =="mean" ):

            adj = sp.coo_matrix(adj)
            rowsum = np.array(adj.sum(1))
            G = np.eye(len(rowsum))
            for i in range(len(rowsum)):
                G[i][i] = rowsum[i][0]

            d_inv_row = np.linalg.inv(G) #.flatten()
            d_inv_row[np.isinf(d_inv_row)] = 0
            print( np.shape(adj))
            print(np.shape(d_inv_row))
            d_inv_row = sp.csr_matrix(d_inv_row)
            AdjMat = adj.dot(d_inv_row).tocoo()

    if( normalization_type == "col"):

        adj = sp.coo_matrix(adj)
        colsum = np.array(adj.sum(0))
        d_inv_col = np.power(colsum, -1).flatten()
        d_inv_col[np.isinf(d_inv_col)] = 0
        d_inv_col_diag = sp.diags(d_inv_col)
        AdjMat = adj.dot(d_inv_col_diag)

    return AdjMat





def PreprocessAdj(adj , aggregation_type , normalization_type , method):


    if( method == "gcn"):
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]) , True , normalization_type , aggregation_type )

    if(method =="GraphSage"):

        adj_normalized = normalize_adj(adj  , True , normalization_type , aggregation_type )


    return Sparse_To_Tuple_Converter(adj_normalized , convSparse= True)



val =0 


if( method == "gcn" or method =="GraphSage"):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data()

if(method == "rnn"):

    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)

    x_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,  maxlen=MAXLEN)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAXLEN)  
    y_train = tf.cast( tf.reshape( y_train , ( np.shape(y_train)[0] , 1 )  )  , dtype=tf.float32)
    y_test = tf.cast(tf.reshape( y_test , ( np.shape(y_test)[0] , 1 ) ) , dtype=tf.float32)

    # print(np.shape(y_test))
    # print("------------")
    # y_train = pd.get_dummies(y_train, columns = [0, 1])
    # y_test = pd.get_dummies(y_test, columns = [0, 1])



if( method =="gcn" or method == "GraphSage"):
    features = PreprocessFeatures(features , True)
    support = [PreprocessAdj(adj , aggregation_type , normalization_type , method )]



num_supports = 1
model_func = GNN



if( method == "gcn" or method =="GraphSage"):

    placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

# if(method == "rnn"):

#     placeholders = {
#     'train_data':  tf.placeholder(tf.float32, shape=( len(train_x),  maxlen )  ) , 
#     'labels': tf.placeholder(tf.float32, shape=(None, 2 ) ),
#     'dropout': tf.placeholder_with_default(0., shape=()),
#     }



if( method == "GraphSage"  or method == "gcn" ):
    model = model_func(placeholders, input_dim=features[2][1], method=method ,  aggregationType=aggregation_type , combinationType=combination_type,  logging=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


# if(method == "rnn"):

# model = RNNTrain( , input_dim=  maxlen , method=method ,  aggregationType=aggregation_type , combinationType=combination_type,  logging=True)






cost_val = []

if( method == "gcn" or method  == "GraphSage"):

    start = 0
    end = 0 

    def construct_feed_dict(features, support, labels, labels_mask, placeholders):
        """Construct feed dictionary."""
        feed_dict = dict()
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['labels_mask']: labels_mask})
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support)) })
        feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
        return feed_dict

    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    def construct_feed_dict_GS(features, support, labels, labels_mask, placeholders , start , window ):
        """Construct feed dictionary."""
        feed_dict = dict()
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['labels_mask']: labels_mask})
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support)) })
        feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
        return feed_dict

    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        start += epoch*10
        window = 200

        if( method =="gcn"):
            feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders )
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        if(method == "GraphSage"):

            feed_dict = construct_feed_dict_GS(features, support, y_train, train_mask, placeholders , start , window )





        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)



        iters.append(epoch)
        val_acc.append(acc)
        # train_acc.append(outs[2])
        cost_res.append(outs[1])

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Results ")

    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("***************************")
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))


    # plt.plot(iters , val_acc , label="validation accuracy" )
    # # plt.plot(iters , train_acc , label="train accuracy")
    # plt.plot( iters, cost_res , label="Cost")
    # plt.xlabel("Iterations")
    # plt.ylabel("Cost/accuracy")
    # plt.legend()
    # plt.show()


m = 32
LayerDense1 = tf.keras.layers.Dense(m) # lyr_a
LayerDense2 = tf.keras.layers.Dense(1)


def GetAccuracy(val_x,val_y ,  getacc ):

    # if(getacc):  ## Debugginf


    val_logits, losses = [], []
    Steps = val_x.shape[1]
    for i in range(0, len(val_x), 256):

        batch_x = val_x[i:i+256]
        batch_y = val_y[i:i+256]
        A = tf.repeat(H_mat, batch_x.shape[0], 0)
        x = FeatureEmbedding(batch_x)
        logits = None
        for t in range(Steps):
            # # RNNmodel2 = RNNmodel(x[:, t ,:], A , m , LayerDense1 , LayerDense2 ) 
            # logits , A2 = RNNmodel.GetVals(  x[:, t ,:] , A  , m , LayerDense1 , LayerDense2  ) 
            
            # RNNmodel = RNNTrain(  x[:, t ,:] , A  , m , LayerDense1 , LayerDense2 ) 
            # logits , A = RNNmodel.GetVals(  x[:, t ,:] , A  , m , LayerDense1 , LayerDense2  ) 
            comb = LayerDense1(tf.concat([A, x[:, t ,:]], 1)) ## combine and pass through dense layer 
            A = tf.nn.tanh(comb) ## apply tanh activation 
            logits = LayerDense2(A) ## pass through another neuron and return value 
            # RNNmodel.RNNFeed( x[:, t ,:] , A  , m  )

        val_logits.extend(list(logits.numpy().flatten()))
    acc = tf.math.reduce_mean(tf.keras.metrics.binary_accuracy(val_y.reshape(-1, 1), np.array(val_logits).reshape(-1, 1), threshold= 0.50))
    return acc



if( method == "rnn" ):


    H_mat =  tf.constant(tf.random.normal([1, m]))
    Steps = x_train.shape[-1]
    dropout = 0.4 
    FeatureEmbedding = tf.keras.layers.Embedding(vocabulary_size+1, 32, input_length = MAXLEN)

    AdamOptimizer = tf.keras.optimizers.Adam(learning_rate= 0.01)


    def BinaryCrossEntropyLoss(logits,labels):

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels)
        return tf.reduce_mean(loss)

    RNNmodel = RNNTrain()  

    for epoch in range(FLAGS.Rnnepochs):
        losses = []

        for i in range(0, len(x_train), 256):

            BatchX = x_train[i:i+256]
            BatchY = y_train[i:i+256]

            A = tf.constant(tf.repeat(H_mat, BatchX.shape[0], 0))

            with tf.GradientTape() as tp:

                x = FeatureEmbedding(BatchX)

                for t in range(Steps):

                    # RNNmodel = RNNTrain(  x[:, t ,:] , A  , m , LayerDense1 , LayerDense2 ) 
                    logits , A = RNNmodel.GetVals(  x[:, t ,:] , A  , m , LayerDense1 , LayerDense2  ) 
                    # comb = LayerDense1(tf.concat([A, x[:, t ,:]], 1)) ## combine and pass through dense layer 
                    # A = tf.nn.tanh(comb) ## apply tanh activation 
                    # logits = LayerDense2(A) ## pass through another neuron and return value 
                    # RNNmodel.RNNFeed( x[:, t ,:] , A  , m  )

                # loss = BinaryCrossEntropyLoss( tf.cast( logits , dtype=tf.float32),  tf.cast( BatchY , dtype=tf.float32 )  )
                loss = BinaryCrossEntropyLoss(logits ,  BatchY   )
                # print( loss.numpy())
            # losses.append( loss.eval(session =tf.compat.v1.Session()) )
            variables = tp.watched_variables()

            grads = tp.gradient(loss, variables ) #, 
                # unconnected_gradients=tf.UnconnectedGradients.ZERO)

            AdamOptimizer.apply_gradients(zip(grads,variables))
        getacc = True  
        test_accuracy = GetAccuracy(x_test,y_test ,  getacc )
        print(  "Epoch Number " + str(epoch) + " Test Accuracy  = "  +  str(  test_accuracy.numpy() )  )

