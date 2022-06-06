import tensorflow.compat.v1 as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS



def Glorot_set_vals(shape, setval , name=None):
    """ Glorot initilization for weights and biases """
    Range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-Range, maxval=Range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def DropoutSparse(x, keep_prob, noise_shape , issparse):
    """ implentation of the dropout function for sparse tensors """

    if issparse:
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(x, dropout_mask)
    else:

        print(" Incorrect Sparsity declaration ")
    return pre_out * (1./keep_prob)


def DotProd(x, y, aggregation_type ,  sparse=False  ):
    """ returns the dor product and support both sparse and dense matrices """
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def CrossEntropyMasked(preds, labels, mask, getloss):
    """Implementation of the Loss function with masking """

    if( getloss):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
    else:

        print("Enable Loss ")
    return tf.reduce_mean(loss)

_LAYER_UIDS = {}

def AccuracyMasked(preds, labels, mask, getacc ):
    """ implentation of Accuracy with masking."""

    if(getacc):
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
    else:

        print("Enable Accuracy Masked")
    return tf.reduce_mean(accuracy_all)




class GNN(object):
    """Class to implement various operatrions of a GNN """

    def __init__(self, placeholders, input_dim, method , aggregationType, combinationType ,  **kwargs):

        # super(GCN, self).__init__(**kwargs)


        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None


        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.aggregationType = aggregationType
        self.method = method
        self.combinationType = combinationType
        self.build()


    def build(self):
            """ Wrapper for _build() """
            with tf.variable_scope(self.name):
                self._build()

            # Build sequential layer model
            self.activations.append(self.inputs)
            for layer in self.layers:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
            self.outputs = self.activations[-1]

            # Store model variables for easy access
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.vars = {var.name: var for var in variables}

            # Build metrics
            self._loss()
            self._accuracy()

            self.opt_op = self.optimizer.minimize(self.loss)




    def _loss(self):
        # Get Loss value 
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += CrossEntropyMasked(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'] , True )


    def _accuracy(self):
    	self.accuracy = AccuracyMasked(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'] , True )

    def _build(self):

        ## Call the graph convolution layers with the necessary specifications 

        self.layers.append(GraphConv(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            method = self.method , 
                                            aggregationType = self.aggregationType , 
                                            combinationType = self.combinationType ,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConv(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            method = self.method , 
                                            aggregationType = self.aggregationType , 
                                            combinationType = self.combinationType ,                                            
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))






def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):


    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])




class GraphConv(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, method ,aggregationType , combinationType ,  dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu,   bias=False,   
                 featureless=False, **kwargs):
        super(GraphConv, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.AdjMat = placeholders['support']
        self.aggregationType = aggregationType
        self.combinationType = combinationType
        self.method = method



    
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.AdjMat)):
                self.vars['weights_' + str(i)] = Glorot_set_vals([input_dim, output_dim], True , 
                                                        name='weights_' + str(i))

                self.vars['bias_' + str(i)] = Glorot_set_vals([input_dim, output_dim], True , 
                                                        name='weights_' + str(i))

                self.vars['weights2_' + str(i)] = Glorot_set_vals([2*output_dim, output_dim], True , 
                                                        name='weights_' + str(i))


        if self.logging:
            self._log_vars()


    def RNNFeed( self, batch_x , HiddenState , dim , name , LayerDense1 , LayerDense2 ):

        ## feedforward layer of RNN as GNN 


        comb = LayerDense1(tf.concat([HiddenState, batch_x], 1)) ## combine and pass through dense layer 
        HiddenState1 = tf.nn.tanh(comb) ## apply tanh activation 
        logits = LayerDense2(HiddenState1) ## pass through another neuron and return value 

        # with tf.variable_scope('rnn' + '_vars'):
        #     self.vars['weights_3' +   str(name)] = glorot([(comb).shape[1], dim], name='weights_' + str(name) )
        #     self.vars['weights_4' + str(name)] = glorot([dim , 1], name='weights_' +  str(name))

        # comb = dot(comb, self.vars['weights_3'  + str(name)], "mean" , 
        #                       sparse= False )
        # HiddenState = tf.nn.tanh(comb)

        # logits = dot(HiddenState, self.vars['weights_4'  + str(name) ], "mean" , 
        #                       sparse= False )

        return logits , HiddenState1



    def _call(self , inputs ):

        x =inputs 

        if self.sparse_inputs :

            x = DropoutSparse(x, 1-self.dropout, self.num_features_nonzero , self.sparse_inputs )

        else:
            x = tf.nn.dropout(x, 1-self.dropout)




        if( self.method == "gcn"): ## GCN implementaion 

            ## aggregation and combination is performed using matrix operations  

            ValAcrossAdj = list()
            for i in range(len(self.AdjMat)):

                val1 = DotProd(x, self.vars['weights_' + str(i)],self.aggregationType , 
                              sparse=self.sparse_inputs)

                val2 = DotProd(self.AdjMat[i], val1, self.aggregationType  , sparse=True )

                ValAcrossAdj.append(val2)

            output = tf.add_n(ValAcrossAdj)

            return self.act(output)  ## apply activation and return 


        if( self.method == "GraphSage"):

            ValAcrossAdj = list()

            ## perform aggregarion using the pure adjecency matrix 
            for i in range(len(self.AdjMat)):

                val1 = DotProd(x, self.vars['weights_' + str(i)],self.aggregationType , 
                              sparse=self.sparse_inputs)

                val2 = DotProd(self.AdjMat[i], val1, self.aggregationType  , sparse=True )
                ValAcrossAdj.append(val2)

            output = tf.add_n(ValAcrossAdj)

            ## GraphSage uses concatination for combination 

            if( self.combinationType == "concat"):

                if self.sparse_inputs:

                    data = list() 
                    val1 = DotProd(x, self.vars['bias_' + str(i)] ,  self.aggregationType  ,
                              sparse=self.sparse_inputs)
                    data.append(val1)
                    data.append(output)
                    output2 = tf.add_n(data) 
                    output = tf.concat([ output  , output2  ] , 1)

                    output = DotProd(output, self.vars['weights2_' + str(i)], self.aggregationType  ,
                              sparse= False)

                else :

                    data = list() 
                    val1 = DotProd(x, self.vars['bias_' + str(i)] ,  self.aggregationType  ,
                              sparse=self.sparse_inputs)
                    data.append(val1)
                    data.append(output)
                    output2 = tf.add_n(data) 

                    output = tf.concat([ output  , output2  ] , 1)

                    output = DotProd(output, self.vars['weights2_' + str(i)], self.aggregationType ,
                              sparse=self.sparse_inputs)

            return self.act(output) 



            
class RNNTrain( GraphConv ):

    ## class to implemnent RNN 

    def __init__(self  ,   **kwargs ):

        # super(RNNTrain, self).__init__(**kwargs)
        self.vars = {}
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= "rnn" )
        self.vars = {var.name: var for var in variables}

        self.xt = None 
        self.HiddenState = None  
        self.logits = None

        num= np.random.randint(10000)

        # self.logits, self.HiddenState = self.RNNFeed( self.xt , self.HiddenState , dim , num , lyr_a , lyr_y )


    def GetVals(self , xt , HiddenState , dim , lyr_a , lyr_y ):
        # num= 0
        # comb = lyr_a(tf.concat([HiddenState, xt], 1)) ## combine and pass through dense layer 
        # HiddenState1 = tf.nn.tanh(comb) ## apply tanh activation 
        # logits = lyr_y(HiddenState1) ## pass through another neuron and return value 
        # return logits, HiddenState1
        num = 0

        return self.RNNFeed( xt , HiddenState , dim , num , lyr_a , lyr_y )

        # return self.logits , self.HiddenState 



