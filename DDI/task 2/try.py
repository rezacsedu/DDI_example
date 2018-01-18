#http://stackoverflow.com/questions/31796909/how-to-add-a-confusion-matrix-to-theano-examples
from conv_net_sentence import *
import theano
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
from cnn_util import *
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")   
#THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -static -word2vec
theano.config.mode = 'FAST_RUN'
theano.config.device = 'cpu'
theano.config.floatX = 'float32'

#### #### #### #### #### #### #### #### #### conv_net_sentence.main #### #### #### #### #### #### #### 

data = get_data(fname = "features.pickle")
all_data, W, max_l = make_idx_data_cv(data, k=50, filter_h=5)
execfile("conv_net_classes.py")    

datasets = split_train_test(all_data)
#### #### #### #### #### #### #### #### #### train_conv_net #### #### #### #### #### #### #### 
# parameters
U = W
lr_decay=0.95
filter_hs=[3,4,5]
conv_non_linear="relu"
hidden_units=[100,5]
shuffle_batch=True
n_epochs=25
sqr_norm_lim=9
non_static=True
batch_size=50
dropout_rate=[0.5]
img_w=50
activations=[Iden]

rng = np.random.RandomState(3435)
img_h = len(datasets[0][0])-1  
filter_w = img_w    
feature_maps = hidden_units[0]
filter_shapes = []
pool_sizes = []
for filter_h in filter_hs:
    filter_shapes.append((feature_maps, 1, filter_h, filter_w))
    pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))

#define model architecture
index = T.lscalar()
x = T.matrix('x')   
y = T.ivector('y')
Words = theano.shared(value = U, name = "Words")
zero_vec_tensor = T.vector()
zero_vec = np.zeros(img_w)
set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
conv_layers = []
layer1_inputs = []
for i in xrange(len(filter_hs)):
    filter_shape = filter_shapes[i]
    pool_size = pool_sizes[i]
    conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                            filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
    layer1_input = conv_layer.output.flatten(2)
    conv_layers.append(conv_layer)
    layer1_inputs.append(layer1_input)


layer1_input = T.concatenate(layer1_inputs,1)
hidden_units[0] = feature_maps*len(filter_hs)    
classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)

#define parameters of the model and update functions using adadelta
params = classifier.params     
for conv_layer in conv_layers:
    params += conv_layer.params


if non_static:
    #if word vectors are allowed to change, add them as model parameters
    params += [Words]


cost = classifier.negative_log_likelihood(y) 
dropout_cost = classifier.dropout_negative_log_likelihood(y)           
grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

#shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
#extra data (at random)
np.random.seed(3435)
if datasets[0].shape[0] % batch_size > 0:
    extra_data_num = batch_size - datasets[0].shape[0] % batch_size
    train_set = np.random.permutation(datasets[0])   
    extra_data = train_set[:extra_data_num]
    new_data=np.append(datasets[0],extra_data,axis=0)
else:
    new_data = datasets[0]


new_data = np.random.permutation(new_data)
n_batches = new_data.shape[0]/batch_size
n_train_batches = int(np.round(n_batches*0.9))
#divide train set into train/val sets 
test_set_x = datasets[1][:,:img_h] 
test_set_y = np.asarray(datasets[1][:,-1],"int32")
train_set = new_data[:n_train_batches*batch_size,:]
val_set = new_data[n_train_batches*batch_size:,:]     
train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
n_val_batches = n_batches - n_train_batches
val_model = theano.function([index], classifier.errors(y),
     givens={
        x: val_set_x[index * batch_size: (index + 1) * batch_size],
         y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                            allow_input_downcast=True)
        
#compile theano functions to get train/val/test errors
test_model = theano.function([index], classifier.errors(y),
         givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
             y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                             allow_input_downcast=True)  


train_model = theano.function([index], cost, updates=grad_updates,
      givens={
        x: train_set_x[index*batch_size:(index+1)*batch_size],
          y: train_set_y[index*batch_size:(index+1)*batch_size]},
                              allow_input_downcast = True)   


test_pred_layers = []
test_size = test_set_x.shape[0]
test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
for conv_layer in conv_layers:
    test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
    test_pred_layers.append(test_layer0_output.flatten(2))


test_layer1_input = T.concatenate(test_pred_layers, 1)


########## Calculation for precision recall and F1 score
test_y_pred = classifier.predict(test_layer1_input)

test_error = T.mean(T.neq(test_y_pred, y))

test_model_all = theano.function([x,y], test_error, allow_input_downcast = True)   

test_y = theano.function([x],test_y_pred,allow_input_downcast = True)

#start training over mini-batches
print '... training'
epoch = 0
best_val_perf = 0
val_perf = 0
test_perf = 0       
cost_epoch = 0    
# while (epoch < n_epochs):

for minibatch_index in np.random.permutation(range(n_train_batches)):
    cost_epoch = train_model(minibatch_index)
    set_zero(zero_vec)


train_losses = [test_model(i) for i in xrange(n_train_batches)]
train_perf = 1 - np.mean(train_losses)
val_losses = [val_model(i) for i in xrange(n_val_batches)]

val_perf = 1- np.mean(val_losses)                        
print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
if val_perf >= best_val_perf:
    best_val_perf = val_perf
    test_loss = test_model_all(test_set_x,test_set_y)        
    test_perf = 1- test_loss         