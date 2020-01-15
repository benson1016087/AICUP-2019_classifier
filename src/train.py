import numpy as np
import pandas as pd 
import sys, os, random
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from keras_bert import get_base_dict, get_model, gen_batch_inputs, load_vocabulary, load_trained_model_from_checkpoint, Tokenizer
import nltk, tqdm
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Lambda, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import regularizers
import tensorflow.compat.v1 as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.tensorflow_backend.set_session(sess)
'''

#tensorflow.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)

df = pd.read_csv('../data/task2_trainset.csv', dtype = str)
cate = df.values[:, -1] 

# generating Y
Y = np.zeros((cate.shape[0], 4))
name = ['THEORETICAL', 'ENGINEERING', 'EMPIRICAL', 'OTHERS']
for i in range(cate.shape[0]):
    for c in cate[i].split(' '):
        for idx in range(4):
            if c == name[idx]:
                Y[i, idx] += 1

# generating X
abstract = df.values[:, 2]

config_path = '../data/bert_config.json'
checkpoint_path = '../data/bert_model.ckpt'
dict_path = '../data/vocab.txt'

# collect words
token_dict = load_vocabulary(dict_path)
tokenizer = Tokenizer(token_dict)
input_data = []
input_seg = []
seq_len = 512 # maximum should be 638, while bert-BASE only support up to 512
done = 0
for i in abstract:
    j = i.replace('$$$', ' ')
    idx, seg = tokenizer.encode(j, max_len = seq_len)
    input_data.append(idx)
    input_seg.append(seg)
    done += 1
    sys.stdout.write(str(done) + '\r')
X = np.asarray(input_data)
seg = np.asarray(input_seg)
random.seed(246601)
rd_seq = random.sample(range(abstract.shape[0]), abstract.shape[0])
Y_train = Y[rd_seq[:-1000]]
Y_val = Y[rd_seq[-1000:]]
X_train = X[rd_seq[:-1000]]
X_val = X[rd_seq[-1000:]]
seg_train = seg[rd_seq[:-1000]]
seg_val = seg[rd_seq[-1000:]]
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

# getting model:
def f1_acc(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis = 0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis = 0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis = 0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis = 0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis = 0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis = 0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return 1 - K.mean(f1)

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training = True, trainable = True, seq_len = seq_len)
model.load_weights(sys.argv[1])
Input_layer = model.inputs[:2]
x = model.layers[-9].output
x = BatchNormalization()(x)
x = Lambda(lambda model: model[:, 0])(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
Output_layer = Dense(3, activation = 'sigmoid')(x)
model = Model(Input_layer, Output_layer)

opt_filepath = sys.argv[2]
checkpoint = ModelCheckpoint(opt_filepath, monitor = 'val_f1_acc', verbose = 1, save_best_only = True, mode = 'max', save_weights_only = True) 
callbacks_list = [ checkpoint ]

#model.fit([X_train, seg_train], Y_train[:, :-1], batch_size = 6, epochs = 3, callbacks = callbacks_list, validation_data = ([X_val, seg_val], Y_val[:, :-1])) 
# set latter layer to trainable
for layer in model.layers[:-5]:
	layer.trainable = False
for layer in model.layers:
	print(layer, layer.trainable)
model.compile(loss = f1_loss, optimizer = Adam(1e-4), metrics = [f1_acc, f1_loss])
model.fit([X_train, seg_train], Y_train[:, :-1], batch_size = 256, epochs = 10, callbacks = callbacks_list, validation_data = ([X_val, seg_val], Y_val[:, :-1])) 

target = 79
for i in range(4):
	
	cnt = 0
	
	for layer in model.layers:
		if cnt > target:
			layer.trainable = True
		else: 
			layer.trainable = False
		cnt += 1

	target += 8

	model.compile(loss = f1_loss, optimizer = Adam(1e-5), metrics = [f1_acc, f1_loss])
	model.fit([X_train, seg_train], Y_train[:, :-1], batch_size = 12 + 4 * i, epochs = 2 * (i + 1), callbacks = callbacks_list, validation_data = ([X_val, seg_val], Y_val[:, :-1])) 

for layer in model.layers:
	print(layer, layer.trainable)

model.compile(loss = f1_loss, optimizer = Adam(1e-4), metrics = [f1_acc, f1_loss])
model.summary()
model.fit([X_train, seg_train], Y_train[:, :-1], batch_size = 256, epochs = 50, callbacks = callbacks_list, validation_data = ([X_val, seg_val], Y_val[:, :-1])) 
