import numpy as np
import pandas as pd
from keras.models import load_model
import sys, os, random 
from keras_bert import get_base_dict, get_model, gen_batch_inputs, load_vocabulary, load_trained_model_from_checkpoint, Tokenizer
from keras.layers import Lambda, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K
import tensorflow.compat.v1 as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.tensorflow_backend.set_session(sess)
'''

config_path = '../data/bert_config.json'
checkpoint_path = '../data/bert_model.ckpt'
dict_path = '../data/vocab.txt'

df = pd.read_csv('../data/task2_public_testset.csv', dtype = str)
abstract = df.values[:, 2]

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

def get_model():
    # getting model:
	model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training = True, trainable = True, seq_len = seq_len)
	Input_layer = model.inputs[:2]
	x = model.layers[-9].output
	x = BatchNormalization()(x)
	x = Lambda(lambda model: model[:, 0])(x)
	x = BatchNormalization()(x)
	x = Dropout(0.2)(x)
	Output_layer = Dense(3, activation = 'sigmoid')(x)
	model = Model(Input_layer, Output_layer)
	model.load_weights(sys.argv[1])
	return model

model = get_model()
Y_pred = model.predict([X, seg], verbose = 1)
'''
print(Y_pred_by_cate)
Y_pred_by_cate = np.load('./Y_test_pred_by_category.npy')
Y_pred = (3*Y_pred + Y_pred_by_cate) / 4
'''
Y_pred = ( Y_pred > 0.5 )
other_pred = np.sum(Y_pred, axis = 1) < 0.9
Y = np.hstack((Y_pred, other_pred.reshape(-1, 1))).astype('int')

opt_path = sys.argv[2]
f = open(opt_path, 'w')
wt_str = 'order_id,THEORETICAL,ENGINEERING,EMPIRICAL,OTHERS\n'
f.write(wt_str)
for i in range(Y_pred.shape[0]):
    wt_str = 'T' + str(i+1).zfill(5) + ','
    wt_str += "{},{},{},{}\n".format(Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3])
    f.write(wt_str)

for i in range(Y_pred.shape[0]):
    wt_str = 'T' + str(i + 20001).zfill(5) + ',0,0,0,0\n'
    f.write(wt_str)
