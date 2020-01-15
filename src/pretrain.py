import numpy as np
import pandas as pd 
import sys, os, random
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from keras_bert import get_base_dict, get_model, gen_batch_inputs, load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, compile_model
import nltk, tqdm
from keras.utils import plot_model
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Lambda, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf

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
def collect_inputs(abstract, tokenizer):
    done = 0
    datas = []
    for i in abstract:
        j = i.replace('$$$', ' ')
        k = tokenizer.tokenize(j)
        new_input = [k[1:-1], [ ]]
        datas.append(new_input)

        done += 1
        sys.stdout.write(str(done) + '\r')
    return datas

def generate_input_by_batch(X, batch_size = 4):
    idx = random.sample(range(len(X)), batch_size)
    X_out = [X[i] for i in idx]
    return X_out

def pretrain_model():
    
	df = pd.read_csv('../data/task2_trainset.csv', dtype = str)
	df_2 = pd.read_csv('../data/task2_public_testset.csv', dtype = str)
	abstract_1 = df.values[:, 2]
	abstract_2 = df_2.values[:, 2]   
	token_dict = load_vocabulary(dict_path)
	token_list = list(token_dict.keys())
	tokenizer = Tokenizer(token_dict)
	X_1 = collect_inputs(abstract_1, tokenizer)
	X_2 = collect_inputs(abstract_2, tokenizer)
	X = X_1 + X_2
	print(len(X))

	model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training = True, trainable = True, seq_len = 512)
	compile_model(model)

	def _generator():
		while True:
			yield gen_batch_inputs(generate_input_by_batch(X), token_dict, token_list, seq_len = 512, mask_rate = 0.3)

	opt_filepath = sys.argv[1]
	checkpoint = ModelCheckpoint(opt_filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min', save_weights_only = True)
	reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1, mode = 'auto', min_delta = 0.1, cooldown = 10, min_lr = 1e-10)
	es = EarlyStopping(monitor = 'val_loss', patience = 50)
	callbacks_list = [ checkpoint, es, reduce_lr ]

	model.fit_generator(generator = _generator(), steps_per_epoch = 500, epochs = 5000, validation_data = _generator(), validation_steps = 200, callbacks = callbacks_list)


if __name__ == '__main__':
    # testing 
    pretrain_model()
