from __future__ import print_function

import numpy as np
import os, sys, gzip
import time
import keras

# from tf_mthcan import hcan
from multitask_hcan import hcan

import argparse

import p3b4 as bmk
import candle
from data_handler import DataHandler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument( '--n_folds', type= int, default= 0, help= 'fold' )
parser.add_argument( '--fold', type= int, default= 0, help= 'fold' )
parser.add_argument( '--tasks', type= str, default= [ 'site' , 'subsite', 'laterality', 'histology', 'behavior', 'grade' ], help= 'fold' )
parser.add_argument( '--preset_weights', action= 'store_true', default= False )
parser.add_argument( '--save_weights', action= 'store_true', default= False )
args = parser.parse_args()
preset_weights = args.preset_weights
save_weights = args.save_weights

def initialize_parameters():

    # Build benchmark object
    p3b3Bmk = bmk.BenchmarkP3B3(bmk.file_path, 'p3b4_default_model.txt', 'keras',
    prog='p3b4_baseline', desc='Hierarchical Convolutional Attention Networks for data extraction from clinical reports - Pilot 3 Benchmark 4')
    
    # Initialize parameters
    gParameters = candle.initialize_parameters(p3b3Bmk)
    #bmk.logger.info('Params: {}'.format(gParameters))

    return gParameters


def fetch_data(gParameters):
    """ Downloads and decompresses the data if not locally available.
        Since the training data depends on the model definition it is not loaded,
        instead the local path where the raw data resides is returned
    """

    path = gParameters['data_url']
    fpath = candle.fetch_file(path + gParameters['train_data'], 'Pilot3', untar=True)
    
    return fpath

def run(gParameters):

    #print( gParameters )
    #fpath = fetch_data(gParameters)
    # Get default parameters for initialization and optimizer functions
    kerasDefaults = candle.keras_default_config()

    learning_rate = gParameters[ 'learning_rate' ]
    batch_size = gParameters[ 'batch_size' ]
    epochs = gParameters[ 'epochs' ]
    dropout = gParameters[ 'dropout' ]

    optimizer = gParameters[ 'optimizer' ]
    if optimizer == 0:
        optimizer = 'adam'
    elif optimizer == 1:
        optimizer = 'adadelta'
    elif optimizer == 2:
        optimizer = 'sgd'
    elif optimizer == 3:
        optimizer = 'rmsprop'

    wv_len = gParameters[ 'wv_len' ]
    word_attn_size = gParameters[ 'word_attn_size' ]
    sent_attn_size = gParameters[ 'sent_attn_size' ]

    ### hjy - subsection
    group = 0
    if group == 0:
        ### respiratory - adenocarcinoma
        start_subsite = 118
        end_subsite = 148
        start_hist = 0
        end_hist = 158
    elif group == 1:
        ### female - ductal carcinoma
        start_subsite = 192
        end_subsite = 224
        start_hist = 159
        end_hist = 202

    data_handler = DataHandler( data_name= 'LA_v2', n_folds= args.n_folds )
    data_handler.get_fold( args.fold, data_mode= 'cnn', task= args.tasks )

    train_x = data_handler.train_tokens
    train_y = data_handler.train_y

    test_x = data_handler.test_tokens
    test_y = data_handler.test_y

    wv_mat = data_handler.wv_mat
    wv_mat = np.random.randn( len( wv_mat ), wv_len ).astype( 'float32' ) * 0.1
    
    ### collect sub
    train_x_sub = []
    train_y_sub = []

    for k in range( len( train_x ) ):
        if train_y[ k, 1 ] >= start_subsite and train_y[ k, 1 ] <= end_subsite and train_y[ k, 3 ] >= start_hist and train_y[ k, 3 ] <= end_hist:
            train_x_sub.append( train_x[ k ] )
            train_y_sub.append( train_y[ k ] )

    train_x = np.array( train_x_sub, dtype= 'int32' )
    train_y = np.array( train_y_sub, dtype= 'int32' )

    test_x_sub = []
    test_y_sub = []

    for k in range( len( test_x ) ):
        if test_y[ k, 1 ] >= start_subsite and test_y[ k, 1 ] <= end_subsite and test_y[ k, 3 ] >= start_hist and test_y[ k, 3 ] <= end_hist:
            test_x_sub.append( test_x[ k ] )
            test_y_sub.append( test_y[ k ] )

    test_x = np.array( test_x_sub, dtype= 'int32' )
    test_y = np.array( test_y_sub, dtype= 'int32' )

    num_classes = []
    for task in range( len( train_y[ 0, : ] ) ):
        nc = len( data_handler.label_encoder_list[ task ].classes_ )
        num_classes.append( nc )

    max_vocab = np.max( train_x )
    max_vocab2 = np.max( test_x )
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2
    vocab_size = max_vocab + 1
    vocab = np.random.rand( vocab_size, wv_len )

#    num_classes = []
#    for task in range( len( train_y[ 0, : ] ) ):
#        cat = np.unique( train_y[ :, task ] )
#        num_classes.append( len( cat ) )
#        train_y[ :, task ] = [ np.where( cat == x )[ 0 ][ 0 ] for x in train_y[ :, task ] ]
#        test_y[ :, task ] = [ np.where( cat == x )[ 0 ][ 0 ] for x in test_y[ :, task ] ]
#    num_tasks = len( num_classes )

    train_samples = train_x.shape[ 0 ]
    test_samples = test_x.shape[ 0 ]

    max_lines = 50
    max_words = 30

    train_x = train_x.reshape( ( train_x.shape[ 0 ], max_lines, max_words ) )
    test_x = test_x.reshape( ( test_x.shape[ 0 ], max_lines, max_words ) )

    #optional masking
    min_lines = 30
    min_words = 5
    mask = []
    for i in range(train_samples+test_samples):
        doc_mask = np.ones((1,max_lines,max_words))
        num_lines = np.random.randint(min_lines,max_lines)
        for j in range(num_lines):
            num_words = np.random.randint(min_words,max_words)
            doc_mask[0,j,:num_words] = 0
        mask.append(doc_mask)
    mask = np.concatenate(mask,0)
    
    # train model
#    model = hcan( vocab, num_classes, max_lines, max_words, 
#                  attention_size= attention_size,
#                  dropout_rate = dropout,
#                  lr = learning_rate,
#                  optimizer= optimizer 
#    )

    model = hcan( vocab, num_classes, max_lines, max_words,
                  word_attn_size = word_attn_size,
                  sent_attn_size = sent_attn_size,
                  dropout_rate = dropout,
                  lr = learning_rate,
                  optimizer = optimizer,
                  preset_weight = preset_weights
    )

    ret = model.train(
        train_x, 
        [ 
            np.array( train_y[ :, 0 ] ), 
            np.array( train_y[ :, 1 ] ), 
            np.array( train_y[ :, 2 ] ), 
            np.array( train_y[ :, 3 ] ),
            np.array( train_y[ :, 4 ] ),
            np.array( train_y[ :, 5 ] )
        ], 
        batch_size= batch_size, epochs= epochs,
        validation_data= [ 
        test_x, 
        [ 
            np.array( test_y[ :, 0 ] ), 
            np.array( test_y[ :, 1 ] ), 
            np.array( test_y[ :, 2 ] ), 
            np.array( test_y[ :, 3 ] ),
            np.array( test_y[ :, 4 ] ),
            np.array( test_y[ :, 5 ] )
        ] 
        #[ np.array( test_y[ :, 0 ] ), np.array( test_y[ :, 1 ] ), np.array( test_y[ :, 2 ] ), np.array( test_y[ :, 3 ] ) ]
        ],
        save_weight = save_weights
        )

    
    return ret


def main():

    os.makedirs('./retention_saves', exist_ok=True)
    if args.preset_weights:
        savepath = os.path.join('./retention_saves', 'val_loss_pretrained')
    else:
        savepath = os.path.join('./retention_saves', 'val_loss_scratch')

    gParameters = initialize_parameters()
    avg_loss = run(gParameters)
    print( "Return: ", avg_loss.history[ 'val_loss' ][-1] )
    np.save(savepath, avg_loss.history['val_loss'])

if __name__ == '__main__':
    main()

    # try:
        # K.clear_session()
    # except AttributeError:      # theano does not have this function
        # pass





