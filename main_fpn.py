import sys
import os
import csv
import numpy as np
import cv2
import math
import pose_utils
import os
import myparse
import renderer_fpn

# SBM Py2 to py3, linux to win, tf2->1 conversion:
# - Added parentheses to all prints.
# - Req lmdb, ConfigParser,
# - ConfigParser lib name changed to configparser.
# - Tensorflow: import tensorflow as tf -> import tensorflow.compat.v1 as tf, disabled eager execution.
# - get_Rts: Changed np.load to allow pickle data by def locally, with np_load.
#            Manually exported PAM array from py2 to dict of weights since pickle is incompatible.
#            Added function load_net_data to read it.
#            Changed lmdb of 1e12 (1tb) bytes to 1e9, since windows seems to cache that much mem.
#            Need to add "encode" in lmdb put / get.
# - pose_model: Unclear usage of shape[1].value - already int.
# - renderer_fpn: Another case of required encode.
# - myutil: Logically erroneous access to dict value [0].
# - renderer: Fixed known issue with cv2.remap limit. Fixed int division.

# SBM Parameters fixed.
input_file = "inputSBM.csv"

## To make tensorflow print less (this can be useful for debug though)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import ctypes; 
print ('> loading getRts')
import get_Rts as getRts
######## TMP FOLDER #####################
_tmpdir = './tmp/'#os.environ['TMPDIR'] + '/'
print ('> make dir')
if not os.path.exists( _tmpdir):
    os.makedirs( _tmpdir )
#########################################
##INPUT/OUTPUT
# input_file = str(sys.argv[1]) #'input.csv' SBM
outpu_proc = 'output_preproc.csv'
output_pose_db =  './output_pose.lmdb'
output_render = './output_render'
#################################################
print ('> network')
_alexNetSize = 227
_factor = 0.25 #0.1

# ***** please download the model in https://www.dropbox.com/s/r38psbq55y2yj4f/fpn_new_model.tar.gz?dl=0 ***** #
model_folder = './fpn_new_model/'
model_used = 'model_0_1.0_1.0_1e-07_1_16000.ckpt' #'model_0_1.0_1.0_1e-05_0_6000.ckpt'
lr_rate_scalar = 1.0
if_dropout = 0
keep_rate = 1
################################
data_dict = myparse.parse_input(input_file)
## Pre-processing the images 
print ('> preproc')
pose_utils.preProcessImage( _tmpdir, data_dict, './',\
                            _factor, _alexNetSize, outpu_proc )
## Runnin FacePoseNet
print ('> run')
## Running the pose estimation
getRts.esimatePose( model_folder, outpu_proc, output_pose_db, model_used, lr_rate_scalar, if_dropout, keep_rate, use_gpu=False )


renderer_fpn.render_fpn(outpu_proc, output_pose_db, output_render)
