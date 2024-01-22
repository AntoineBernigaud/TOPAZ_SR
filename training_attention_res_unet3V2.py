#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from data_generator3V2 import Data_generator, generate_dates, convert_date_format, \
                            load_standardization_data
from models.attention_res_net3V2 import Att_Res_UNet
from net_util import save_model_parameters
import tensorflow as tf
import time
import numpy as np


# In[3]:


tf.keras.utils.set_random_seed(420)
print("GPUs available: ", tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[4]:


t0 = time.time()
experiment_name = "Attention_Res_UNet"

date_min_train = '1995_008'
date_max_train = '2012_001'
#date_max_train = '2013_006'

date_min_valid = '2012_008'
#date_min_valid = '2013_013'
date_max_valid = '2015_361'

days_range = 7 # number of days between two restart
dates_learning = generate_dates(date_min_train, date_max_train, days_range)
#print('date list of learning days:')
#print(dates_learning)
# print('date list of restart converted in cice format:')
# converted_dates = [convert_date_format(date) for date in dates_list]
# print(converted_dates)
#print(f'number of training days: {len(dates_learning)}')
# print(f'To notice: the last date generated in the list is: {dates_list[-1]}')
dates_valid = generate_dates(date_min_valid, date_max_valid, days_range)
#print('date list of validation days:')
#print(dates_valid)
print(f'number of validation days: {len(dates_valid)}')

#
paths = {}
root_data_dir = 'data'
root_output_dir = 'outputs'

paths["data_residuals"] = os.path.join(root_data_dir,"residuals")
paths["data_LR_upsampled"] = os.path.join(root_data_dir,"LR_upsampled")

paths["standard"] = root_data_dir
paths["outputs"] = root_output_dir
paths["model_weights"] = os.path.join(root_output_dir,"Model_weights",experiment_name)
paths["checkpoints"] = os.path.join(root_output_dir,"Model_weights",experiment_name,"CheckpointsV2")
#
for var in paths:
    if os.path.isdir(paths[var]) == False:
        os.system("mkdir -p " + paths[var])
#
file_standardization_LR_upsampled = os.path.join(paths["standard"],"standard_LR_upsampled.h5")
file_standardization_res = os.path.join(paths["standard"],"standard_residuals.h5")
file_standardization_bathy = os.path.join(paths["standard"],"standard_bathy_HR.h5")
file_standardization_ssh_LR_upsampled = os.path.join(paths["standard"],"standard_ssh_LR_upsampled.h5")

file_checkpoints = os.path.join(paths["checkpoints"],"Checkpoints_V2.h5")
#
if os.path.isfile(file_checkpoints) == True:
    os.system("rm " + file_checkpoints)


# In[5]:


## The predictor must be of the form 'varname-layer-xx' or 'varname-cat-xx' if they 
## have an ice category (varname is the same as in the restart and iced files
## layers is between 1 and 50 and cat is between 1 and 5
## exceptions : "tp5_mask", "tp5_bathy", "tp5_lat", "iceumask", "aicenSumMask", "aicenSumMask015", "ssh_upsampled"

list_predictors = ["temp-layer-1","temp-layer-2","temp-layer-3","temp-layer-4","temp-layer-5", \
                "temp-layer-6","temp-layer-7","temp-layer-8","temp-layer-9","temp-layer-10", \
                "saln-layer-1","saln-layer-2","saln-layer-3","saln-layer-4","saln-layer-5", \
                "saln-layer-6","saln-layer-7","saln-layer-8","saln-layer-9","saln-layer-10", \
                "dp-layer-1","dp-layer-2","dp-layer-3","dp-layer-4","dp-layer-5", \
                "dp-layer-6","dp-layer-7","dp-layer-8","dp-layer-9","dp-layer-10", \
                "u-layer-1","u-layer-2","u-layer-3","u-layer-4","u-layer-5", \
                "u-layer-6","u-layer-7","u-layer-8","u-layer-9","u-layer-10", \
                "v-layer-1","v-layer-2","v-layer-3","v-layer-4","v-layer-5", \
                "v-layer-6","v-layer-7","v-layer-8","v-layer-9","v-layer-10", \
                "ubavg-layer-0", "vbavg-layer-0", "pbavg-layer-0", "pbot-layer-0", "psikk-layer-0", \
                "thkk-layer-0", "dpmixl-layer-0",
                "uvel-cat-1", "vvel-cat-1", "scale_factor-cat-1", "swvdr-cat-1",
                "strocnxT-cat-1", "strocnyT-cat-1",
                "stressp_1-cat-1", "stressp_2-cat-1", "stressp_3-cat-1", "stressp_4-cat-1",
                "stressm_1-cat-1", "stressm_2-cat-1", "stressm_3-cat-1", "stressm_4-cat-1",
                "stress12_1-cat-1", "stress12_2-cat-1", "stress12_3-cat-1", "stress12_4-cat-1",
                "iceumask", "frz_onset-cat-1", "fsnow-cat-1", "aicen-cat-1", "aicen-cat-2",
                "aicen-cat-3", "aicen-cat-4", "aicen-cat-5", "vicen-cat-1", "vicen-cat-2",
                "vicen-cat-3", "vicen-cat-4", "vicen-cat-5", "vsnon-cat-1", "vsnon-cat-2",
                "vsnon-cat-3", "vsnon-cat-4", "vsnon-cat-5", "Tsfcn-cat-1", "Tsfcn-cat-2",
                "Tsfcn-cat-3", "Tsfcn-cat-4", "Tsfcn-cat-5", "iage-cat-1", "iage-cat-2",
                "iage-cat-3", "iage-cat-4", "iage-cat-5", "FY-cat-1", "FY-cat-2", "FY-cat-3",
                "FY-cat-4", "FY-cat-5", "alvl-cat-1", "alvl-cat-2", "alvl-cat-3", "alvl-cat-4",
                "alvl-cat-5", "vlvl-cat-1", "vlvl-cat-2", "vlvl-cat-3", "vlvl-cat-4",
                "vlvl-cat-5", "apnd-cat-1", "apnd-cat-2", "apnd-cat-3", "apnd-cat-4", "apnd-cat-5",
                "hpnd-cat-1", "hpnd-cat-2", "hpnd-cat-3", "hpnd-cat-4", "hpnd-cat-5",
                "ipnd-cat-1", "ipnd-cat-2", "ipnd-cat-3", "ipnd-cat-4", "ipnd-cat-5", "dhs-cat-1",
                "dhs-cat-2", "dhs-cat-3", "dhs-cat-4", "dhs-cat-5", "ffrac-cat-1", "ffrac-cat-2",
                "ffrac-cat-3", "ffrac-cat-4", "ffrac-cat-5", "sice001-cat-1", "sice001-cat-2",
                "sice001-cat-3", "sice001-cat-4", "sice001-cat-5", "qice001-cat-1",
                "qice001-cat-2", "qice001-cat-3", "qice001-cat-4", "qice001-cat-5",
                "sice002-cat-1", "sice002-cat-2", "sice002-cat-3", "sice002-cat-4", "sice002-cat-5",
                "qice002-cat-1", "qice002-cat-2", "qice002-cat-3", "qice002-cat-4", "qice002-cat-5",
                "sice003-cat-1", "sice003-cat-2", "sice003-cat-3", "sice003-cat-4", "sice003-cat-5",
                "qice003-cat-1", "qice003-cat-2", "qice003-cat-3", "qice003-cat-4", "qice003-cat-5",
                "sice004-cat-1", "sice004-cat-2", "sice004-cat-3", "sice004-cat-4", "sice004-cat-5",
                "qice004-cat-1", "qice004-cat-2", "qice004-cat-3", "qice004-cat-4", "qice004-cat-5",
                "sice005-cat-1", "sice005-cat-2", "sice005-cat-3", "sice005-cat-4", "sice005-cat-5",
                "qice005-cat-1", "qice005-cat-2", "qice005-cat-3", "qice005-cat-4", "qice005-cat-5",
                "sice006-cat-1", "sice006-cat-2", "sice006-cat-3", "sice006-cat-4", "sice006-cat-5",
                "qice006-cat-1", "qice006-cat-2", "qice006-cat-3", "qice006-cat-4", "qice006-cat-5",
                "sice007-cat-1", "sice007-cat-2", "sice007-cat-3", "sice007-cat-4", "sice007-cat-5",
                "qice007-cat-1", "qice007-cat-2", "qice007-cat-3", "qice007-cat-4", "qice007-cat-5",
                "qsno001-cat-1", "qsno001-cat-2", "qsno001-cat-3", "qsno001-cat-4", "qsno001-cat-5",
                "tp5_bathy", "tp5_mask", "tp5_lat"]

list_targets = ["temp-layer-1","temp-layer-2","temp-layer-3","temp-layer-4","temp-layer-5", \
                "temp-layer-6","temp-layer-7","temp-layer-8","temp-layer-9","temp-layer-10", \
                "saln-layer-1","saln-layer-2","saln-layer-3","saln-layer-4","saln-layer-5", \
                "saln-layer-6","saln-layer-7","saln-layer-8","saln-layer-9","saln-layer-10", \
                "dp-layer-1","dp-layer-2","dp-layer-3","dp-layer-4","dp-layer-5", \
                "dp-layer-6","dp-layer-7","dp-layer-8","dp-layer-9","dp-layer-10", \
                "u-layer-1","u-layer-2","u-layer-3","u-layer-4","u-layer-5", \
                "u-layer-6","u-layer-7","u-layer-8","u-layer-9","u-layer-10", \
                "v-layer-1","v-layer-2","v-layer-3","v-layer-4","v-layer-5", \
                "v-layer-6","v-layer-7","v-layer-8","v-layer-9","v-layer-10", \
                "ubavg-layer-0", "vbavg-layer-0", "pbavg-layer-0", "pbot-layer-0", "psikk-layer-0", \
                "thkk-layer-0", "dpmixl-layer-0",
                "uvel-cat-1", "vvel-cat-1", "scale_factor-cat-1", "swvdr-cat-1",
                "strocnxT-cat-1", "strocnyT-cat-1",
                "stressp_1-cat-1", "stressp_2-cat-1", "stressp_3-cat-1", "stressp_4-cat-1",
                "stressm_1-cat-1", "stressm_2-cat-1", "stressm_3-cat-1", "stressm_4-cat-1",
                "stress12_1-cat-1", "stress12_2-cat-1", "stress12_3-cat-1", "stress12_4-cat-1",
                "iceumask", "frz_onset-cat-1", "fsnow-cat-1", "aicen-cat-1", "aicen-cat-2",
                "aicen-cat-3", "aicen-cat-4", "aicen-cat-5", "vicen-cat-1", "vicen-cat-2",
                "vicen-cat-3", "vicen-cat-4", "vicen-cat-5", "vsnon-cat-1", "vsnon-cat-2",
                "vsnon-cat-3", "vsnon-cat-4", "vsnon-cat-5", "Tsfcn-cat-1", "Tsfcn-cat-2",
                "Tsfcn-cat-3", "Tsfcn-cat-4", "Tsfcn-cat-5", "iage-cat-1", "iage-cat-2",
                "iage-cat-3", "iage-cat-4", "iage-cat-5", "FY-cat-1", "FY-cat-2", "FY-cat-3",
                "FY-cat-4", "FY-cat-5", "alvl-cat-1", "alvl-cat-2", "alvl-cat-3", "alvl-cat-4",
                "alvl-cat-5", "vlvl-cat-1", "vlvl-cat-2", "vlvl-cat-3", "vlvl-cat-4",
                "vlvl-cat-5", "apnd-cat-1", "apnd-cat-2", "apnd-cat-3", "apnd-cat-4", "apnd-cat-5",
                "hpnd-cat-1", "hpnd-cat-2", "hpnd-cat-3", "hpnd-cat-4", "hpnd-cat-5",
                "ipnd-cat-1", "ipnd-cat-2", "ipnd-cat-3", "ipnd-cat-4", "ipnd-cat-5", "dhs-cat-1",
                "dhs-cat-2", "dhs-cat-3", "dhs-cat-4", "dhs-cat-5", "ffrac-cat-1", "ffrac-cat-2",
                "ffrac-cat-3", "ffrac-cat-4", "ffrac-cat-5", "sice001-cat-1", "sice001-cat-2",
                "sice001-cat-3", "sice001-cat-4", "sice001-cat-5", "qice001-cat-1",
                "qice001-cat-2", "qice001-cat-3", "qice001-cat-4", "qice001-cat-5",
                "sice002-cat-1", "sice002-cat-2", "sice002-cat-3", "sice002-cat-4", "sice002-cat-5",
                "qice002-cat-1", "qice002-cat-2", "qice002-cat-3", "qice002-cat-4", "qice002-cat-5",
                "sice003-cat-1", "sice003-cat-2", "sice003-cat-3", "sice003-cat-4", "sice003-cat-5",
                "qice003-cat-1", "qice003-cat-2", "qice003-cat-3", "qice003-cat-4", "qice003-cat-5",
                "sice004-cat-1", "sice004-cat-2", "sice004-cat-3", "sice004-cat-4", "sice004-cat-5",
                "qice004-cat-1", "qice004-cat-2", "qice004-cat-3", "qice004-cat-4", "qice004-cat-5",
                "sice005-cat-1", "sice005-cat-2", "sice005-cat-3", "sice005-cat-4", "sice005-cat-5",
                "qice005-cat-1", "qice005-cat-2", "qice005-cat-3", "qice005-cat-4", "qice005-cat-5",
                "sice006-cat-1", "sice006-cat-2", "sice006-cat-3", "sice006-cat-4", "sice006-cat-5",
                "qice006-cat-1", "qice006-cat-2", "qice006-cat-3", "qice006-cat-4", "qice006-cat-5",
                "sice007-cat-1", "sice007-cat-2", "sice007-cat-3", "sice007-cat-4", "sice007-cat-5",
                "qice007-cat-1", "qice007-cat-2", "qice007-cat-3", "qice007-cat-4", "qice007-cat-5",
                "qsno001-cat-1", "qsno001-cat-2", "qsno001-cat-3", "qsno001-cat-4", "qsno001-cat-5"]

#list_predictors = ["temp-layer-1","temp-layer-2","temp-layer-3","temp-layer-4","temp-layer-5"]
#list_predictors = ["temp-layer-1"] #,"temp-layer-2","temp-layer-3","temp-layer-4","temp-layer-5"]
list_targets = ["temp-layer-1"] #,"vsnon-cat-3"] #,"temp-layer-2","temp-layer-3","temp-layer-4","temp-layer-5"]

print(list_predictors,list_targets)
#
model_params = {"list_predictors": list_predictors,
                "list_targets": list_targets, 
                "dim": (760, 800), # (jdm,idm)
                "cropped_dim": (768, 800), #
                "batch_size": 4,
                "n_filters": [64*(i+1) for i in range(6)], #Ref Cyril: 32
                "activation": "relu",
                "kernel_initializer": "he_normal",
                "batch_norm": True,
                "pooling_type": "Average", # Average
                "dropout": 0,
               }
#
compile_params = {"initial_learning_rate": 0.005, 
                  "decay_steps": 2550, # 2550
                  "decay_rate": 0.5,
                  "staircase": True,
                  "n_epochs": 30, #100,
                  }
#
model_and_compile_params = {**model_params, **compile_params}


# In[6]:


standard_LR_upsampled = load_standardization_data(file_standardization_LR_upsampled)
standard_res = load_standardization_data(file_standardization_res)
standard_bathy = load_standardization_data(file_standardization_bathy)
standard_ssh_LR_upsampled = load_standardization_data(file_standardization_ssh_LR_upsampled)


params_train = {"list_predictors": model_params["list_predictors"],
                "list_labels": model_params["list_targets"],
                "list_dates": dates_learning,
                "standard_res": standard_res,
                "standard_LR_upsampled": standard_LR_upsampled,
                "standard_bathy": standard_bathy,
                "standard_ssh_LR_upsampled": standard_ssh_LR_upsampled,
                "batch_size": model_params["batch_size"],
                "path_data_res": paths["data_residuals"],
                "path_data_LR_upsampled": paths["data_LR_upsampled"],
                "dim": model_params["dim"],
                "cropped_dim": model_params["cropped_dim"],
                "shuffle": True,
                "res_normalization":1,
                }
#
params_valid = {"list_predictors": model_params["list_predictors"],
                "list_labels": model_params["list_targets"],
                "list_dates": dates_valid,
                "standard_res": standard_res,
                "standard_LR_upsampled": standard_LR_upsampled,
                "standard_bathy": standard_bathy,
                "standard_ssh_LR_upsampled": standard_ssh_LR_upsampled,
                "batch_size": model_params["batch_size"],
                "path_data_res": paths["data_residuals"],
                "path_data_LR_upsampled": paths["data_LR_upsampled"],
                "dim": model_params["dim"],
                "cropped_dim": model_params["cropped_dim"],
                "shuffle": True,
                "res_normalization":1,
                }
#
train_generator = Data_generator(**params_train)
valid_generator = Data_generator(**params_valid)


# In[7]:


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = compile_params["initial_learning_rate"],
    decay_steps = compile_params["decay_steps"],
    decay_rate = compile_params["decay_rate"],
    staircase = compile_params["staircase"])

def masked_SSIM(weight, num_channels):
    def SSIM_loss(y_true, y_pred):
        # Assuming y_true and y_pred have shape (batch_number, width, height, channels)
        
        # Extract the true and predicted fields for each channel
        true_fields = [y_true[:, 0:-8, :, i] for i in range(num_channels)]
        pred_fields = [y_pred[:, 0:-8, :, i] for i in range(num_channels)]

        # Multiply true and predicted fields by the weight mask
        masked_true_fields = [true_fields[i] * weight for i in range(num_channels)]
        masked_pred_fields = [pred_fields[i] * weight for i in range(num_channels)]

        # Repeat each field 3 times along the last dimension
        #test = tf.tile(masked_true_fields[0], [1, 1, 1, 3])
        masked_true_fields = [tf.tile(field[..., tf.newaxis], [1, 1, 1, 3]) for field in masked_true_fields]
        masked_pred_fields = [tf.tile(field[..., tf.newaxis], [1, 1, 1, 3]) for field in masked_pred_fields]

        # Compute SSIM loss for each field
        ssim_losses = [1 - tf.image.ssim_multiscale(masked_true_fields[i], masked_pred_fields[i], max_val=1.0) for i in range(num_channels)]

        # Sum the SSIM losses for all channels
        total_ssim_loss = tf.reduce_sum(ssim_losses)

        return total_ssim_loss

    return SSIM_loss


def masked_RMSE(weight):
    def RMSE_loss(y_true, y_pred):
        cropped_y_true = y_true[:, 0:-8, :, 0]
        cropped_y_pred = y_pred[:, 0:-8, :, 0]
        weighted_diff = (cropped_y_true - cropped_y_pred)*weight
        return tf.sqrt(tf.reduce_mean(tf.square(weighted_diff)))
    return RMSE_loss

tp5_mask = np.load( os.path.join(paths["data_residuals"],'tp5mask.npy') )

tp5_mask = 1 - tp5_mask
tp5_mask = tf.convert_to_tensor(tp5_mask, dtype=tf.float32)
tp5_mask = tf.expand_dims(tp5_mask, axis=0)

tp5_rmse_loss = masked_RMSE(tp5_mask)

opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

#
unet_model = Att_Res_UNet(**model_params).make_unet_model()
print(type(unet_model))
print(unet_model.summary())
unet_model.compile(loss=masked_SSIM(tp5_mask,len(list_targets)), metrics = tp5_rmse_loss, optimizer = opt)
print("Model compiled")
#
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = file_checkpoints, save_weights_only = True,
                                                monitor = 'val_loss', mode = 'min', verbose = 2,
                                                save_best_only = True)


model_history = unet_model.fit(train_generator, validation_data = valid_generator, 
                               epochs = compile_params["n_epochs"], verbose = 2, 
                               callbacks = [checkpoint])
print("Model fitted")


# In[8]:

import pickle
filename_model = f'UNet.h5'
unet_model.save_weights(os.path.join(root_output_dir,filename_model))
#
file_model_training_history = os.path.join(paths["outputs"],f"Training_historyV2.pkl")
pickle.dump(model_history.history, open(file_model_training_history, "wb"))
#
t1 = time.time()
dt = t1 - t0
print("Computing time: " + str(dt) + " seconds")
