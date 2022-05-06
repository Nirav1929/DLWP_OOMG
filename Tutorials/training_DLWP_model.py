#   --------------------------------------------------------------------------
#   Training a DLWP-CS model
#   --------------------------------------------------------------------------
#   Now we use the data processed in the previous two notebooks to train a 
#   convolutional neural network on weather data mapped to the cubed sphere. 
#   We will construct the same convolutional neural network with the cubed 
#   sphere as in Weyn et al. (2020), with the exception of having only two 
#   variables (Z500 and T2) instead of their four, and without the constant 
#   land-sea mask and topography data. This will seem like a fairly involved 
#   example but much simpler constructions are also possible using the 
#   DLWPNeuralNet class instead of the functional Keras API. I also highly 
#   recommend having this model train on a GPU with at least 4 GB of video 
#   memory.

#   Required packages
#   No new packages are needed here beyond the main DLWP-CS requirements in 
#   the README.

#   Parameters
#   Let's start with some basic user-selected parameters, beginning with the 
#   file paths, which you'll need to change.

import os
os.chdir(os.pardir)

#root_directory = '/ourdisk/hpc/ai2es/nirav/data'
root_directory = '/ourdisk/hpc/ai2es/ablowe/DLWP'
predictor_file = os.path.join(root_directory, 'tutorial_z500_t2m_CS.nc')    #ABL
model_file = os.path.join(root_directory, 'dlwp-cs_1yr')
log_directory = os.path.join(root_directory, 'logs', 'dlwp-cs_1yr')

print('ABL1')
#   --------------------------------------------------------------------------
#   Some parameters:
#   - cnn_model_name: name of the function which constructs the model
#   - base_filter_number: number of convolutional filters in the first 
#       convolutional layer
#   - min_epochs: minimum number of training epochs
#   - max_epochs: maximum number of training epochs
#   - patience: if the validation loss does not go down for this number of 
#       epochs, end training
#   - batch_size: for mini-batches for SGD training
#   - shuffle: if True, shuffles the training data samples

cnn_model_name = 'unet2'
base_filter_number = 32
min_epochs = 0
max_epochs = 20
#max_epochs = 3
patience = 2
batch_size = 32
shuffle = True

#   Variable selection. This follows the coordinates in the data file. 
#   Make sure to use lists to avoid losing dimensions. 
#   Set add_solar to True to include insolation variable.
io_selection = {'varlev': ['z/500', 't2m/0']}
add_solar = True

print('ABL2')
#   --------------------------------------------------------------------------
#   These parameters govern the time stepping in the model.

#   - io_time_steps: the number of input/output time steps directly 
#       ingested/predicted by the model
#   - integration_steps: the number of forward sequence steps on which to 
#       minimize the loss function of the model
#   - data_interval: the number of steps in the data file that constitute a 
#       "time step." Here we use 2, and the data contains data every 3 hours, 
#       so the effective time step is 6 h.
#   - loss_by_step: either None (equal weighting) or a list of weighting 
#       factors for the loss function at each integration step.
io_time_steps = 2
integration_steps = 2
data_interval = 2
loss_by_step = None

#   Specify the data used for validation and training. Here we use 2013-14 for 
#   training and 2015-16 for validation, leaving aside 2017-18 for testing.
import pandas as pd

#   TUTORIAL
train_set = list(pd.date_range('2013-01-01', '2014-12-31 21:00', freq='3H'))
validation_set = list(pd.date_range('2015-01-01', '2016-12-31 21:00', freq='3H'))
#train_set = list(pd.date_range('2013-01-01', '2013-03-02 21:00', freq='3H'))
#validation_set = list(pd.date_range('2015-01-01', '2015-03-02 21:00', freq='3H'))
##   FULL
#train_set = list(pd.date_range('1970-01-01', '2010-12-31 21:00', freq='3H'))
#validation_set = list(pd.date_range('2011-01-01', '2016-12-31 21:00', freq='3H'))

#   If you have a newer Nvidia GPU (Tesla V100, GeForce RTX series) with 
#   tensor cores, you can enable mixed precision for faster training.
use_mp_optimizer = False

print('ABL3')
#   --------------------------------------------------------------------------
#   Create a DLWP model
#   --------------------------------------------------------------------------
#   Since the data generators depend on the model (granted it's an outdated 
#   dependency), we make the model instance first.

from DLWP.model import DLWPFunctional
print('ABL3.5')

dlwp = DLWPFunctional(is_convolutional=True, time_dim=io_time_steps)

print('ABL4')
#   --------------------------------------------------------------------------
#   Load data and create data generators
#   --------------------------------------------------------------------------
#   DLWP-CS includes powerful data generators that produce batches of training 
#   data on-the-fly. This enables them to load only the time series into memory #   instead of repetitive samples of data. On the downside, it makes reading 
#   training data from disk virtually impossibly slow. First, load the data.
import xarray as xr
print('ABL4.5')

data = xr.open_dataset(predictor_file)
train_data = data.sel(sample=train_set)
validation_data = data.sel(sample=validation_set)

print('ABL5')
#   Create the training data generator. Here we use the ArrayDataGenerator 
#   class, which has a nifty pre-processing function to create a single numpy 
#   array of data. The SeriesDataGenerator is more intuitive and would work 
#   equally well. The only reason I don't use the latter is because I thought 
#   the overhead when using xarray objects instead of pure numpy might slow 
#   things down. It doesn't.
from DLWP.model.preprocessing import prepare_data_array
from DLWP.model import ArrayDataGenerator

print('Loading data to memory...')
train_array, input_ind, output_ind, sol = prepare_data_array(train_data, 
    input_sel=io_selection,output_sel=io_selection, add_insolation=add_solar)
generator = ArrayDataGenerator(dlwp,train_array,rank=3,input_slice=input_ind,
    output_slice=output_ind,input_time_steps=io_time_steps,
    output_time_steps=io_time_steps,sequence=integration_steps,
    interval=data_interval, insolation_array=sol,batch_size=batch_size,
    shuffle=shuffle, channels_last=True,drop_remainder=True)

#   Now do the same for the validation data.
print('Loading validation data to memory...')
val_array, input_ind, output_ind, sol = prepare_data_array(validation_data,
    input_sel=io_selection,output_sel=io_selection,add_insolation=add_solar)
val_generator = ArrayDataGenerator(dlwp,val_array,rank=3,input_slice=input_ind,
    output_slice=output_ind,input_time_steps=io_time_steps,
    output_time_steps=io_time_steps,sequence=integration_steps,
    interval=data_interval, insolation_array=sol,batch_size=batch_size,
    shuffle=False, channels_last=True)

#   Since TensorFlow 2.0, using multiprocessing in training Keras models has 
#   bad behavior (memory leaks). Therefore, for better performance, I recommend
#   using the tf_data_generator function to create a tensorflow.data.Dataset 
#   generator object. It does require knowing the names of the inputs and 
#   outputs to use this, however, so we hack the names here. This will become 
#   more evident in the next section.
from DLWP.model import tf_data_generator

input_names=['main_input']+['solar_%d' % i for i in range(1,integration_steps)]
tf_train_data=tf_data_generator(generator,batch_size=batch_size,
    input_names=input_names)
tf_val_data = tf_data_generator(val_generator,input_names=input_names)
print('ABL6')

#   --------------------------------------------------------------------------
#   Create the CNN model architecture
#   --------------------------------------------------------------------------
#   Now the fun part! 
#   Here we create all of the layers that will go into the model. A few notes:
#   - The generator produces a list of inputs when integration_steps is greater 
#       than 1:
#       - main input, including insolation
#       - insolation for step 2
#       - insolation for step 3...
#   - We use our custom layers for padding and convolutions on the cubed sphere
#   - We can use the Keras 3D layers for operations on the 3D spatial structure
#       of the cubed sphere
#   - There are more layers defined here than actually used in the model 
#       architecture. That's ok.

from tensorflow.keras.layers import Input, UpSampling3D, AveragePooling3D, concatenate, ReLU, Reshape, Concatenate, Permute
from DLWP.custom import CubeSpherePadding2D, CubeSphereConv2D

#   Shortcut variables. The generator provides the expected shape of the data
cs = generator.convolution_shape
cso = generator.output_convolution_shape
input_solar = (integration_steps > 1 and add_solar)

#   Define layers. Must be defined outside of model function so we use the 
#   same weights at each integration step.
main_input = Input(shape=cs, name='main_input')
if input_solar:
    solar_inputs = [Input(shape=generator.insolation_shape, name='solar_%d' % d) for d in range(1, integration_steps)]
cube_padding_1 = CubeSpherePadding2D(1, data_format='channels_last')
pooling_2 = AveragePooling3D((1, 2, 2), data_format='channels_last')
up_sampling_2 = UpSampling3D((1, 2, 2), data_format='channels_last')
relu = ReLU(negative_slope=0.1, max_value=10.)
conv_kwargs = {'dilation_rate': 1,'padding': 'valid','activation': 'linear',
    'data_format': 'channels_last'}
skip_connections = 'unet' in cnn_model_name.lower()
conv_2d_1 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
conv_2d_1_2 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
conv_2d_1_3 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
conv_2d_2 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
conv_2d_2_2 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
conv_2d_2_3 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
conv_2d_3 = CubeSphereConv2D(base_filter_number * 4, 3, **conv_kwargs)
conv_2d_3_2 = CubeSphereConv2D(base_filter_number * 4, 3, **conv_kwargs)
conv_2d_4 = CubeSphereConv2D(base_filter_number * 4 if skip_connections else base_filter_number * 8, 3, **conv_kwargs)
conv_2d_4_2 = CubeSphereConv2D(base_filter_number * 8, 3, **conv_kwargs)
conv_2d_5 = CubeSphereConv2D(base_filter_number * 2 if skip_connections else base_filter_number * 4, 3, **conv_kwargs)
conv_2d_5_2 = CubeSphereConv2D(base_filter_number * 4, 3, **conv_kwargs)
conv_2d_5_3 = CubeSphereConv2D(base_filter_number * 4, 3, **conv_kwargs)
conv_2d_6 = CubeSphereConv2D(base_filter_number if skip_connections else base_filter_number * 2, 3, **conv_kwargs)
conv_2d_6_2 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
conv_2d_6_3 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
conv_2d_7 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
conv_2d_7_2 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
conv_2d_7_3 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
conv_2d_8 = CubeSphereConv2D(cso[-1], 1, name='output', **conv_kwargs)

#   --------------------------------------------------------------------------
#   Now we actually create the output using the functional API. For each 
#   operation in the model, we call the appropriate layer on an input tensor x.
#   This function performs the operations inside a U-Net, including the skipped
#   connections with concatenation along the channels dimension. This is the 
#   sequence of operations to get input data to a prediction, but it is not the
#   whole model, since that one must predict a sequence of 2
#   (integration_steps = 2). That will be next.

def unet2(x):
    x0 = cube_padding_1(x)
    x0 = relu(conv_2d_1(x0))
    x0 = cube_padding_1(x0)
    x0 = relu(conv_2d_1_2(x0))
    x1 = pooling_2(x0)
    x1 = cube_padding_1(x1)
    x1 = relu(conv_2d_2(x1))
    x1 = cube_padding_1(x1)
    x1 = relu(conv_2d_2_2(x1))
    x2 = pooling_2(x1)
    x2 = cube_padding_1(x2)
    x2 = relu(conv_2d_5_2(x2))
    x2 = cube_padding_1(x2)
    x2 = relu(conv_2d_5(x2))
    x2 = up_sampling_2(x2)
    x = concatenate([x2, x1], axis=-1)
    x = cube_padding_1(x)
    x = relu(conv_2d_6_2(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_6(x))
    x = up_sampling_2(x)
    x = concatenate([x, x0], axis=-1)
    x = cube_padding_1(x)
    x = relu(conv_2d_7(x))
    x = cube_padding_1(x)
    x = relu(conv_2d_7_2(x))
    x = conv_2d_8(x)
    return x

#   --------------------------------------------------------------------------
#   Next we manipulate the result of the CNN back to inputs to the same CNN, 
#   add the new insolation input, and pass it through again. This allows us to
#   minimize the loss function at each step of the sequence. Adding the 
#   insolation looks complicated because the array includes a time dimension
#   whereas the data inputs are flattened time/variables.
from tensorflow.keras.layers import Reshape, Concatenate, Permute

def complete_model(x_in):
    outputs = []
    model_function = globals()[cnn_model_name]
    is_seq = isinstance(x_in, (list, tuple))
    xi = x_in[0] if is_seq else x_in
    outputs.append(model_function(xi))
    for step in range(1, integration_steps):
        xo = outputs[step - 1]
        if is_seq and input_solar:
            xo = Reshape(cs[:-1] + (io_time_steps, -1))(xo)
            xo = Concatenate(axis=-1)([xo, Permute((2, 3, 4, 1, 5))(x_in[step])])
            xo = Reshape(cs)(xo)
        outputs.append(model_function(xo))
    return outputs

#   --------------------------------------------------------------------------
#   Now we use our known inputs to get the outputs from complete_model and 
#   construct a Keras Model.
from tensorflow.keras.models import Model

if not input_solar:
    inputs = main_input
else:
    inputs = [main_input]
    if input_solar:
        inputs = inputs + solar_inputs
model = Model(inputs=inputs, outputs=complete_model(inputs))

#   --------------------------------------------------------------------------
#   Define a loss function and an optimizer. 
#   We use the default mean-squared-error and the Adam optimizer.
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

loss_function = 'mse'

# Get an optimizer, with mixed precision if requested
opt = Adam()
if use_mp_optimizer:
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

#   --------------------------------------------------------------------------
#   Finally, we are ready to compile our DLWP model.

dlwp.build_model(model,loss=loss_function,loss_weights=loss_by_step,optimizer=opt, metrics=['mae'])
print(dlwp.base_model.summary())

#   --------------------------------------------------------------------------
#   Train the DLWP model
#   --------------------------------------------------------------------------
#   Now let's train the model. First, define a few callbacks:
#   - history: save the model metrics as it trains
#   - early: a custom callback used to stop training after patience epochs, 
#       but only after a minimum number of epochs
#   - tensorboard: TensorFlow's complete logging
#   - GeneratorEpochEnd: when using tf_data_generator, this is used to shuffle
#       the samples
from tensorflow.keras.callbacks import History, TensorBoard
from DLWP.custom import EarlyStoppingMin, SaveWeightsOnEpoch, GeneratorEpochEnd

history = History()        #ABL
###callback = TensorBoard(log_dir=log_directory)
early = EarlyStoppingMin(
    monitor='val_loss' if validation_data is not None else 'loss', 
    min_delta=0.,min_epochs=min_epochs,max_epochs=max_epochs,patience=patience,
    restore_best_weights=True, verbose=1)
tensorboard = TensorBoard(log_dir=log_directory, update_freq='epoch')

#   Fit the model! Don't worry about warnings about the data sequences.
import time

start_time = time.time()
dlwp.fit_generator(tf_train_data,epochs=max_epochs+1,verbose=1,
    validation_data=tf_val_data,
    callbacks=[history,early,GeneratorEpochEnd(generator)])
    #callbacks=[tensorboard,early,GeneratorEpochEnd(generator)])     #ABL
end_time = time.time()

#   --------------------------------------------------------------------------
#   Save the model. We use the DLWP utility function, which saves the DLWP 
#   instance as well.
from DLWP.util import save_model

#save_model(dlwp, model_file, history=history)
save_model(dlwp, model_file)        #ABL
print('Wrote model %s' % model_file)

#   --------------------------------------------------------------------------
#   Finally, print some loss metrics.

print("\nTrain time -- %s seconds --" % (end_time - start_time))

score = dlwp.evaluate(*val_generator.generate([]), verbose=0)
print('Validation loss:', score[0])
print('Other scores:', score[1:])

#   --------------------------------------------------------------------------





