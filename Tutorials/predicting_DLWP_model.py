#   --------------------------------------------------------------------------
#   Predicting with a DLWP-CS model
#   --------------------------------------------------------------------------
#   Finally we will explore using the advanced functionality in DLWP-CS to make
#   a time-series global weather prediction with our trained DLWP-CS model. 
#   We will save the prediction to a netCDF file and apply inverse scaling to 
#   get physical variables back. Again, I recommend having this model run on a 
#   GPU with at least 4 GB of video memory.

#   Required packages
#   No new packages are needed here beyond the main DLWP-CS requirements in 
#   the README.

#   Parameters
#   Some user-specified parameters. The scale_file contains the mean and 
#   standard of the data (which was dropped in the cubed sphere remapping). 
#   The map_files were produced by the cubed sphere remapping. We can re-use 
#   them here so we don't have to generate them again.

import os
os.chdir(os.pardir)

#root_directory = '/ourdisk/hpc/ai2es/nirav/data'
root_directory = '/ourdisk/hpc/ai2es/ablowe/DLWP'
predictor_file = os.path.join(root_directory,'tutorial_z500_t2m_CS.nc')
scale_file = os.path.join(root_directory,'tutorial_z500_t2m.nc')

model = os.path.join(root_directory, 'dlwp-cs_tutorial')
map_files = ('map_LL91x180_CS48.nc', 'map_CS48_LL91x180.nc')


'''
#out_directory = './'
#predictor_file = os.path.join(root_directory, 'ERA5', 'tutorial_z500_t2m_CS.nc')
predictor_file = os.path.join(root_directory, 'tutorial_z500_t2m_CS.nc')    #ABL
model_file = os.path.join(out_directory, 'dlwp-cs_tutorial1')
#log_directory = os.path.join(out_directory, 'logs', 'dlwp-cs_tutorial')
log_directory = os.path.join(out_directory, 'logs1', 'dlwp-cs_tutorial')

'''

#   We'll resurrect some parameters from the training tutorial. See that 
#   notebook for definitions. Note that we omit data_interval because we simply
#   select only every 6 hours from the dataset.

io_selection = {'varlev': ['z/500', 't2m/0']}
add_solar = True
io_time_steps = 2

#   --------------------------------------------------------------------------
#   Specify the data used for prediction. We'll make weekly forecasts in the 
#   test set, initialized at 0 UTC. We need to specify a subset of the data
#   that contains all these initializations.

import numpy as np
import pandas as pd
import xarray as xr

#validation_set = pd.date_range('2016-12-31', '2018-12-31', freq='6H')
validation_set = pd.date_range('2016-12-31', '2017-02-01', freq='6H')   #ABL
validation_set = np.array(validation_set, dtype='datetime64[ns]')

#   Set the initialization dates, the numer of foreward forecast hours, 
#   and the time step (could be automated...)

#dates = pd.date_range('2017-01-01', '2018-12-31', freq='7D')
dates = pd.date_range('2017-01-01', '2017-02-01', freq='7D')
initialization_dates = xr.DataArray(dates)
num_forecast_hours = 5 * 24
dt = 6

#   --------------------------------------------------------------------------
#   Load the DLWP model
#   --------------------------------------------------------------------------
from DLWP.util import load_model, remove_chars, is_channels_last

dlwp = load_model(model)

# File to save the forecast
forecast_file = os.path.join(root_directory, 'forecast_%s.nc' % remove_chars(model.split(os.sep)[-1]))

#   --------------------------------------------------------------------------
#   Open the data and create the data generator
#   --------------------------------------------------------------------------
all_ds = xr.open_dataset(predictor_file)
predictor_ds = all_ds.sel(sample=validation_set)

from DLWP.model import SeriesDataGenerator

sequence=dlwp._n_steps if hasattr(dlwp,'_n_steps') and dlwp._n_steps>1 else None

val_generator=SeriesDataGenerator(dlwp,predictor_ds,rank=3,
    add_insolation=add_solar,input_sel=io_selection,output_sel=io_selection,
    input_time_steps=io_time_steps,output_time_steps=io_time_steps,
    shuffle=False,sequence=sequence,batch_size=32,load=False,
    channels_last=is_channels_last(dlwp))

#   --------------------------------------------------------------------------
#   Create the estimator and make a prediction
#   --------------------------------------------------------------------------
#   We use the handy TimeSeriesEstimator class to intelligently produce a time
#   series forecast. This class depends on the model and the data generator.

from DLWP.model import TimeSeriesEstimator

estimator = TimeSeriesEstimator(dlwp, val_generator)

print('Predicting with model %s...' % model)

# Select the samples from the initialization dates. The first "time" input to the model is actually one time step earlier
samples = np.array([int(np.where(val_generator.ds['sample'] == s)[0]) for s in initialization_dates]) - io_time_steps + 1

time_series = estimator.predict(num_forecast_hours // dt, samples=samples, verbose=1)

# Transpose if channels_last was used for the model
if is_channels_last(dlwp):
    time_series = time_series.transpose('f_hour', 'time', 'varlev', 'x0', 'x1', 'x2')

#   Scale the variables back to real data.
if scale_file is None:
    scale_ds = predictor_ds
else:
    scale_ds = xr.open_dataset(scale_file)
sel_mean = scale_ds['mean'].sel(io_selection)
sel_std = scale_ds['std'].sel(io_selection)
time_series = time_series * sel_std + sel_mean

#   For some reason the time series output, when saved to netCDF, is not 
#   compatible with TempestRemap. I have yet to figure out why. But there is a
#   function in the DLWP verify module that re-formats a time series and
#   produces output that TempestRemap is happy with...

from DLWP.verify import add_metadata_to_forecast_cs

fh = np.arange(dt, time_series.shape[0] * dt + 1., dt)
time_series = add_metadata_to_forecast_cs(time_series.values,fh,
    predictor_ds.sel(**io_selection).sel(sample=initialization_dates))

#   Save the prediction in cubed sphere format. Drop the string "varlev" 
#   coordinate for TempestRemap.

time_series.drop('varlev').to_netcdf(forecast_file + '.cs')

#   --------------------------------------------------------------------------
#   Remap the forecast to a latitude-longitude grid
#   --------------------------------------------------------------------------
from DLWP.remap import CubeSphereRemap

csr = CubeSphereRemap(to_netcdf4=True)
csr.assign_maps(*map_files)
csr.convert_from_faces(forecast_file + '.cs', forecast_file + '.tmp')
csr.inverse_remap(forecast_file + '.tmp', forecast_file, '--var', 'forecast')
os.remove(forecast_file + '.tmp')
#   That was fast, right?


