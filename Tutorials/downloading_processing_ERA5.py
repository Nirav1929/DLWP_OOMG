#!/usr/bin/env python
# coding: utf-8

# ## Downloading and processing ERA5 data
# 
# In this tutorial, we will use the DLWP data module to fetch and pre-process data from ERA5 to use in a DLWP weather prediction model. For the sake of simplicity, we use only a select few variables over a few years.
# 
# #### Python packages required here not in the base requirements
# 
# Let's start by installing the `cdsapi` package, which is required for retrieval of data. (See the README for packages already required for DLWP that need to also be installed.) Note that to use `cdsapi` you will need to register for an API key at CDS, following [their instructions](https://cds.climate.copernicus.eu/api-how-to).

# In[ ]:
import sys
#sys.path.append('/home/nirav1929/OWP/DLWP-CS_retry/DLWP-CS')
sys.path.append('/home/ablowe/DLWP-CS')



# ### Retrieve data
# 
# Define the variables and levels we want to retrieve. Single-level variables ignore the "levels" parameter. Also note that not all variables in the ERA5 dataset are coded with their parameter names as of now. We also take a reduced sample of years in the dataset.

# In[ ]:


variables = ['geopotential', '2m_temperature']
levels = [500]
years = list(range(1979, 2018))


# Initialize the data retriever. You'll want to change the directory to where you want to save the files.

# In[ ]:


import os
os.chdir(os.pardir)
from DLWP.data import ERA5Reanalysis

#data_directory = '/ourdisk/hpc/ai2es/nirav/data'
data_directory = '/ourdisk/hpc/ai2es/ablowe/DLWP'
os.makedirs(data_directory, exist_ok=True)
era = ERA5Reanalysis(root_directory=data_directory, file_id='tutorial')
era.set_variables(variables)
era.set_levels(levels)


# Download data! Automatically uses multi-processing to retrieve multiple files at a time. Note the parameter `hourly` says we're retrieving only every 3rd hour in the data, which is available hourly. The optional parameter passed to the retrieval package specifies that we want data interpolated to a 2-by-2 latitude-longitude grid.

# In[ ]:


era.retrieve(variables,levels,years=years,hourly=3,
    request_kwargs={'grid':[2., 2.]},verbose=True,delete_temporary=True)


# Check that we got what we wanted after the retrieval is done:

# In[ ]:


era.open()
print(era.Dataset)


# ### Process data for ingestion into DLWP
# 
# Now we use the DLWP.model.Preprocessor tool to generate a new data file ready for use in a DLWP Keras model. Some preliminaries... Note that we assign level "0" to the single-level 2m temperature data. I highly recommend using "pairwise" data processing, which means that each variable is matched to a level pair-wise. The length of the variables and levels lists should be the same. Also note that you only need to specify whole days in the dates. It takes care of the hourly data automatically.

# In[ ]:


import pandas as pd
from DLWP.data.era5 import get_short_name

dates = list(pd.date_range('1979-01-01', '2018-12-31', freq='D').to_pydatetime())
variables = get_short_name(variables)
levels = [500, 0]
#processed_file = '%s/tutorial_z500_t2m.nc' % data_directory
processed_file = '%s/tutorial_z500_t2m_1979_2018.nc' % data_directory


# Process data! For proper use of data in a neural network, variables must be normalized relative to each other. This is typically done simply by removing mean and dividing by standard deviation (`scale_variables` option). To save on memory use, we normally calculate the global mean and std of the data in batches. Since this is a small dataset, we can use a large batch size to make it go faster.

# In[ ]:


from DLWP.model import Preprocessor

pp = Preprocessor(era, predictor_file=processed_file)
pp.data_to_series(batch_samples=10000, variables=variables, levels=levels, pairwise=True,
                  scale_variables=True, overwrite=True, verbose=True)


# Show our dataset, then clean up. We also save to a version with no string coordinates (might be needed for tempest-remap in the next tutorial).

# In[ ]:


print(pp.data)
pp.data.drop('varlev').to_netcdf(processed_file + '.nocoord')
era.close()
pp.close()

