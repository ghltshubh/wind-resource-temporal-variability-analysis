# wind-resource-temporal-variability-analysis

### Problem statement: 

- Wind resource varies highly across space and at different temporal scales, e.g., sub-hourly, diurnal, monthly, and annual. (https://github.com/NREL/hsds-examples/blob/master/datasets/WINDToolkit.md)
- Understanding this variability is crucial for many wind energy modeling applications. 
- Develop a prototype analysis tool that examines wind speed variability at diurnal and monthly temporal scales at an individual location of your choosing.


### Dataset:

* HDF5 file (wtk_conus_2012.h5)
    * ['coordinates',
    * 'meta',
    * 'windspeed_100m',
    * 'windspeed_10m',
    * 'windspeed_120m',
    * 'windspeed_140m',
    * 'windspeed_160m',
    * 'windspeed_200m',
    * 'windspeed_40m',
    * 'windspeed_60m',
    * 'windspeed_80m'
    * …] 

### Exploratory data analysis:

- File read benchmark
- Array size: 8784, 2488136
- windspeed file indexing:

  `hdf_file = h5pyd.File(file_url, 'r')`
  
  `wspd = hdf_file["windspeed_100m"]`
  
  `wspd[time_index1:time_index2:step,`
  
  `loc_index1:loc_index2:step]`
  
  `wspd[ : , loc_index1:loc_index2:step]`

- Data points laid out in a 2D grid
- Subsample using step: 16 or 32 steps
  
  `wspd[ : , loc_index1:loc_index2:16]`

![Screenshot 2023-10-31 at 8 09 23 PM](https://github.com/ghltshubh/wind-resource-temporal-variability-analysis/assets/16928813/692b9b44-34eb-4906-ac41-27cbe5e88832)


![Screenshot 2023-10-31 at 8 01 55 PM](https://github.com/ghltshubh/wind-resource-temporal-variability-analysis/assets/16928813/b7a5f092-9f35-4aa7-bcc0-e15926ad56dd)


### To run the app 
1. Clone the repo
2. Install requirements.txt packages
3. Run `streamlit run app.py`


### Some insights: Denver county, Colorado

![Screenshot 2023-10-31 at 8 10 49 PM](https://github.com/ghltshubh/wind-resource-temporal-variability-analysis/assets/16928813/b1c817f0-2c74-4de5-90d1-0d82789def50)


- Yearly average wind speed increases with altitude (height)
- Occasionally wind speed at lower altitudes could be more than higher altitudes
- High variability in maximum wind speeds at all heights
- Variability in maximum wind speed increases with height
- Wind gusts usually occur during late evening or early morning.

TODO: Sample fewer data points for larger counties (San Bernardino County CA, Coconino County, AZ, Nye County, NV, etc.) to improve load times. 
