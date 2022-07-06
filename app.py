from math import pi
import streamlit as st
import streamlit.components.v1 as components
from scipy.spatial import cKDTree
from bokeh.plotting import figure
from bokeh.models import HoverTool, DatetimeTickFormatter, ColumnDataSource
from bokeh.palettes import Category10
import matplotlib.pyplot as plt
import folium
import folium.plugins as plugins
import numpy as np
import pandas as pd
import retrieve_data as rd

# -- app constants
APP_TITLE = 'Wind Resource Temporal Variability Analysis'
HEIGHTS = [10, 40, 60, 80, 100, 120, 140, 160, 200] # in meter
MIN_DATE = '2012-01-01'
MAX_DATE = '2012-12-31'
START_TIME = '00:00:00'
END_TIME = '23:00:00'

# -- page config
st.set_page_config(page_title=APP_TITLE, page_icon="🌍", layout='wide', initial_sidebar_state='expanded')
st.markdown(
    f"""
    # {APP_TITLE}
    Input data parameters in the sidebar
    """
)


# -- Read and cache metadata
@st.cache
def read_meta():
    """Read local meta data file

    Returns:
        metadaframe: returns local meta data
    """

    meta = pd.read_csv('meta.csv')
    return meta


# -- Get state county list from meta data and save it to a file
try:
    meta = read_meta()
except:
    with st.spinner('Loading resources. Please wait...'):
        meta = rd.get_metadata(rd.get_data())
        meta.to_csv('meta.csv', index=False)
meta = meta.replace('None', np.nan).dropna(axis=0, how='any').reset_index(drop=True)
STATE_COUNTY_DICT = meta.groupby(['state'])['county'].apply(lambda grp: list(grp.value_counts().index)).to_dict()


# -- plots
def create_plots(params):
    
    lat = params['lat']
    lon = params['lon']
    state = params['state']
    county = params['county']
    date_from = params['date_from']
    date_to = params['date_to']
    wspd_bool_dict = params['wspd']
    spatial_analysis_type = params['spatial_analysis_type']
    
    coords_county, coords_state, wspd_df = run_analysis(params)
    
    # -- analysis type: point
    if spatial_analysis_type == 'Point':
        st.markdown('#### Summary statistics:')
        col1_1, col1_2, col1_3 = st.columns([1.5, 1, 1.5])  
        with col1_1:
            st.markdown("**Maximum windspeed**")
            st.write(pd.DataFrame({'windspeed': wspd_df.max(), 'date/time': wspd_df.idxmax()}))
            
            st.markdown("**Minimum windspeed**")
            st.write(pd.DataFrame({'windspeed': wspd_df.min(), 'date/time': wspd_df.idxmin()}))

            st.markdown("**Monthly mean windspeed**")
            columns = wspd_df.resample('M').mean().index.month
            monthly_avg = pd.DataFrame(wspd_df.resample('M').mean().T)
            monthly_avg.columns = columns
            st.write(monthly_avg)
            
        with col1_2:
            st.markdown("**Mean windspeed**")
            st.write(pd.DataFrame({'windspeed': wspd_df.mean()}))

        with col1_3:
            st.markdown(f"""
            **Location:** {coords_county} county, {coords_state}
            """)
            st.map(pd.DataFrame([{'latitude': params['lat'],'longitude': params['lon']}]))
        st.write("##") # add some space b/w charts
        
        st.markdown('#### Daily time series (time vs. windspeed (m/s))')
        st.markdown('**Daily maximum**')
        p = generate_line_plot( wspd_df.resample('D').max())
        st.bokeh_chart(p, use_container_width=True)
        st.write("##") # add some space b/w charts

        st.markdown('**Daily mean**')
        p = generate_line_plot(wspd_df.resample('D').mean())
        st.bokeh_chart(p, use_container_width=True)
        st.write("##") # add some space b/w charts

        st.markdown('**Daily minimum**')
        p = generate_line_plot( wspd_df.resample('D').min())
        st.bokeh_chart(p, use_container_width=True)
        st.write("##") # add some space b/w charts

        wspd_df["day"] = wspd_df.index.day
        wspd_df["hour"] = wspd_df.index.hour
        wspd_df["month"] = wspd_df.index.month

        st.markdown(
            """
            #### 30 day or 12 month x 24 hour mean windspeeds (m/s)
            Choosing 1 month _date range_ and day _x-axis_ shows gusty days
            """)
        grid_plot_xaxis = st.radio("Choose x-axis:", ('day', 'month'), index=0, horizontal=True)
        agg = wspd_df.groupby([grid_plot_xaxis,"hour"]).mean()
        df_cols = [col_name for col_name in wspd_df.columns if 'windspeed' in col_name]
        num_cols = 3
        num_rows = len(df_cols)//num_cols+1
        df_col_idx = 0
        for _ in range(num_rows):
            cols = st.columns(num_cols)
            for col in cols:
                if df_col_idx < len(df_cols):
                    agg_ = agg.reset_index().pivot(index=grid_plot_xaxis, columns="hour", values=df_cols[df_col_idx])
                    fig, ax = plt.subplots()
                    ax.set_xlabel('hour')
                    ax.set_ylabel(grid_plot_xaxis)
                    ax.set_title(df_cols[df_col_idx])
                    im = ax.imshow(agg_)
                    plt.colorbar(im, ax=ax)
                    col.pyplot(fig)
                    df_col_idx = df_col_idx+1
    
    # -- analysis type: area
    elif spatial_analysis_type == 'Area':
        st.markdown(f"""
            **Location:** {coords_county} county, {coords_state}
            """)
        data = []
        d =  wspd_df.to_dict('records')
        for dictionary in d:
            data.append([[float(k.split('_')[0]), float(k.split('_')[1]), v] for k, v in dictionary.items()])

        m = folium.Map([data[0][0][0], data[0][0][1]], tiles="stamentoner", zoom_start=7)

        hm = plugins.HeatMapWithTime(data=data, radius=15, auto_play=True, position='bottomright')

        hm.add_to(m)

        hm.save(f'./heatmap.html')


def generate_line_plot(wspd_df):
    p = figure(y_range=(0, 22), x_axis_type='datetime')
    colors = iter(Category10[list(Category10.keys())[-1]])
    circle_colors = []
    for col_name, values in wspd_df.items():
        color = next(colors)
        circle_colors.extend([color] * len(values))
        p.line(x=wspd_df.index, y=values, legend_label=col_name, line_width=1.5, color=color)
    r = p.circle(x=wspd_df.index.tolist() * wspd_df.columns.size, y=wspd_df.T.values.flatten(), size=5, color=circle_colors, alpha=0.5)
    p.add_tools(HoverTool(
        tooltips=[
            ("Date", "$x{%F}"),
            ("Windspeed", "$y{1.11} m/s")
        ],
        formatters={
            '$x':'datetime'
        }
        ))
    p.xaxis.formatter=DatetimeTickFormatter(
        hours=["%d %B %Y"],
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"],
    )
    p.xaxis.major_label_orientation = pi/4 # pi/4
    p.hover.renderers = [r]

    return p

# -- Get the nearest lat lon in the dataset
def nearest_site(tree, lat, lon):
    """Get nearest lat lon in the dataset from the user input lat lon.

    Args:
        tree (kdtree object): KDTree object of the lat lon from the dataset
        lat (float): latitude
        lon (float): longitude

    Returns:
        index: position of the nearest lat lon in the dataset
    """

    lat_lon = np.array([lat, lon])
    _, pos = tree.query(lat_lon)
    return pos


# -- run_analysis
def run_analysis(params):

    """Runs analysis on the params passed from the app UI

    Args:
        params (dict): data query param dictionary
    """
    
    meta = read_meta()
        
    lat = params['lat']
    lon = params['lon']
    state = params['state']
    county = params['county']
    date_from = params['date_from']
    date_to = params['date_to']
    wspd_bool_dict = params['wspd']
    spatial_analysis_type = params['spatial_analysis_type']
    
    hdf_file = rd.get_data()
    time_index = pd.to_datetime(hdf_file['time_index'][...].astype(str))   # TODO: save locally

    # -- analysis type: point
    if spatial_analysis_type == 'Point':
        wspd = {}
        dset_coords = np.array(meta[['latitude', 'longitude']])
        tree = cKDTree(dset_coords)
        nearest_idx = nearest_site(tree, lat, lon)
        coords_meta = meta[(meta['latitude'] == dset_coords[nearest_idx][0]) & (meta['longitude'] == dset_coords[nearest_idx][1])]
        state = coords_meta['state'].values[0]
        county = coords_meta['county'].values[0]
        coords_index = coords_meta.index.values[0]
        date_from_index = time_index.get_loc(f'{date_from} {START_TIME}')
        date_to_index = time_index.get_loc(f'{date_to} {END_TIME}') 
        for height in HEIGHTS:
            if wspd_bool_dict[height]:
                dset = hdf_file[f'windspeed_{height}m']
                wspd[f'windspeed_{height}m'] = dset[date_from_index:date_to_index, coords_index] / dset.attrs['scale_factor']
        wspd_df = pd.DataFrame(wspd, index=time_index[date_from_index:date_to_index])
        return county, state, wspd_df

    # -- analysis type: area
    elif spatial_analysis_type == 'Area':
        dx=12
        
        meta_indices = meta[(meta['state'] == state) & (meta['county'] == county)]
        dset = hdf_file[f'windspeed_{100}m'] 
        wspd = dset[:, min(meta_indices.index):max(meta_indices.index):dx] / dset.attrs['scale_factor']
        col_list = (meta_indices['latitude'].astype(str) + '_' + meta_indices['longitude'].astype(str)).values
        columns = (meta['latitude'][min(meta_indices.index):max(meta_indices.index)+1:dx].astype(str) + '_' + meta['longitude'][min(meta_indices.index):max(meta_indices.index)+1:dx].astype(str)).values
        wspd_df = pd.DataFrame(wspd, index=time_index, columns=columns)
        wspd_df.drop(columns=[col for col in wspd_df if col not in col_list], inplace=True)
        wspd_df = wspd_df.resample('D').mean()
        wspd_df_normalized = ((wspd_df - wspd_df.min().min()) / (wspd_df.max().max() - wspd_df.min().min()))
        state = params['state']
        county = params['county']
        return county, state, wspd_df_normalized


# -- sidebar parameters
with st.sidebar:
    params = {
        'lat': 39.7392,
        'lon': -104.9903,
        'state': 'Colorado',
        'county': 'Montezuma',
        'date_from': '2012-01-01 00:00:00',
        'date_to': '2012-12-31 00:00:00',
        'wspd': {},
        'spatial_analysis_type': 'Point'
   }
    st.subheader('Spatial feature')
    
    spatial_analysis_type = st.radio('Select type',['Point', 'Area'], 0)
    if spatial_analysis_type == 'Point':
        params['lat'] = float(st.text_input('Latitude', '39.7392'))
        params['lon'] = float(st.text_input('Longitude', '-104.9903'))
        params['state'] = ''
        params['county'] = ''
    elif spatial_analysis_type == 'Area':
        params['lat'] = np.nan
        params['lon'] = np.nan  
        params['state'] = st.selectbox('State', sorted([state for state in STATE_COUNTY_DICT.keys()]))
        params['county'] = st.selectbox('County', sorted([county for county in STATE_COUNTY_DICT[params['state']]]))

    st.subheader("Date range")
    params['date_from'] = str(st.date_input('From', pd.to_datetime(MIN_DATE), min_value=pd.to_datetime(MIN_DATE), max_value=(pd.to_datetime(params['date_to'])-pd.to_timedelta('30 days'))))
    params['date_to'] = str(st.date_input('to', pd.to_datetime(MAX_DATE), min_value=(pd.to_datetime(params['date_from'])+pd.to_timedelta('30 days')), max_value=pd.to_datetime(MAX_DATE)))

    height_checkbox = {}
    if spatial_analysis_type == 'Point':
        st.subheader('Height(s)')
        for height in HEIGHTS:
            if height == 100:
                height_checkbox[height] = (st.checkbox(f'{height} m', True))
            else:
                height_checkbox[height] = (st.checkbox(f'{height} m'))
        params['wspd'] = height_checkbox
    else:
        st.subheader('Height')
        height_option = st.selectbox('', [f'{height} m' for height in HEIGHTS], 4)
        params['wspd'] = {k:False for k in HEIGHTS}
        key = int(height_option.split()[0])
        params['wspd'][key] = True
    st.markdown("##")

if spatial_analysis_type == 'Point':
    params['spatial_analysis_type'] = spatial_analysis_type
    create_plots(params)

elif spatial_analysis_type == 'Area':
    params['spatial_analysis_type'] = spatial_analysis_type
    create_plots(params)
    with open('heatmap.html') as f:
        components.html(f.read(), height=500)