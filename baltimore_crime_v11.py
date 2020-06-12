#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing all necessary libs
import numpy as np
import pandas as pd 
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import cm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KernelDensity

from scipy.interpolate import griddata
import scipy as sp
import scipy.ndimage

#pip install geojsoncontour
import geojsoncontour
import branca

import folium
from folium import plugins
from folium.plugins import HeatMap
from folium import FeatureGroup, LayerControl, Map, Marker


#Follow the steps in below link for geopandas
#https://geoffboeing.com/2014/09/using-geopandas-windows/
import geopandas as gpd
import json

#pip install geopy
from geopy.geocoders import Nominatim
from pandas import DataFrame

from sklearn.neighbors import BallTree

import datetime


# In[ ]:


#Method to remove NaN rows in dataframe
def Remove_NaN_InData(data,col_name):
    data[col_name].replace('', np.nan, inplace=True)
    data.dropna(subset=[col_name], inplace=True)
    data=data.reset_index(drop=True)


# In[ ]:


'''Baltimore Crime Data: For plotting crimes'''


# In[ ]:


#Importing The Baltimore Crime Data
df_Crime = pd.read_csv('BPD_Part_1_Victim_Based_Crime_Data.csv',low_memory=False)
print("Baltimore Crime data is imported")


# In[ ]:


#Baltimore Crime Data: Preprocessing
def Filter_Crime_On_Month(df_Crime,sample_month):
    #Formating CrimeDate to specific format
    df_Crime['CrimeDate'] =  pd.to_datetime(df_Crime['CrimeDate'], format='%m/%d/%Y')
    
    #Adding Year and month columns to filter for easy stimulation
    df_Crime['year'] = pd.DatetimeIndex(df_Crime['CrimeDate']).year
    df_Crime['month'] = pd.DatetimeIndex(df_Crime['CrimeDate']).month
    
    #Filtering only data for 2019
    df_Crime=df_Crime.loc[df_Crime['year']== 2019]
    df_Crime=df_Crime.loc[df_Crime['month']== sample_month]
    
    #Remove NaN in Data
    print("Length of file before Removing Empty Rows: ",len(df_Crime))
    Remove_NaN_InData(df_Crime,'Latitude')
    Remove_NaN_InData(df_Crime,'Longitude')
    Remove_NaN_InData(df_Crime,'Neighborhood')
    print("Length of file After Removing Empty Rows: ",len(df_Crime))
    
    #Checking for Outliers and removing them
    print("Checking for Outliers and removing them")
    plt.figure()
    plt.boxplot(df_Crime['Longitude'], 0, 'gD')
    plt.show()

    plt.figure()
    plt.boxplot(df_Crime['Latitude'], 0, 'gD')
    plt.show()

    Q1 = df_Crime['Longitude'].quantile(0.25)
    Q3 = df_Crime['Longitude'].quantile(0.75)
    IQR = Q3 - Q1

    df_Crime = df_Crime [(df_Crime['Longitude'] >= Q1 - 1.5 * IQR) & (df_Crime['Longitude'] <= Q3 + 1.5 *IQR)]

    Q1 = df_Crime['Latitude'].quantile(0.25)
    Q3 = df_Crime['Latitude'].quantile(0.75)
    IQR = Q3 - Q1

    df_Crime = df_Crime [(df_Crime['Latitude'] >= Q1 - 1.5 * IQR) & (df_Crime['Latitude'] <= Q3 + 1.5 *IQR)]

    print("Length of file After Removing Outliers: ",len(df_Crime))
    #print("Displaying the data after preprocessing")
    #print(df_Crime)
    
    return df_Crime


# In[ ]:


'''Baltimore Police Call for service Data: For plotting policing data'''


# In[ ]:


#Importing The Police Call For Service Data
#Over here it has 7M records for time consumption we are considering only nrows=2L
df_police_COS = pd.read_csv('PoliceLocation_With_Location.csv',low_memory=False)
print("Baltimore Police Call for Service data is imported")

#Baltimore Police Data: Preprocessing
def Filter_PoliceData_On_Month(df_police_COS):    
    #Remove NaN in Data
    print("Length of file before Removing Empty Rows: ",len(df_police_COS))
    Remove_NaN_InData(df_police_COS,'ZipCode')
    Remove_NaN_InData(df_police_COS,'Community_Statistical_Areas')
    Remove_NaN_InData(df_police_COS,'Location')
    print("Length of file After Removing Empty Rows: ",len(df_police_COS))
 
    return df_police_COS


# In[ ]:


'''Baltimore Census Data: For plotting neighbhorhoods, population and race'''


# In[ ]:


#Importing The Neighbhorhood Census Data
df_neighood = pd.read_csv('2010_Census_Profile_by_Neighborhood_Statistical_Areas.csv',low_memory=False)
print("Baltimore Census Data is imported")


# In[ ]:


#Considereing Only The following Columns
df_neighood=df_neighood[['the_geom', 'Name', 'Population', 'White', 'Blk_AfAm', 'AmInd_AkNa',
       'Asian', 'NatHaw_Pac', 'Other_Race', 'TwoOrMore', 'Hisp_Lat', 'Male',
       'Female']]

#Checking for empty rows and removing them
Remove_NaN_InData(df_neighood,'the_geom')
Remove_NaN_InData(df_neighood,'Name')
print("Length of file After Removing Empty Rows: ",len(df_neighood))


# In[ ]:


'''Baltimore JSON file: For getting the multipolygon coordinates to plot neighbhorhood'''


# In[ ]:


#Importing the JSON file and converting to GEOJSON to GeoPandas dataframe
df_geojson = gpd.read_file('baltimore.json', driver='GeoJSON')
df_geojson=df_geojson[['name','geometry']]
print("Baltimore JSON file is imported")


# In[ ]:


#Adding Columns to df_geojson
df_geojson['Male']=df_neighood[['Male']]
df_geojson['Female']=df_neighood[['Female']]
df_geojson['White']=df_neighood[['White']]
df_geojson['Blk_AfAm']=df_neighood[['Blk_AfAm']]
df_geojson['AmInd_AkNa']=df_neighood[['AmInd_AkNa']]
df_geojson['Asian']=df_neighood[['Asian']]
df_geojson['NatHaw_Pac']=df_neighood[['NatHaw_Pac']]
df_geojson['Other_Race']=df_neighood[['Other_Race']]
df_geojson['TwoOrMore']=df_neighood[['TwoOrMore']]
df_geojson['Hisp_Lat']=df_neighood[['Hisp_Lat']]
df_geojson['Crime_Count']=""
df_geojson['Police_Count']=""
df_geojson['Detected_Crime_Count']=""

#Since GeoJson and census data might have certain empty census value replace 0's with Not Available
df_geojson.replace(0.0,'NA', inplace=True) 

#Assigning Baltimore Latitude and Longitude values
balti_location=[39.2904,-76.6122]
feature_group = FeatureGroup(name='Do_Not_Untick')


# In[ ]:


def draw_KDE_Map(i_data,i_geo_map,g1,cl):
    geomap=i_geo_map
    #Using Kernel Density Estimation
    X = i_data[['Longitude', 'Latitude']].values
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01)
    kde.fit(X)
    print("Generating KDE score")
    log_dens = kde.score_samples(X)
    
    #Genrating KDE Contours
    # Setup
    debug     = False
    # Setup colormap
    colors = cl
    vmin   = log_dens.min()
    vmax   =log_dens.max()
    levels = len(colors)
    cm     = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)
    #The original data
    x_orig = np.asarray(i_data['Longitude'].values.tolist())
    y_orig = np.asarray(i_data['Latitude'].values.tolist())
    z_orig = np.asarray(log_dens.tolist())
    # Make a grid
    x_arr          = np.linspace(np.min(x_orig), np.max(x_orig), 500)
    y_arr          = np.linspace(np.min(y_orig), np.max(y_orig), 500)
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
    # Grid the values
    z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')
    # Gaussian filter the grid to make it smoother
    sigma = [7,7]
    z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')
    # Create the contour
    contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels, alpha=0.5, colors=colors, linestyles='None', vmin=vmin, vmax=vmax)
 
    # Convert matplotlib contourf to geojson
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=contourf,
        min_angle_deg=3.0,
        ndigits=5,
        stroke_width=1,
        fill_opacity=1)

    # Plot the contour plot on folium
    g1.add_child(folium.GeoJson(
        geojson,
        style_function=lambda x: {
            'color':     x['properties']['stroke'],
            'weight':    x['properties']['stroke-width'],
            'fillColor': x['properties']['fill'],
            'opacity':   0.5,
        }))
    
    geomap.add_child(feature_group)
    geomap.add_child(g1)
    #folium.LayerControl().add_to(geomap)
    
    # Add the colormap to the folium map
    cm.caption = 'Kernel Density'
    geomap.add_child(cm)
    
    # Fullscreen mode
    plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap)
    display(geomap)
    #geomap.save(f'folium_contour_temperature_map.html')
    
    return(geomap)


# In[ ]:


def plot_Geom_Map_ForKDESamples(i_data,i_geomap,i_color,g2):
    
    #Heat map using folium
    Sample_Location = i_data.loc[:, ['Longitude', 'Latitude','Neighborhood']]
    for i in range(len(Sample_Location)):
        Longitude = Sample_Location.iloc[i][0]
        Latitude = Sample_Location.iloc[i][1]
        g2.add_child(
            folium.CircleMarker(
                location = [Latitude, Longitude],radius=1,color=i_color,
                fill=True,
                fill_color=i_color
            )
        )
        
    i_geomap.add_child(g2)  
    print("Map showing KDE with sample plots")
    display(i_geomap)
    
    return i_geomap,Sample_Location


# In[ ]:


'''Find nearest neighbors for all source points from a set of candidate points'''


# In[ ]:


#Method to calculate closest crime location for a police officer
def get_nearest(src_points, candidates, k_neighbors=1):

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


# In[ ]:


'''For each point in left GeoDataFrame, find closest point in right GeoDataFrame and return them.
Assumes input Points are in WGS84 projection (lat/lon)'''


# In[ ]:


def nearest_neighbor(left_gdf, right_gdf, return_dist=False):

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]
    

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius

    return closest_points


# In[ ]:


def Updated_Police_KDE(dc,dcf):
    closest_stops=None
    gdf_crime = gpd.GeoDataFrame(dc, geometry=gpd.points_from_xy(dc.Longitude, dc.Latitude))
    
    gdf_police = gpd.GeoDataFrame(dcf, geometry=gpd.points_from_xy(dcf.Longitude, dcf.Latitude))
    
    closest_stops = nearest_neighbor(gdf_police, gdf_crime, return_dist=True)
    
    print("Checking whether we have got new detected crimes == police length")
    print(len(closest_stops), '==', len(gdf_police))
    
    print("Police detecting crime within 5000mt(3.1 Miles) distance")
    closest_stops=closest_stops.loc[closest_stops['distance']<=5000]
    print("Length of detected crimes",len(closest_stops))
    
    return closest_stops


# In[ ]:


def plot_Geom_Map(data,i_map,g1):
   
    for i in range(len(data)):
        Longitude = data.iloc[i][0]
        Latitude = data.iloc[i][1]
        g1.add_child(folium.CircleMarker(location = [Latitude, Longitude],radius=7,color='blue',
        fill=True,
        fill_color='#00000000'))
        
        
    i_map.add_child(feature_group)
    i_map.add_child(g1)
    display(i_map)
    return i_map


# In[ ]:


def draw_balti_border(i_final_map,g1,i_type):

    #Add chloropleth layer: optional
    folium.Choropleth(
        geo_data='baltimore.json',
        name='choropleth',
        data=df_neighood,
        columns=['Name', 'Population'],
        key_on='feature.properties.name',
        fill_color='YlGn',
        fill_opacity=0.4,
        line_opacity=0.1,
        legend_name='Population').add_to(i_final_map)

    #Create Style function for GeoJson method
    style_function = lambda x: {
        'fill_color':'YlGn',
        'color':'black',
        'weight':0.8,
        'line_opacity':0.0,
        'fillOpacity': 0.0
    }
    
    if(i_type=='analysis'):
        print(i_type)
        #Create Style function for GeoJson method
        g1.add_child(folium.GeoJson(
            df_geojson,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['name', 'White','Blk_AfAm','Crime_Count','Police_Count','Detected_Crime_Count'],
                aliases=['Neighbourhood','White','Blk_AfAm','Crime_Count','Police_Count','Detected_Crime_Count'],
                localize=True
            )
        ))
    else:
         g1.add_child(folium.GeoJson(
            df_geojson,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['name','Male','Female', 'White','Blk_AfAm','AmInd_AkNa','Asian','NatHaw_Pac','Other_Race','TwoOrMore','Hisp_Lat'],
                aliases=['Neighbourhood','Male Population','Female Population','White','Blk_AfAm','AmInd_AkNa','Asian','NatHaw_Pac','Other_Race','TwoOrMore','Hisp_Lat'],
                localize=True
            )
        ))
    
    
    i_final_map.add_child(feature_group)
    i_final_map.add_child(g1)
    
    #Inorder to know labels in map while zooming we can use the followig: optional
    folium.map.CustomPane('labels').add_to(i_final_map)
    folium.TileLayer('CartoDBPositronOnlyLabels',
                     pane='labels').add_to(i_final_map)
    
    return (i_final_map)


# In[ ]:


for monthint in range(1,13):
    balti_map = folium.Map(location=balti_location, tiles="cartodbpositron", zoom_start=12)
    data_Crime=None
    data_police_COS=None
    map_for_KDE_crime=None
    map_for_Samples_crime=None
    map_for_KDE_police=None
    map_for_Samples_police=None
    final_map=None
    month = datetime.date(1900, monthint, 1).strftime('%B')
    print("Month: ",month)
    color_crime=['#FFFFB2', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#B10026']
    color_police=['#b2b2ff','#9999ff','#7f7fff','#6666ff','#4c4cff','#3232ff', '#0000ff']
    
    #Stimulate Crime KDE
    data_Crime=Filter_Crime_On_Month(df_Crime,monthint)  
    #Display Map for Crime Data
    dc_g = folium.plugins.FeatureGroupSubGroup(feature_group, 'KDE_MAP_ForCrime')
    map_for_KDE_crime=draw_KDE_Map(data_Crime,balti_map,dc_g,color_crime)
    dc_g = folium.plugins.FeatureGroupSubGroup(feature_group, 'Plot_ForCrime')
    map_for_Samples_crime,Crime_Location=plot_Geom_Map_ForKDESamples(data_Crime,map_for_KDE_crime,'#76B736',dc_g)

    #Stimulate Crime KDE
    if(monthint==1):
        #Stimulate Crime KDE
        data_police_COS=Filter_PoliceData_On_Month(df_police_COS)
        #Display Map for Police Data
        #The Police uses the crime KDE for 1st time
        dc_g = folium.plugins.FeatureGroupSubGroup(feature_group, 'Plot_ForPolice')
        map_for_Samples_police,Police_Location=plot_Geom_Map_ForKDESamples(data_police_COS,map_for_Samples_crime,'#922B5E',dc_g)

    else:
        #The police will update their KDE based on detected crime data
        dc_g = folium.plugins.FeatureGroupSubGroup(feature_group, 'KDE_MAP_ForPolice')
        map_for_KDE_police=draw_KDE_Map(updated_police_data,balti_map,dc_g,color_police)
        dc_g = folium.plugins.FeatureGroupSubGroup(feature_group, 'Plot_ForPolice')
        map_for_Samples_police,Police_Location=plot_Geom_Map_ForKDESamples(updated_police_data,map_for_Samples_crime,'#922B5E',dc_g)

    #Combine both map to find distance and plot one single map with detected crimes
    updated_police_data=Updated_Police_KDE(Crime_Location,Police_Location)


    print("Map Showing Detected Crimes")
    dc_g = folium.plugins.FeatureGroupSubGroup(feature_group, 'Detected_Crime')
    op_map=plot_Geom_Map(updated_police_data,map_for_Samples_police,dc_g)
    print("Map Showing Detected Crimes with neighbhorhood")
    dc_g = folium.plugins.FeatureGroupSubGroup(feature_group, 'Baltimore_neighbhoorhood')
    final_map=draw_balti_border(op_map,dc_g,'normal')
    folium.LayerControl().add_to(final_map)
    fname=str(monthint)+'_Map_For_Month_'+month+'.html'
    final_map.save(outfile=fname)
    display(final_map)

    for i in df_neighood['Name'].values:
        print(i.upper())
        d=Crime_Location[Crime_Location.Neighborhood.str.upper()==i.upper()]
        c=Police_Location[Police_Location.Neighborhood.str.upper()==i.upper()]
        e=updated_police_data[updated_police_data.Neighborhood.str.upper()==i.upper()]
        #df_geojson = df_geojson.append({'Crime_Count' : d.Neighborhood.count()} , ignore_index=True)
        df_geojson.loc[(df_geojson.name == i),'Crime_Count']=d.Neighborhood.count()
        df_geojson.loc[(df_geojson.name == i),'Police_Count']=c.Neighborhood.count()
        df_geojson.loc[(df_geojson.name == i),'Detected_Crime_Count']=e.Neighborhood.count()
        
    v_map = folium.Map(location=balti_location, tiles="cartodbpositron", zoom_start=12)
    analysis_map=draw_balti_border(v_map,dc_g,'analysis')
    folium.LayerControl().add_to(analysis_map)
    fname_analys=str(monthint)+'_Analysis_Map_For_Month_'+month+'.html'
    analysis_map.save(outfile=fname_analys)
    display(analysis_map)


# In[ ]:


df_geojson


# In[ ]:




