#!/usr/bin/env python
# coding: utf-8

# In[1]:


# sys, file and nav packages:
import datetime as dt
import json
import functools
import time

# math packages:
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF

# charting:
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns

import IPython
from PIL import Image as PILImage
from IPython.display import Markdown as md
from IPython.display import display
from myst_nb import glue

import time

start_date = '2020-03-01'
end_date ='2021-05-31'

a_qty = 20

a_fail_rate = .5

use_fail = False

unit_label = 'p/100m'

# survey data:
dfx= pd.read_csv('resources/checked_sdata_eos_2020_21.csv')

dfBeaches = pd.read_csv("resources/beaches_with_land_use_rates.csv")
dfCodes = pd.read_csv("resources/codes_with_group_names_2015.csv")

# set the index of the beach data to location slug
dfBeaches.set_index('slug', inplace=True)

# set the index of to codes
dfCodes.set_index("code", inplace=True)

# code description map
code_d_map = dfCodes.description

# code material map
code_m_map = dfCodes.material

pdtype = pd.core.frame.DataFrame
pstype = pd.core.series.Series

def scaleTheColumn(x):
    
    xmin = x.min()
    xmax = x.max()
    xscaled = (x-xmin)/(xmax-xmin)
    
    return xscaled

def rotateText(x):
    return 'writing-mode: vertical-lr; transform: rotate(-180deg);  padding:10px; margins:0; vertical-align: baseline;'

def cleanSurveyResults(data):
    # performs data cleaning operations on the
    # default data
    
    data['loc_date'] = list(zip(data.location, data["date"]))
    data['date'] = pd.to_datetime(data["date"])
    
    # get rid of microplastics
    mcr = data[data.groupname == "micro plastics (< 5mm)"].code.unique()
    
    # replace the bad code
    data.code = data.code.replace('G207', 'G208')
    data = data[~data.code.isin(mcr)]
    
    # walensee has no landuse values
    data = data[data.water_name_slug != 'walensee']   
    
    return data

class SurveyResults:
    """Creates a dataframe from a valid filename. Assigns the column names and defines a list of
    codes and locations that can be used in the CodeData class.
    """
    
    file_name = 'resources/checked_sdata_eos_2020_21.csv'
    columns_to_keep=[
        'loc_date',
        'location', 
        'river_bassin',
        'water_name_slug',
        'city',
        'w_t', 
        'intersects', 
        'code', 
        'pcs_m',
        'quantity'
    ]
        
    def __init__(self, data: str = file_name, clean_data: bool = True, columns: list = columns_to_keep, w_t: str = None):
        self.dfx = pd.read_csv(data)
        self.df_results = None
        self.locations = None
        self.valid_codes = None
        self.clean_data = clean_data
        self.columns = columns
        self.w_t = w_t
        
    def validCodes(self):
        # creates a list of unique code values for the data set    
        conditions = [
            isinstance(self.df_results, pdtype),
            "code" in self.df_results.columns
        ]

        if all(conditions):

            try:
                valid_codes = self.df_results.code.unique()
            except ValueError:
                print("There was an error retrieving the unique code names, self.df.code.unique() failed.")
                raise
            else:
                self.valid_codes = valid_codes
                
        
    def surveyResults(self):
        
        # if this method has been called already
        # return the result
        if self.df_results is not None:
            return self.df_results
        
        # for the default data self.clean data must be called        
        if self.clean_data is True:
            fd = cleanSurveyResults(self.dfx)
            
        # if the data is clean then if can be used directly
        else:
            fd = self.dfx
        
        # filter the data by the variable w_t
        if self.w_t is not None:
            fd = fd[fd.w_t == self.w_t]            
         
        # keep only the required columns
        if self.columns:
            fd = fd[self.columns]
        
        # assign the survey results to the class attribute
        self.df_results = fd
        
        # define the list of codes in this df
        self.validCodes()
        
        return self.df_results
    
    def surveyLocations(self):
        if self.locations is not None:
            return self.locations
        if self.df_results is not None:
            self.locations = self.dfResults.location.unique()
            return self.locations
        else:
            print("There is no survey data loaded")
            return None    




# this defines the css rules for the note-book table displays
header_row = {'selector': 'th:nth-child(1)', 'props': f'background-color: #FFF;'}
even_rows = {"selector": 'tr:nth-child(even)', 'props': f'background-color: rgba(139, 69, 19, 0.08);'}
odd_rows = {'selector': 'tr:nth-child(odd)', 'props': 'background: #FFF;'}
table_font = {'selector': 'tr', 'props': 'font-size: 12px;'}
table_css_styles = [even_rows, odd_rows, table_font, header_row]

# the intersect data
dtoi_o = pd.read_csv("resources/buffer_output/distance_to_intersection.csv")

columns = [ "river_bass", "feature", "city", "location", "NAMN_2", "BREITE", "KLASSE_2", "HOC", "feature", "distance", "OBJVAL"]
dtoi = dtoi_o[columns].copy()
rename = {"NAMN_2":"name", "BREITE":"size", "KLASSE_2":"class", "HOC":"hoc", "NAMN":"name","KLASSE":"class"}
dtoi.rename(columns=rename, inplace=True)

# designate a column to merge on
dtoi["merge_col"] = list(zip(dtoi.location, dtoi["name"], dtoi["size"], dtoi["class"]))
dtoi.drop_duplicates("merge_col", inplace=True)

# the length data
dtoi_l = pd.read_csv("resources/buffer_output/intersection_length.csv")
columns = ["river_bass", "feature", "city", "location", "NAMN", "BREITE", "KLASSE", "HOC", "feature", "length", "OBJVAL"]
dtol = dtoi_l[columns].copy()
dtol.rename(columns=rename, inplace=True)

# designate a column to merge on
dtol["merge_col"] = list(zip(dtol.location, dtol["name"], dtol["size"], dtol["class"]))

# merge the lenght and intersection data
these_merge_cols = ["length","name","merge_col"]
ind = dtoi.merge(dtol[these_merge_cols], on="merge_col")
ind = ind[["location", "name_x","distance", "length", "size", "class"]].copy()

# collecting survey data
fdx = SurveyResults()
df = fdx.surveyResults()
df = df[df.location.isin(ind.location.unique())].copy()
df = df[df.w_t != "r"]

no_luse_data = ["linth_route9brucke",
                "seez_spennwiesenbrucke",
                'limmat_dietikon_keiserp',
                "seez"]

# use the same criteria from the porject results
codes = df[df.quantity > 20].code.unique()

ints_and_data = df[["loc_date","location", "city", "code", "pcs_m"]].merge(ind, on="location")

locations = df.location.unique()

data = ints_and_data[(ints_and_data.code.isin(codes)) & (ints_and_data.location.isin(locations))].copy()
data.fillna(0, inplace=True)

columns = ["distance", "length", "size", "class"]

def collectCorrelation(data, codes, columns):
    results = []
    for code in codes:
        d = data[data.code == code]
        dx = d.pcs_m.values
        for name in columns:
            dy = d[name].values
            c, p = stats.spearmanr(dx, dy)
            
            results.append({"code":code, "variable":name, "rho":c, "p":p})
    return results

def resultsDf(rhovals: pdtype = None, pvals: pdtype = None)-> pdtype:
    results_df = []
    for i, n in enumerate(pvals.index):
        arow_of_ps = pvals.iloc[i]
        p_fail = arow_of_ps[ arow_of_ps > 0.05]
        arow_of_rhos = rhovals.iloc[i]
        
        for label in p_fail.index:
            arow_of_rhos[label] = 0
        results_df.append(arow_of_rhos)
    
    return results_df

def styleBufferResults(buffer_results):
    buffer_results.columns.name = None
    bfr = buffer_results.style.format(precision=2).set_table_styles(table_css_styles)
    bfr = bfr.background_gradient(axis=None, vmin=buffer_results.min().min(), vmax=buffer_results.max().max(), cmap="coolwarm")
    bfr = bfr.applymap_index(rotateText, axis=1)
    
    return bfr           
            


corellation_results = collectCorrelation(data, codes, columns)
crp = pd.DataFrame(corellation_results)
pvals = crp.pivot(index="code", columns="variable", values="p")
rhovals = crp.pivot(index="code", columns="variable", values="rho")


# # River discharge and lake intersections
# 
# In the intitial report and in the project-results sample the influence of river inputs was quantitied by the number of river intersects within 1500 m of a survey location. With this method 13 possible correlations were identified, 11 positive and two negative. This method does not take into account the distance to the intersection, the lenght of the river section withing the 1500 m buffer nor does it consider the size of the inputs.
# 
# Here we consider the distance, the length, the size and the class of each river within 2 km of the survey location. 

# ```{figure} resources/images/stream_length_buffer_land_use.jpeg
# ---
# name: dist_to_int
# ---
# ` `
# ```
# {numref}`figure %s: <dist_to_int>` Measuring the distance to the intersection and length of the intersection in the 2 k buffer. Location: grand-clos, St. Gingolph - Lac Léman.

# ## Extracting the values from the map layer
# 
# The map layers that are publicly available have changed since the land-use attributes were originally considered for the project. At that time we did not consider the length or distance. The size and the class of each river was not indicated on the previous map layers either. All that has changed:
# 
# 1. There are fewer rivers and streams in the new map layers
# 2. Each river (section) is labled with the size, class, name and designated as man-made or natural.
# 
# To extract the required data for the analysis for each location and river the following steps were followed:
# 
# 1. Identify locations of interest
# 2. Construct a buffer around each point
# 3. Mark the intersection of the river with the buffer and the lake
# 4. Calculate the length of that section
# 5. Calculate the straight line distance from the survey location to the point where the river leaves the buffer and enters the lake
# 
# Most locations have more than one intersection. Which means that the survey result for a code is considered under all the possible conditions for each location. The results from St. Gingolph illustrate this:

# In[2]:


data[(data.location == 'grand-clos') & (data.code == "Gfrags")&(data.loc_date == ("grand-clos", "2020-05-07"))].head()


# ## The data
# 
# Only surveys from lakes are considered. While it is possible to do the same analysis on river locations, the results would not be comparable. Lakes are zones of low flow in a river bassin. When products/objects enter the lake from a river, they go from a zone of high flow to low flow. Objects of different densities may travel different distances once they hit the lake.

# In[3]:


## The survey results
locations = df.location.unique()
samples = df.loc_date.unique()
lakes = df[df.w_t == "l"].drop_duplicates("loc_date").w_t.value_counts().values[0]

codes_identified = df[df.quantity > 0].code.unique()
codes_possible = df.code.unique()
total_id = df.quantity.sum()

data_summary = {
    "n locations": len(locations),
    "n samples": len(samples),
    "n lake samples": lakes,
    "n identified object types": len(codes_identified),
    "n possible object types": len(codes_possible),
    "total number of objects": total_id
}

pd.DataFrame(index = data_summary.keys(), data=data_summary.values(), columns=["total"]).style.set_table_styles(table_css_styles)


# In[4]:


styleBufferResults(pd.DataFrame(resultsDf(rhovals, pvals)))


# __Notes:__
# 
# __negative correlations__ the size and class are in the inverse of the distance and lenght paramenters. That is that rivers with a large size paramater are smaller than those with a small size parameter. The class parameter is the stream classification in relation to the ocean. Therefore, objects that are positively correlated with size and class were found more often at the intersects of smaller and less important rivers.

# ### Total correlations, total positive correlations, total negative corrrelations

# In[5]:


def countTheNumberOfCorrelationsPerBuffer(pvals: pdtype = None, rhovals: pdtype = None) -> (pdtype, pstype):
    
    # the number of times p <= 0.05
    number_p_less_than = (pvals <= 0.05).sum()
    number_p_less_than.name = "correlated"
    
    # the number of postive correlations
    number_pos = (rhovals > 0).sum()
    number_pos.name = "positive"
    
    # the number of negative correlations
    number_neg = (rhovals < 0).sum()
    number_neg.name = "negative"

    ncorrelated = pd.DataFrame([number_p_less_than, number_pos, number_neg])
    ncorrelated["total"] = ncorrelated.sum(axis=1)
    totals = ncorrelated.total
    
    
    return ncorrelated, totals

ncorrelated, total = countTheNumberOfCorrelationsPerBuffer(pvals, rhovals)
ncorrelated


# In[6]:


today = dt.datetime.now().date().strftime("%d/%m/%Y")
where = "Biel, CH"

my_block = f"""

This script updated {today} in {where}

> \u2764\ufe0f what you do everyday

_ANALYSTATHAMMERDIRT_
"""

md(my_block)

