# Indentifying accumulation and leakage with Spearmans Rho

Data from a national beach litter survey of lakes and rivers is compared to landuse using Spearmans rank correlation coeffeicient. Spearmans Rho is easy to implement in python or a spreadsheet and may serve as an initial automated assessment for local administrations when considering the results of beach litter surveys.

The value of Rho is defined for selected objects that reach a user defined frequency (ratio of samples where some were found / the number of samples) or density (number of objects/100 m) threshhold. The frequency or abundance is considered with respect to the landuse categories defined by the Swiss National Statistical Survey. The threshold values were tested against diffrent buffer zones around the survey locations.

_Measuring land use_
![distance to intersection](https://github.com/hammerdirt-analyst/landuse/blob/main/resources/images/stream_length_buffer_land_use.jpeg)

## Contents

__Note:__ These are large files. Git Large File Storage (Git LFS) is enabled, if the .csv files do not download check [Git LFS](https://git-lfs.com/)

### Resources

* The complete data set from the 2020-2021 beach-litter survey. The project was sponsored by the Swiss Confederation, the executive summary and regional reports can be found here [_IQAASL end of sampling_](https://hammerdirt-analyst.github.io/IQAASL-End-0f-Sampling-2021/titlepage.html).

### Notebooks


#### Spearmans Rho for all locations:

1. project_results.ipynb : the test was conducted at different buffer radiuses 1'500 m - 10'000 m
2. hex-3000-m.ipynb: the test at 1'500 m was is repeated with the most recent map values using only lakes
3. hex-3000-m-rivers.ipynb: the test at 1'500 m was is repeated with the most recent map values using only rivers

#### Length to discharge point and length of river section:

1. consider_distance_to_river.ipynb

#### Probability

1. probability.ipynb

The data was collected as part of publicly funded project. Please acknowledg appropriately.

For more information contact analyst@hammerdirt.

__Love__ what you do everyday


 
