# Assessing land-use influence on macrolitter abundance in freshwater systems

Notes and calculations for the manuscript *__Near or Far: Revealing the role of land-use on macrolitter
distribution in Swiss freshwater__*

_ver=0.01_

_Suvey locations April - August 2021 source:_ [IQAASL](https://hammerdirt-analyst.github.io/IQAASL-End-0f-Sampling-2021/index.html)
:::{image} resources/images/intro_map.jpeg
:alt: Map of IQAASL locations
:class: bg-primary mb-1
:width: 1200px
:align: center
:::


## Abstract

Rivers are often considered conduits for macrolitter pollution into the sea. However, freshwater ecosystems are
also themselves highly exposed to macrolitter detrimental effects, such as ingestion by local fauna. Long-term and large-scale
assessments on macrolitter on riverbanks and lake shores can provide an understanding of litter presence in freshwater systems,
notably in terms of abundance, composition and origin of items.

## Purpose

This document details the methods used for the land use chapter of the IQAASL report and expands the analysis to include distance to river intersections and length of river network. The environmental conditions are approximated by calculating the land-use conditions within a buffer of r=1'500 m around each survey location. The data is extracted from [swissTLMRegio](https://www.swisstopo.admin.ch/de/geodata/landscape/tlmregio.html) using the predefined land use categories in the corresponding vector layers. In section 4 the measured distance from a survey location to any river intersections within the buffer is measured as well as the total length of any intersecting rivers in the buffer. In section five the land-use is calculated with a hexagon shaped buffer (as opposed to circular buffer), and the most recent publicly available map layers are substitued for the previous versions used in the first three sections. The purpose can be summarized as follows:

1. review the previous analysis
2. add to the previous analysis by selecting objects based on abundance or frequency
3. determine if length and distance to river intersection is a better indicator of pollution source than just the number of intersections
4. quantify the analysis in terms of number of objects accounted for
5. update the inferences using the new map layers
6. establish the method for infering trash densities using survey results and environmental conditions
5. Compare the results of Spearmans Rho to those from a Bayesien inference table

## Introduction

To better control (eliminate) trash in the environment it is essential that the limited resources that are attributed to this domain be used as efficiently as possible. Correctly identifying zones of accumulation and or the objects that are accumulating in the watershed would enable more coordinated and precise actions between stakeholders. The process of identification needs to fulfill certain operating requirements to be effective:

* accurate
* repeatable
* scale-able (up and down)

Beach-litter data is `count` data. Gathered by volunteers following a protocol, the data is highly variable for many reasons. In most studies the median is less than the mean and in the case of the data for this study, the standard deviation is greater than the mean. Statistical tests dependent on linear relationships may not be appropriate, the guide from the JRC suggests using a Negative binomial distribution for modeling extreme events. Spearman's &rho; does not require that the two variables be continuous, nor is there an assumption of _normality_. The test is included in most standard computing libraries and spreadsheets, the process is easy to automate and integrate.  Spearman's &rho; or Spearman's rank correlation coefficient is a non parametric test of rank correlation between two variables. Spearmans &rho; defines the magnitude (how much it approaches linear) of monotonic relationships and the direction. When &rho; is 0 there is no evidence of a monotonic relationship. Values of 1 and -1 signify a perfect monotonic relationship between two variables.

The calculations are the same at each Geographic level. The number of samples and the mix of land-use attributes changes with every subset of data. It is those changes and how they relate to the magnitude of trash encountered that concerns these inquiries. This document assumes the reader knows what beach-litter monitoring is and how it applies to the health of the environment.

A statistical test is not a replacement for common sense. It is an another piece of evidence to consider along with the results from previous studies, the researchers personal experience as well as the history of the problem within the geographic constraints of the data.

__Authors:__ Louise Schreyers, Roger Erismann, Montserrat Filella, Christian Ludwig

For information regarding the contents of this document conata analyst@hammerdirt.ch