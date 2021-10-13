# Causal Analysis: Effects of Nursing Home Facilities on Health Inspection Rating

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**[Overview](#overview)** | **[Method](#method)** | **[Datasets](#data)** | **[Statistics](#statistics)**


## OVERVIEW <a id='overview'></a>

<p align="justify"> Established by the United States Centers for Medicare and Medicaid Services (CMS) in 2009, the star rating is a system that helps people to make decisions in which nursing home the senior residents would reside. The decision process to choosing the right facility is not easy and soulwrenching. However, during the COVID-19 pandemic, this system is a failure largely due to a lack of data audit and self-report bias. For example, the hypothesis testing is unable to confirm that number of COVID-19 deaths at five-star facilities are different from one-star facilities at the significance level (Silver-Greenberg and Gebeloff, 2021). The stipulation in this study that demonstrates this issue is that, implicitly, the CMS institution faces a crisis of decline in public trust. For that specific reason, the causal analysis is a linchpin in addressing this problem. </p>

## METHOD <a id='method'></a>

<p align ='justify'> In attempting to establish causal relationship, there are two different frameworks, which are data science and econometrics. The data science approach begins by searching patterns in the data in order to test model against data. But the role of econometrics is reversed. For example, the ecnometric approach begins by writing a causal model of economic behavior and its underyling assumptions, followed by determining whether the available data fits in the causal model. </p>

## DATASETS <a id ='data'></a>

<p align = 'justify'> Two datasets obtain from CMS databases are Minimum Data Set (MDS) Quality Measures and Provider Information datasets. The MDS dataset contains over 15,000 different providers from 50 states plus District of Columbia. The target variable is measure quality score. But none of variables holds predictive power for measure quality score. Some features are useful for statistical insights. The second dataset contains more than 80 features with at least 15,000 entities. At least 70 features are usable for prediction. </p>

- [MDS Quality Measures](https://github.com/jonahwinninghoff/Springboard_Capstone_Project/blob/main/Assets/NH_QualityMsr_MDS_Jun2021.csv.zip?raw=true)
- [Provider Information](https://github.com/jonahwinninghoff/Springboard_Capstone_Project/blob/main/Assets/NH_ProviderInfo_Aug2021.csv.zip?raw=true)

## STATISTICAL INFERENCE <a id ='statistics'></a> 

<p align = 'justify'> The meausre quality in the MDS dataset is in use to describe the overall rating of what each facility does well associated with every measure code. The measure code associates with measure description that explains how this score is calculated. But as indicated by the complete data quality report, the score is not normally distributed. The Empirical Cumulative Distribution Function (ECDF) tool is undertaken to compare empirical distribution with theoretical *Beta* distribution and determine if the empirical distribution is parametric. The alpha and beta for theoretical one is unknown. The identifications for both parameters are: </p> 

<div align = 'center'> <img src="https://latex.codecogs.com/svg.image?\hat\alpha&space;=&space;\bar&space;x&space;\left[\frac{\bar&space;x&space;(1-&space;\bar&space;x)}{s^2&space;-&space;1}\right]" title="\hat\alpha = \bar x \left[\frac{\bar x (1- \bar x)}{s^2 - 1}\right]" />  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="https://latex.codecogs.com/svg.image?\hat\beta&space;=&space;(1-&space;\bar&space;x)&space;\left[&space;\frac{\bar&space;x&space;(1-&space;\bar&space;x)}{s^2-1}&space;\right]&space;" title="\hat\beta = (1- \bar x) \left[ \frac{\bar x (1- \bar x)}{s^2-1} \right] " /> </div>

<p align = 'justify'>Because this continuous score is ε(0,100) and cross-sectional, the continuous version of binomial called *Beta* distribution is use. Both parameters help build this theoretical distribution using random generator (Sinharay, 2010). As the result shown, the empirical distribution is not consistent with the theoretical one. The shape of distribution is unlikely to alter due to Law of Large Number. The adjusted density plot is a demonstration. </p>
