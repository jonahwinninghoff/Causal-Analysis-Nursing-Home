# Causal Analysis: Effects of Nursing Home Facilities on Health Inspection Rating

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**[Overview](#overview)** | **[Method](#method)** | **[Datasets](#data)** | **[Statistics](#statistics)** | **[Wrangling](#wrangling)** | **[Modeling](#model)**


## OVERVIEW <a id='overview'></a>

<p align="justify"> Established by the United States Centers for Medicare and Medicaid Services (CMS) in 2009, the star rating is a system that helps people to make decisions in which nursing home the senior residents would reside. The decision process to choosing the right facility is not easy and soulwrenching. However, during the COVID-19 pandemic, this system is a failure largely due to a lack of data audit and self-report bias. For example, the hypothesis testing is unable to confirm that number of COVID-19 deaths at five-star facilities are different from one-star facilities at the significance level (Silver-Greenberg and Gebeloff, 2021). The stipulation in this study that demonstrates this issue is that, implicitly, the CMS institution faces a crisis of decline in public trust. For that specific reason, the causal analysis is a linchpin in addressing this problem.</p>

## METHOD <a id='method'></a>

<p align ='justify'> In attempting to establish causal relationship, there are two different frameworks, which are data science and econometrics. The data science approach begins by searching patterns in the data in order to test model against data. But the role of econometrics is reversed. For example, the ecnometric approach begins by writing a causal model of economic behavior and its underyling assumptions, followed by determining whether the available data fits in the causal model.</p>

## DATASETS <a id ='data'></a>

<p align = 'justify'> Two datasets obtain from CMS databases are <i> Minimum Data Set </i> (MDS) <i> Quality Measures </i> and <i>Provider Information </i> datasets. The MDS dataset contains over 15,000 different providers from 50 states plus District of Columbia. The target variable is measure quality score. But none of variables holds predictive power for measure quality score. Some features are useful for statistical insights. The second dataset contains more than 80 features with at least 15,000 entities. At least 70 features are usable for prediction.</p>

- [MDS Quality Measures](https://github.com/jonahwinninghoff/Springboard_Capstone_Project/blob/main/Assets/NH_QualityMsr_MDS_Jun2021.csv.zip?raw=true)
- [Provider Information](https://github.com/jonahwinninghoff/Springboard_Capstone_Project/blob/main/Assets/NH_ProviderInfo_Aug2021.csv.zip?raw=true)

## STATISTICAL INFERENCE <a id ='statistics'></a> 

<p align = 'justify'> The meausre quality in the MDS dataset is in use to describe the overall rating of what each facility does well associated with every measure code. The measure code associates with measure description that explains how this score is calculated. But as indicated by the complete data quality report, the score is not normally distributed. The Empirical Cumulative Distribution Function (ECDF) tool is undertaken to compare empirical distribution with theoretical  <i>Beta</i> distribution and determine if the empirical distribution is parametric. The alpha and beta for theoretical one is unknown. The identifications for both parameters are:</p> 

<div align = 'center'> <img src="https://latex.codecogs.com/svg.image?\hat\alpha&space;=&space;\bar&space;x&space;\left[\frac{\bar&space;x&space;(1-&space;\bar&space;x)}{s^2&space;-&space;1}\right]" title="\hat\alpha = \bar x \left[\frac{\bar x (1- \bar x)}{s^2 - 1}\right]" />  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="https://latex.codecogs.com/svg.image?\hat\beta&space;=&space;(1-&space;\bar&space;x)&space;\left[&space;\frac{\bar&space;x&space;(1-&space;\bar&space;x)}{s^2-1}&space;\right]&space;" title="\hat\beta = (1- \bar x) \left[ \frac{\bar x (1- \bar x)}{s^2-1} \right] " /> </div>

<p align = 'justify'>Because this continuous score is ε(0,100) and cross-sectional, the continuous version of binomial called <i>Beta</i> distribution is use. Both parameters help build this theoretical distribution using random generator (Sinharay, 2010). As the result shown, the empirical distribution is not consistent with the theoretical one. The shape of distribution is unlikely to alter due to Law of Large Number. The adjusted density plot is a demonstration.</p>

![](https://user-images.githubusercontent.com/52142917/137056882-eb193352-6e31-489b-80d9-b5b9aa992b89.png)

<p align = 'justify'>The permutation test can be instead used for testing the null hypothesis. This hypothesis is that two groups are within the same distribution. Since this hypothesis tests against 13 different measure codes, the alpha level with Bonferroni correction is 0.38%. The purpose of Bonferroni correction is to ensure that the chance of Type I Error is minimal.</p>

![image](https://user-images.githubusercontent.com/52142917/137058706-77dec24b-4136-4a86-a0a0-bb1a3c404959.png)

<p align = 'justify'> The expected difference is by averaging scores tied with particular measure code and subtracting by these without this code. As indicated by the plot, the confidence interval is difficult to be seen due to small standard errors. The margins of errors are between 0.001 and 0.002. As the plot shown, the observed difference for catheter-related measure corde is lower than -40 while the observed difference for depressive-related measure code is higher than 60. Intuitively, the facilities are likely to perform poorly with treating residetns who have catheter inserted and left in their bladders while they do well with treating residents who have depressive symptoms. However, this plot is not established with causal relationship due to self-report bias (CMS, 2021).</p>

## DATA WRANGLING <a id ='wrangling'></a>

<p align = 'justify'> As mentioned earlier, the <i> Provider Information </i> dataset has many features. The challenge with this dataset is that several features are redundant (some are perfectly correlated) and wrangling process requires automation techniques to identify redundancies and leakages. The leakage refers to which the features in the training process do not exist when integrating the production, in turn, causes the predictive scores to overestimate. This is common mistake in data science. For example, the total weighted health survey score as a feature predicts the health inspection rating is a form of leakage. The final process is to investigate if several leaked features are overlooked. As a result, there are 30 leaked features found in this dataset. As a result, the number of features is reduced to 37. </p>

## PREDICTIVE MODELING <a id ='model'></a>

<p align = 'justify'>The objective for predictive modeling is that a model should explain at least 80% of the variance for the target variable and it should be generalizable. Not only that, it should be well-calibrated. In order to determine if the criteria could be met, this dataset separates into three sets: training set, validation set, and testing set. The training set is in use for a model to learn while the validation set is in use for tuning hyperparameter. Finally, the testing set is unseen dataset. The feature and model selections are undertaken to maximize model performance. </p> 
  
<p align = 'justify'> But there are 36 features in total besides target variable. An attempt to select features manually is not possible since the total possible combinations are 68,719,476,736. Two automations are, for that specific reasons, in use in attempting to optimize a model, which are the least shrinkage and selection operator (lasso) and Bayes optimal feature selection.</p>

The lasso regularization
