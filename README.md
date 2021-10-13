# Causal Analysis: Effects of Nursing Home Facilities on Health Inspection Rating

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**[Overview](#overview)** | **[Method](#method)** | **[Datasets](#data)** | **[Statistics](#statistics)** | **[Wrangling](#wrangling)** | **[Modeling](#model)** | **[Econometrics](#econ)** | **[Insights](#insights)**


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

![image](https://user-images.githubusercontent.com/52142917/137194232-c0272480-efe7-4d39-bb61-7b11cae9b590.png)

<p align = 'justify'>The permutation test can be instead used for testing the null hypothesis. This hypothesis is that two groups are within the same distribution. Since this hypothesis tests against 13 different measure codes, the alpha level with Bonferroni correction is 0.38%. The purpose of Bonferroni correction is to ensure that the chance of Type I Error is minimal.</p>

![image](https://user-images.githubusercontent.com/52142917/137194101-13d8e2e6-f4f9-4066-9373-5880afd38b7c.png)

<p align = 'justify'> The expected difference is by averaging scores tied with particular measure code and subtracting by these without this code. As indicated by the plot, the confidence interval is difficult to be seen due to small standard errors. The margins of errors are between 0.001 and 0.002. As the plot shown, the observed difference for catheter-related measure corde is lower than -40 while the observed difference for depressive-related measure code is higher than 60. Intuitively, the facilities are likely to perform poorly with treating residetns who have catheter inserted and left in their bladders while they do well with treating residents who have depressive symptoms. However, this plot is not established with causal relationship due to self-report bias (CMS, 2021).</p>

## DATA WRANGLING <a id ='wrangling'></a>

<p align = 'justify'> As mentioned earlier, the <i> Provider Information </i> dataset has many features. The challenge with this dataset is that several features are redundant (some are perfectly correlated) and wrangling process requires automation techniques to identify redundancies and leakages. The leakage refers to which the features in the training process do not exist when integrating the production, in turn, causes the predictive scores to overestimate. This is common mistake in data science. For example, the total weighted health survey score as a feature predicts the health inspection rating is a form of leakage. The final process is to investigate if several leaked features are overlooked. As a result, there are 30 leaked features found in this dataset. As a result, the number of features is reduced to 37. </p>

## PREDICTIVE MODELING <a id ='model'></a>

<p align = 'justify'>The objective for predictive modeling is that a model should explain at least 80% of the variance for the target variable and it should be generalizable. Not only that, it should be well-calibrated. In order to determine if the criteria could be met, this dataset separates into three sets: training set, validation set, and testing set. The training set is in use for a model to learn while the validation set is in use for tuning hyperparameter. Finally, the testing set is unseen dataset. The feature and model selections are undertaken to maximize model performance. </p> 
  
<p align = 'justify'> But there are 36 features in total besides target variable. An attempt to select features manually is not possible since the total possible combinations are 68,719,476,736. Two automations are, for that specific reasons, in use in attempting to optimize a model, which are the least shrinkage and selection operator (lasso) and Bayes optimal feature selection.</p>

<p align = 'justify'>  The lasso regularization is a popular method that proves to be successful in data science. This function is robust to the outlier but it is not differentiable due to piecewise function inderivative (Boehmke, 2021). This algorithm the ability to identify unimportant predictors due to sparsity (predictors are set to 0). The Bayes optimal feature selection is different becuase it is an iterative process by balancing its needs of exploration and exploitation depending on three functions.</p>

- <p align = 'justify'> <b>Objective function:</b> the true shape of this function is not observable and it can only reveal some data points that can otherwise be expensive to compute.</p>

- <p align = 'justify'> <b>Surrogate function:</b> the probablistic model is being built to exploit what is known and it alters in light of new information.</p>

- <p align = 'justify'> <b>Acquistion function:</b> this function is to calculate a vector of hyperparameters that likely yield higher local maximum of objective function using surrogate function.</p>

<p align = 'justify'>One of these studies shows that this approach has the ability to compete with many state-of-the-art methods (Ahmed et al., 2015).</p>

<p align = 'justify'> As a result, the best lambda for lasso is 0.005 and the R-squared score is 32.19% for the validation set while the R-squared score for Bayes optimal feature selection is 32.87%. However, the number of features in total for lasso is 16, which is lower than this Bayesian approach (equal to 21 features in total).</p>

<p align = 'justify'>The best model is the light gradient boosting, which equal to 43.47% for validation set. This model is already optimized using Bayesian search theory. The R-squared score for testing set is 41.92%. The mean absolute error (MAE) score is 0.77. For example, if this model predicts that the health inspection rating is 3.5, the acutal score may fall somewhere between 2.73 and 4.27. This error scores indicates how large the error is by averaging the difference between predicted and acutal values.</p>

## ECONOMETRIC METHOD <a id = 'econ'></a>

<p align = 'justify'> To the extent that the causal relationship fails to establish, the previous discussion indicates the limits of data science approach. Resolving these issues are by adopting the econometric approach. Some suggest that the minimal requirement for this approach is that the adjusted R-squared score must be positive. But in technical sense. this approach stipulates with model misspecification. More importantly, Gauss Markov assumptions are fundamental of the econometric approach, though the causal may depart from some assumptions, as the following: </p>

- **A1:** linearity in parameters
- **A2:** no perfect collinearity
- **A3:** zero conditional mean error
- **A4:** homoskedasticity and no serial correlation
- **A5:** normality of the error

<p align = 'justify'>For this analysis, the causal model does not follow A1 and A4 assumptions. The endogenous variable (or target variable) is limited between 1 and 5 points that is also known to be Limited Dependent Variable (LDV). The popular solution is logarithmic transformation without Monte Carlo simulation is, unfortunately, incorrect. This particular model misspecification is called Duan's Smear.</p>

<div align = 'center'><img src="https://latex.codecogs.com/svg.image?log(y_i)=x_i&space;\beta&space;&plus;&space;u_i" title="log(y_i)=x_i \beta + u_i" /></div>

<div align = 'center'><img src="https://latex.codecogs.com/svg.image?e^{log(y_i)}=e^{x_i&space;\beta&space;&plus;&space;u_i}" title="e^{log(y_i)}=e^{x_i \beta + u_i}" /></div>

<div align = 'center'><img src="https://latex.codecogs.com/svg.image?E\left&space;[&space;y_i&space;|&space;x_i&space;\right&space;]=e^{x_i&space;\beta}\int&space;&space;e^{u_i}&space;\partial&space;u_i" title="E\left [ y_i | x_i \right ]=e^{x_i \beta}\int e^{u_i} \partial u_i" /></div>

<p align = 'justify'>This logarithmic transformation algebra demonstrates the mathematical issues with assumption that error term is independent of exogenous variable (or predictor). The logarithmic transformation may be in violation of A3 due to exogenity. This term refers to which the exogenous variable is correlated with error term.</p>

<div align = 'center'><img src="https://latex.codecogs.com/svg.image?e^{x_i&space;\beta}&space;&plus;&space;u_i&space;\not\approx&space;e^{x_i&space;\hat\beta}&space;\frac{1}{n}\sum_{i=1}^n&space;e^{\hat&space;u_i}" title="e^{x_i \beta} + u_i \not\approx e^{x_i \hat\beta} \frac{1}{n}\sum_{i=1}^n e^{\hat u_i}" /></div>

The solution is to use nonlinear least square called Probit with Quasi-Maximum Likelihood Estimation condition (QMLE). This model is more efficient to heteroskedasticity. The heteroskedasticity is the variance of residual term that is not constant through the regression. For example, the causla model is consistent and asymptotically normal where V is not proportional to A. The assumption for consistency is:

<div align = 'center'><img src="https://latex.codecogs.com/svg.image?\sqrt{n}(\hat\theta_{QMLE}&space;-&space;\theta_0)\sim^a&space;N(0,A^{-1}VA^{-1})&space;" title="\sqrt{n}\hat\theta_{QMLE} - \theta_0)\sim^a N(0,A^{-1}VA^{-1}) " /></div>

<div align = 'center'><img src="https://latex.codecogs.com/svg.image?A&space;=&space;-&space;E\left&space;[&space;H(\omega_i,\theta_0)&space;\right&space;]" title="A = - E\left [ H(\omega_i,\theta_0) \right ]" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/svg.image?V&space;=&space;E[&space;s(\omega_i,\theta_0)s(\omega_i,\theta_0)']" title="V = E[ s(\omega_i,\theta_0)s(\omega_i,\theta_0)']" /></div>

<p align = 'justify'>Because the Breusch-Pagan test confirms that the null hypothesis that the error variances are all equal is rejected at the significance level, this assumption for consistency is in use. At least one coefficient is heteroskedastic. More importnatly, the causal model is contemporaneously exogenous, a weaker version of strict exogenity. In other words, the serial correlation may be existed since the <i>Provider Information</i> dataset is cross-sectional.</p>

<div align = 'center'><img src="https://latex.codecogs.com/svg.image?y_i&space;=&space;\phi&space;(\beta_0&space;&plus;&space;\beta_1&space;bed_i&space;&plus;&space;\beta_2&space;hr_i&space;&plus;&space;cond_i&space;\beta)&space;&plus;&space;u_i" title="y_i = \phi (\beta_0 + \beta_1 bed_i + \beta_2 hr_i + cond_i \beta) + u_i" /></div>

<p align = 'justify'>The causal model contains &#934;(.) that is Probit function. The <i>cond<sub>i</sub>&#946;</i> is a vector of conditions that include the level of quality care and competency the staff provides, type of location area and kind of environment residents live in, and kind of vulnerability residents have.</p>

## ACTIONABLE INSIGHTS <a id='insights'></a>

<p align = 'justify'> When the causal relationship is established, each coefficient of exogenous variables is statistically different from zero at 5% using Huber-White (HCO covariance type) robust standard errors. The pseudo-R-squared score is 32.54% while the p-value for LLR is below 5%. The null hypothesis for LLR is that the fit of the intercept-only model and causal model are equal is rejected. Every coefficient of exogenous variables is an important contributor to this model except for intercept. The p-value for intercept is 0.881.</p>

![image](https://user-images.githubusercontent.com/52142917/137193687-e4111f44-9e3d-4820-aefe-a1a469574a6a.png)

<p align = 'justify'> When all variables are being isolated and the causal model is specified, the plot shows that the registered and license practical nurses have significant effects on health inspection rating as expected. Interestingly, with the nurse aides being present in facilities, their positive effect on health inspection rating is relatively less. When the interaction term is applied, there are a few interesting results that are important to mention.</p>

- <p align = 'justify'>When the <i>h<sub>i</sub></i> interacts with <i>nur_aid<sub>i</sub></i>, the increase in number of hours that the nurse aides spend with residents each day has little to no impact on health inspection rating.</p>
- <p align = 'justify'>The registered nurses that interacts with <i>h<sub>i</sub></i> have a positive impact on this rating at the significance level.</p>
- <p align = 'justify'>More surprisingly, when the <i>h<sub>i</sub></i> interacts with <i>lpn<sub>i</sub>, the coefficient is -0.0881 with margin of error equal 0.07, and that is, the incremental increase in the number of hours that license practical nurses spend has a negative impact on this rating at the significance level.</p>


