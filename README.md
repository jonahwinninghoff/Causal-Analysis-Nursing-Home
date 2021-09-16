# Nursing Home Capstone Project

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**[Overview](#overview)** | **[Table of Models](#models)**


## OVERVIEW <a id='overview'></a>

There is clear evidence that the nursing home residents are particularly vulnerable to abandonment, neglect, exploitation, and abuse. Given that the COVID-19 pandemic holds the key metrics to reveal some answers, the rating of nursing-home services proves to be misleading. The technological challenge is to find more appropriate approaches supporting the decision-makers who want to send elders to nursing homes. This is the purpose of the data science project.

# Table of Tested Models <a id='models'></a>

<table>
	<tr>
		<th>Model</th>
		<th>Parameters</th>
		<th>Hyperparameters</th>
		<th>Validation Set Metrics</th>
 	</tr>
 	<tr>
  		<td>Linear Regression: OLS</td>
   		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), Independent: TFIDVectorizer(stop_words = 'english', max_df = 0.8, min_df = 2)</td>
		<td></td>
		<td>
			<ul>
				<li><b>R2: </b>90.91%</li>
				<li><b>Adj R2: </b>90.91%</li>
				<li><b>MAE: </b>0.0687</li>
				<li><b>RMSE: </b>0.1108</li>
			</ul>
		</td>
 	</tr>
	<tr>
		<td>Linear Regression: OLS</td>
		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), Independent: 18 principal components</td>
		<td></td>
		<td>
			<ul>
				<li><b>R2: </b>69.28%</li>
				<li><b>Adj R2: </b>69.28%</li>
				<li><b>MAE: </b>0.1531</li>
				<li><b>RMSE: </b>0.2036</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Lasso Regression with Bayesian Optimization</td>
		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), 		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), Independent: TFIDVectorizer(stop_words = 'english', max_df = 0.8, min_df = 2)</td>
</td>
		<td>
			<ul>
				<li><b>Lambda Lasso: </b>0.01</li>
			</ul>
		</td>
		<td>
			<ul>
				<li><b>R2: </b>74.64%</li>
				<li><b>Adj R2: </b>74.63%</li>
				<li><b>MAE: </b>0.1552</li>
				<li><b>RMSE: </b>0.1849</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Lasso Regression with Bayesian Optimization</td>
		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), 		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), Independent: 18 principal components</td>
</td>
		<td>
			<ul>
				<li><b>Lambda Lasso: </b>0.01</li>
			</ul>
		</td>
		<td>
			<ul>
				<li><b>R2: </b>79.26%</li>
				<li><b>Adj R2: </b>79.25%</li>
				<li><b>MAE: </b>0.1329</li>
				<li><b>RMSE: </b>0.1673</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Random Forest Regression</td>
		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), Independent: TFIDVectorizer(stop_words = 'english', max_df = 0.8, min_df = 2)</td>
		<td>
			<ul>
				<li><b>Criterion: </b>MSE</li>
				<li><b>Number of Trees: </b>100</li>
				<li><b>Minimum Samples Split: </b>2</li>
				<li><b>Maximum Features: </b>5</li>
			</ul>
		</td>
		<td>
			<ul>
				<li><b>R2: </b>90.91%</li>
				<li><b>Adj R2: </b>90.91%</li>
				<li><b>MAE: </b>0.0687</li>
				<li><b>RMSE: </b>0.1108</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Random Forest Regression</td>
		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), Independent: 18 principal components</td>
		<td>
			<ul>
				<li><b>Criterion: </b>MSE</li>
				<li><b>Number of Trees: </b>100</li>
				<li><b>Minimum Samples Split: </b>2</li>
				<li><b>Maximum Features: </b>5</li>
			</ul>
		</td>
		<td>
			<ul>
				<li><b>R2: </b>90.91%</li>
				<li><b>Adj R2: </b>90.91%</li>
				<li><b>MAE: </b>0.0687</li>
				<li><b>RMSE: </b>0.1108</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Light Gradient Boosting Model with Bayesian Optimization</td>
		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), Independent: TFIDVectorizer(stop_words = 'english', max_df = 0.8, min_df = 2)</td>
		<td>
			<ul>
				<li><b>Lambda Lasso: </b>0.03</li>
				<li><b>Lambda Ridge: </b>0.05</li>
				<li><b>Learning Rate: </b>0.37</li>
				<li><b>Number of Iterative Trees: </b>15</li>
			</ul>
		</td>
		<td>
			<ul>
				<li><b>R2: </b>90.91%</li>
				<li><b>Adj R2: </b>90.91%</li>
				<li><b>MAE: </b>0.0687</li>
				<li><b>RMSE: </b>0.1108</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Light Gradient Boosting Model with Bayesian Optimization</td>
		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), 		<td>Continuous Target: Home Nurse Rating 100%-0% (1-0), Independent: Independent: 18 principal components</td>
</td>
		<td>
			<ul>
				<li><b>Lambda Lasso: </b>0.15</li>
				<li><b>Lambda Ridge: </b>0.34</li>
				<li><b>Learning Rate: </b>0.18</li>
				<li><b>Max Depth: </b>5</li>
				<li><b>Number of Iterative Trees: </b>39</li>
			</ul>
		</td>
		<td>
			<ul>
				<li><b>R2: </b>90.91%</li>
				<li><b>Adj R2: </b>90.91%</li>
				<li><b>MAE: </b>0.0687</li>
				<li><b>RMSE: </b>0.1108</li>
			</ul>
		</td>
	</tr>
</table>


