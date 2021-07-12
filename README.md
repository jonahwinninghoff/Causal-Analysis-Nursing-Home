# Nursing Home Capstone Project

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**[Overview](#overview)** | **[Table of Models](#models)**


## OVERVIEW <a id='overview'></a>

There is clear evidence that the nursing home residents are particularly vulnerable to abandonment, neglect, exploitation, and abuse. Given that the COVID-19 pandemic holds the key metrics to reveal some answers, the rating of nursing-home services proves to be misleading. The technological challenge is to find more appropriate approaches supporting the decision-makers who want to send elders to nursing homes. This is the purpose of the data science project.

# Table of Models with 10-fold Average Scores <a id='models'></a>

<table>
	<tr>
		<th>Model</th>
		<th>Parameters</th>
		<th>Metrics</th>
		<th>Comment</th>
 	</tr>
 	<tr>
  		<td>Logistic Regression</td>
   		<td>Target: Binary Score, Independent: TFIDVectorizer(stop_words = 'english', max_df = 0.25, min_df = 5) </td>
		<td>
			- <b>Accuracy: </b>67.9%
			- <b>Precision: </b>31.9%
			- <b>Mean Absolute Error: </b>0.321
			- <b>Brier Score: </b>0.321
		</td>
		<td>Brier Score needs to be below 0.25</td>
 	</tr>
	<tr>
		<td>Logistic Regression</td>
		<td>Target: Binary Score, Independent: TFIDVectorizer(stop_words='english')</td>
		<td>
			- <b>Accuracy: </b>67.9%
			- <b>Precision: </b>31.9%
			- <b>Mean Absolute Error: </b>0.321
			- <b>Brier Score: </b>0.321
		</td>
		<td>Given that the result is identical, the logistic regression model is not working.</td>
	</tr>
</table>
			
