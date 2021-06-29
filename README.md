# Nursing Home Capstone Project

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/jonahwinninghoff/Springboard/graphs/commit-activity)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**[Overview](#overview)** | **[Problem Statement](#problem)** | **[Context](#context)** | **[Criteria for Success](#criteria)** | **[Scope of Solution Space](#scope)** | **[Constraints](#constraints)** | **[Potential Stakeholders](#stakeholders)** | **[Data Sources](#sources)**


### OVERVIEW <a id='overview'></a>

There is clear evidence that the nursing home residents are particularly vulnerable to abandonment, neglect, exploitation, and abuse. Given that the COVID-19 pandemic holds the key metrics to reveal some answers, the rating of nursing-home services proves to be misleading. The technological challenge is to find more appropriate approaches supporting the decision-makers who want to send elders to nursing homes. This is the purpose of my project proposal.

Ensuring that my project proposal is deliverable, the discussion is in extent of  problem identification. The problem identification comprises seven sections, as follow: problem statement formation, context, criteria for success, constraints, stakeholders, scope of solution space, and data sources. When the identification component is established, the discussion about a few key questions will be next.

### PROBLEM STATEMENT <a id='problem'></a>

Given that the nursing home industry in the western region relies on the trust of consumers as a public good, how can the percentage of rating system inconsistent with quality of nursing services be identified as a baseline that may be in use to evaluate the rating system? What is an alternative measurement that its accuracy is 10% higher than this baseline before this metric system is readily available on August 1, 2021?

## CONTEXT <a id='context'></a>

Using a simple star rating system as a key driver is—on which the patients are being sent to nursing homes—for the consumers to make decisions. This system has been in use since 2009 and it is now a popular way to educate themselves. On March 13, 2021, more than 130,000 nursing-home residents died of Covid-19, which accounts for—roughly—25% of total deaths in the United States, which is historically tragic (the number of deaths remains underestimated). Equally importantly, this pandemic does offer key insights that much information about this rating system itself is inaccurate. For example, the number of Covid-19 deaths at five-star facilities are not significantly different from one-star facilities. The U.S. Centers for Medicare & Medicaid Services (CMS) relies on three grades for this rating system, as follows: 

- Performance during the state health inspections
- Amount of time nurses spend with residents
- Quality of care that residents receive

Enormous though the CMS database is, it is dependent on a mixture of on-site examinations and self-reported data ([Silver-greenberg and Gebeloff, 2021](#reference)). The self-reported data is not immune to moral hazard—referring to behavior deviating from cooperative agreement. Department of Health and Human Services responsible for fraud monitoring is unable to preclude this moral hazard. The problem is that the incentive to falsify the report may be due to business perception of tradeoff between cost of cooperation and that of deviation. In other words, the nursing-home industry perceives the fraud, rather than as a punishment, as an opportunity cost—to great lengths they are willing to take to achieve.


## CRITERIA FOR SUCCESS <a id='criteria'></a>

The relevant metrics for successes are:

- The analysis should be able to separate self-report data from on-site examinations and the key metrics on self-report data with the tendency to inflate or deflate should also be identified.
-	The percentage of rating system inconsistent with the quality of nursing home services should be established as a baseline and the accuracy of alternative measurement should be 10% more than the current rating system.
-	So as for understanding the incentive mechanism, this analysis should incorporate economic reasoning, modeling, and econometrics.
-	The user-friendly interface with reproducible research and presentation should be readily available on or before August 1, 2021.
-	The analysis should convince stakeholders and their decision-making should be based on the analysis. 

## SCOPE OF SOLUTION SPACE <a id = 'scope'></a>

The data analytics focuses on three different objectives, as follow: identifying the percentage of discrepancy in the rating system as a baseline, incorporating an economic understanding of incentive mechanism, and creating an alternative measurement that is 10% accurate more than its baseline. Two components needed for this solution space are CMS relational databases and external datasets containing at least two records that are believed to associate with the same entity.

## CONSTRAINTS <a id = 'constraints'></a>

-	Based on New York Times’ cross-examination, most information submitted to CMS is incorrect due to self-report data and the incidents are often unreported.
-	The on-site examination may be at risk for observer bias (for example, the inspection report classifies the sexual assault as a “category F” violation, a low-level problem).
-	The on-site examination may also be at risk for adverse selection (for example, nursing-home employers have more information than consumers, they conceal this information from them and health inspectors, and they are able to anticipate when the inspectors pay visits).

## POTENTIAL STAKEHOLDERS <a id = 'stakeholders'></a>

-	Chiquita Brooks-LaSure (CMS Administrator)
- Jonathan Blum (CMS Principal Deputy Administrator)
-	Erin Richardson (CMS Chief of Staff)
-	Ms. Harrington (professor emeritus at the University of California)

## DATA SOURCES <a id = 'sources'></a>

- [Quality Assessment](https://www.cms.gov/Medicare/Quality-Initiatives-Patient-Assessment-Instruments/NursingHomeQualityInits/Staffing-Data-Submission-PBJ)
- [Payment Data](https://data.cms.gov/browse?q=daily+nurse+staffing)
- [MDS Quality Measures Data](https://data.cms.gov/provider-data/dataset/djen-97ju)
- [Penalties Data](https://data.cms.gov/provider-data/dataset/g6vv-u9sr)
- [Documentation](https://data.cms.gov/Special-Programs-Initiatives-Long-Term-Care-Facili/PBJ-Public-Use-Files-Data-Documentation/ygny-gzks)

## REFERENCES <a id = "reference"></a>

[Silver-greenberg, Jessica, and Robert Gebeloff. “Maggots, Rape and Yet Five Stars: How U.S. Ratings of Nursing Homes Mislead the Public.” How U.S. Ratings of Nursing Homes Mislead the Public, The New York Times, 13 Mar. 2021.](https://www.nytimes.com/2021/03/13/business/nursing-homes-ratings-medicare-covid.html) <a id = 'silver-greenberg'></a>
