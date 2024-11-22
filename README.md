# Analyzing Phishing URL data for detecting potential phishing attacks using various classification algorithms

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)


## Project Overview
This project aims to create and assess machine learning models for accurately distinguishing between Legitimate and Phishing URL. Through the utilization of different classification algorithms, the project endeavors to pinpoint the most efficient model(s) for this classification task. The primary objective is to bolster cybersecurity measures by furnishing a dependable tool for promptly detecting potential Phishing URL and features associated with it, consequently safeguarding websites from malicious activities.

## Data Description
The dataset is in two folds, the PhishingURL dataset and the test dataset. The Phishing dataset consists of 212216 rows and 53 columns. The test dataset consists of 23579 rows and 52 columns. The dataset contains detailed information related to the World Wide Web and URL’s. The data was extracted from UCI Machine Learning Repository Learning [Download here](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) with key details:

- The column names are 'obs', 'URLLength', 'DomainLength', 'IsDomainIP', 'TLD','URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 'NoOfSubDomain', 'HasObfus cation', 'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegit sInURL', 'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoO fOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'Ha sTitle', 'DomainTitleMatchScore', 'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive', 'NoOfU RLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit ', 'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay',  'Crypto', 'Has CopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS',  'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef', 'label'
  
- The observation column si a sequential ID, it may not influence the Phishing attack data and so it was removed for modelling purposes. The columns were either extracted from URL features or HTML features. Some of the URL features include:
  - `TLD`: Top Level Domain is the part of the domain name, such as .com or .edu
  - `IsDomainIP`: a URL using an IP address instead of a domain name
  - `NoOfSubDomain`: Subdomain is part of URL that appears before domain name

- Some of the HTML features include:
  - `NoOfJS`:  JavaScript is a programming language that can be embedded in HTML to create interactive webpage.
  - `NoOfSelfRef, NoOfEmptyRef, NoOfExternalRef`: Hyperlinks (href) are clickable links that allow users to navigate between webpages or navigate to external webpages

- `Class(label)`: Indicates whether the website or webpage is associated with a legitimate URL or a Phishing URL.  This is coded as 0 for Phishing URL and 1 for legitimate URL respectively

## Data Cleaning and Preparation
- The classification imbalance of the label variable is less severe and can be ignored
- There are no missing values in the training and testing dataset
-  In distributing the target variable (Label), the legitimate URL which is coded as 1 represents 57.2 percent while the phishing URL which coded as 0 represents 42.8 percent out of 212216 total counts of the PhishingURL dataset
- There were no duplicates in either the Phishing URL dataset or test dataset.
- There were only two categorical variables, and the remaining were all numerical variables.
- The two categorical variables were Label and TLD. The Label variable had only two categories while the TLD variable 681 categories
- The PhishingURL dataset is partitioned into training and validation dataset. The Proportion to be specified in the validataion dataset is 20% and the remaining 80% will be used for training dataset

## Exploratory Data Analysis
These processes are done to help us become familiar with our dataset, understanding its structure, variables, and the relationships between them. It also provides insights into the distribution and nature of data, which aids in identifying the types of data (categorical, numerical, etc.) and their distributions. 



