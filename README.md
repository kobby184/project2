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

![Bar graph](/plots/labelsbar.png)

This is a graph showing the distribution of the target variable. Here the 0 and color blue represents the phishing URl cyber treats and the 1 and color orange represents the legitimate URL used on various websites. There are 121365 legitimate URL counts and 90851 phishing URL counts


![Pair Plot](/plots/pairplot.png)

We can explore the relationships between some of the numeric variables using a pair plot. If we also color the observations based on the target variable status, we can know how the feature is distributed based on the target variable. In this pair plot we are looking at the patterns between the target variable with respect to some selected features. Looking at the density plot along the diagonal, there are no features that cleanly separate the target variable. The scatter plot of URLLength and NoOfDigitsInURL shows some sign of a clear separation as compared to the other features.

![correlation heatmap](/plots/corrheat.png)

The table above shows correlation between the numerical variables. Correlation levels relate to the legend besides the graph. High Positive Correlation: Implies that as one feature rises, the other feature also tends to rise and vice versa. For instance, a strong positive correlation between URLLength and NoOfLettersInURL implies that when there is a higher lenght of URL there is a possibilty of the number of letters in the URL also being high. as well. High Negative Correlation: Suggests that as one feature increases, the other feature tends to decrease and vice versa. For example, a negative correlation between URLCharProb and DigitRatioInURL may indicate that lesser URLCharProb may have higher DigitRatioInURL and vice versa. Low or No Correlation: Indicates the absence of a linear relationship between the features, implying that they can independently contribute to the model without redundancy.

![Mean of features grouped by binary response](/plots/barplt.png)

Surprisingly, the average line of code used for Phishing URL is higher than the legitimate URL. However, when we look at the largest line length, the average legitimate URL is far more than that of Phishing URL. Cybercriminals may use encrypted text to hide the actual code from the user. A longer line of code may indicate code obfuscation.

## Model Selection

In this part of the analysis, we fit 5 models of classification to predict the class of cyber threats, that’s the Phishing URL events as function of all the other 51 predictors. For model building purposes the PhishingURL dataset is Partitioned into a training dataset and validation dataset. Since we are comparing multiple models or algorithms, the validation data helps us in selecting the best-performing model. It also provides a fair evaluation metric for comparing different models under the same conditions. This was partitioned into 80% for training and 20% for validation respectively. Each model is first trained on the training data set and predicted using the validation dataset.
To get the best model, the accuracy of each model is taken into consideration and a confusion matrix is also plotted to check the prediction accuracy of each model. The F-score and Precision were used; however, our main evaluation metric was focused on the accuracy score and the confusion matrix. The variable of importance is also highlighted in each model process.
After comparing the accuracy score of each model, the best model(s) is selected. We refit the best model(s) by pooling the partitioned data together as our new training dataset and predict using our test dataset. The predicted target variable in the form of 0 or 1 is then store as Y_hat and associated predicted probabilities P_hat, with respect to the 0, that’s the phishing URL is computed and saved in CSV file.

### REGULARIZE LOGISTIC REGRESSION
Is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X. This means this algorithm is a perfect model for our analysis

![Confusion matrix of regularized logistic regression](/plots/regconfusion.png)

A logistic regression with L2 regularization penalty is used, a parameter grid for the regularization strength 'C' is defined. I also use a grid search cross validation to find the best value of the regularization strength. The best model is evaluated on the validation dataset. The coefficient of the best logistic regression model is extracted and is used to measure the feature importance and is plotted for visualization. A confusion matrix is also plotted to summarize and display the number of accurate and inaccurate instances based on our model’s prediction. L2 regularization is used because is less prone to overfitting.
The confusion matrix of the regularized logistic regression above had a true positive of 18162.This means the model was able to find a total count of 18162 having both predicted and actual values of legitimate URL, which is not a cyber threat. The model was able to correctly predict 24116 counts of threat that were Phishing URL. It incorrectly predicted 82 legitimate URL instead of a Phishing URL and incorrectly predicted 84 Phishing URL instead legitimate URL. The model had 99.6% accuracy score; this makes it almost near perfect for model selection.

![Feature Importance regularized logistic regression](/plots/featureimportan.png)

This graph is the variable importance selection for the regularized logistic regression. It shows some variables have positive impact on the response variable. It can be seen that features like URLLength,NoOfLettersInURL, DomainLength,NoOfSelfRef ,etc contribute most to the models prediction process. Features like IsDomainIp,Bank,URLCharProb ,etc have less contribution to the model's prediction process. Out of the 51 features 41 variables had an impact in the prediction process

### XGBOOSTING CLASSIFIER

XGBoost, or eXtreme Gradient Boosting, is a machine learning algorithm that's used to train and test models on large amounts of data. It's a popular choice for data scientists because it's open-source, free to use, and has many features that make it efficient and flexible. XGBoost is designed for speed and can outperform other models, such as logistic regression. It can be used in solving classification probelems. This makes it a perfect algorithm for our model selection analysis





