# Analyzing Phishing URL data for detecting potential phishing attacks using various classification algorithms

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Selection](#model-selection)
- [SUMMARY OF RESULTS](#summary-of-results)


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

![Confusion Matrix XGboost classifier](/plots/xgcm.png)

The data was pre-processed and split into train and validation datasets of 80% and 20% respectively. The data has no missing values. The data was checked for duplication and various data types were checked and changed to fit the model. One hot encoding was done for the categorical variables. A feature importance was used to check the variables having more predictive power on our target variable. confusion matrix is also plotted to get detailed view of the classification model's performance.All parameters were left at default to serve as a benchmark to evaluate other models and algorithms. XGBoost builds an ensemble of decision trees sequentially.

The xgboosting classifer had 100% accuracy. The confusion matrix above shows a total count of 18246 indicating that the model was able able predict all of the positive class, that URL that are legitgimate. It also correctly predicted a total count 24198 URL that that are threat to an organization. The model did not incorrectly predict any legitimate URL nor Phishing URL. This makes xgboosting a best for model selection in our subsequent analysis.

![Feature Selection XGboost classifier](/plots/fixg.png)

The graph above shows the most important variables selected in the predictive process of xgboosting. In order of importance it can be seen that URLSimilarityIndex had the the most predictive power followed by LineOfCode and then IsHTTPS. This can furthur be understood by the default plot importance function that comes with the xgboosting classifier. This is explain below

The plot importance function that comes with the xgboosting classifier is used to check some of the variables that has more predictive power on the target varible based on three different metric. That's gain,cover and weight metric.

![Feature Selection XGboost classifier](/plots/xgbclas.png)

The graph above is a feature importance based on the weights. Weight represents the number of times a feature is used to split the data across all trees in the model.Features with higher weights are considered more important because they are involved in more splits, thereby having a greater impact on the model's predictions. It can be seen that out of the 51 features 17 were involved the prediction process, however, URLSimilarityIndex,LineOfCode and IsHTTPS have higher weights meaning they were involved in the prediction process.

![Feature Selection XGboost classifier](/plots/xgbclas1.png)

The graph above is a feature importance based on the Gain.Gain is the average gain of splits which use the feature. In other words, it represents the average improvement in the loss function (objective) brought by a feature when it is used to split the data at a decision tree node. Gain reflects the contribution of a feature to the model’s predictive power. Higher gain values indicate that a feature has a higher impact on the model’s predictions.Here too it can be seen that the features URLSimilarityIndex,LineOfCode and IsHTTPS have higher gains using the default plot importance of the xgboosting classifier.


![Feature Selection XGboost classifier](/plots/xgbclas2.png)

The graph above is a feature importance based on the cover.Cover measures the relative quantity of observations related to each feature. Specifically, it is the average coverage (number of samples) of the splits which use the feature.Cover indicates how often and extensively a feature is used in the model. A feature with high cover means it is used in many splits and thus plays a significant role in the model’s structure.Once again URLSimilarityIndex,LineOfCode and IsHTTPS features played an important role in the model's structure.

It can be seen that when you look at all the three in-built feature plot importance , out of the 51 features 17 were more invovled in the structure and spliting process of the xgboosting model. Howver URLSimilarityIndex,LineOfCode and IsHTTPS played more important role in the predictive process.

### Support Vector Machine Classifier

A Support Vector Machine (SVM) classifier is a supervised machine learning algorithm that analyzes data for classification and regression analysis. Its primary objective is to find the optimal hyperplane (decision boundary) in an N-dimensional space that distinctly classifies data points into different categories. this makes a best model for our classification problem.

![confusion matrix of support vector machine classifier](/plots/svmc.png)

A non-linear support vector machine classifier is used with a rbf kernel.The SVM classifier with an RBF kernel involves initializing the classifier, we train the model training data, making predictions on our validation data, and evaluating its performance using the confusion matrix, accuracy score, etc. The rbf kernel is used because it provides a robust framework for tackling non-linear classification tasks. It is suitable when there is no prior knowledge about the data distribution or when the decision boundary is highly irregular (non-linear). It implicitly maps the data into a higher-dimensional space.

The confusion matrix above shows the accuracy and inaccuracy score of how predictive the non-linear SVM classifier predicts our target variable. The model was able to correctly predict 17696 legitmate URL and also correctly predict 20089 cyber threats phishing URL. The model incorrectly predicted 4109 legitimate URL instead of cyber threat URL and also wrongly predicted 550 cyber threats URL instead of a legitimate URL. The svm classifier had a prediction accuracy score of 89%. This makes the model less attractive for model selection in our subsequent analysis.

### Decision Tree Classifier

A decision tree classifier for binary response is a type of supervised machine learning model that predicts binary outcomes (i.e., outcomes that can take one of two possible values, typically represented as 0 and 1).The algorithm splits the data at each node based on a chosen feature and a splitting criterion. For binary classification, common splitting criteria include Gini impurity and entropy (information gain)

![confusion matrix of decision tree](/plots/treecm.png)

Here our training data is recursively partitioned into feature space based on the values of the features and default splitting criteria (e.g., Gini impurity, entropy).At each step, the algorithm selects the feature that best separates the data into more homogeneous groups (with respect to the target variable).By the end of the training process, the decision tree model is created, which consists of a series of nodes and branches that represent the learned decision rules derived from the training data.The validation data is used to evaluate the performance of our trained decision tree classifier. It serves as an independent dataset that the model hasn't seen during training, allowing you to assess how well the model generalizes to new, unseen data.For each instance in the test data, the decision tree model follows the learned decision rules to predict the corresponding class label (0 or 1 in the case of binary classification). This is then evaluated using a confusion matrix, accuracy score,etc.

The decision tree classifer had 100% accuracy. The confusion matrix above shows a total count of 18246 indicating that the model was able able predict all of the positive class, that URL that are legitgimate. It also correctly predicted a total count 24198 URL that that are threat to an organization. The model did not incorrectly predict any legitimate URL nor Phishing URL. This makes decision tree a best model for selecytion in our subsequent analysis.


![Feature Selection decision tree](/plots/treefeat.png)

The graph above shows the most important variables selected in the predictive process of decision tree classifier. In order of importance it can be seen that URLSimilarityIndex had the the most predictive power followed by LineOfCode .That is only 2 features had more predictive power in the prediction process out of all the 51 features

### Random Forest Classifier

A Random Forest classifier is an ensemble learning method that combines multiple decision trees to improve the performance and robustness of the model.It is a powerful and versatile machine learning model that leverages the strength of multiple decision trees to achieve robust and accurate predictions, making it a popular choice across various applications in machine learning.The classification aspect of random forest make a perfect model for our analysis.

![confusion matrix random forest](/plots/rfcm.png)

The random forest classifer had 100% accuracy. The confusion matrix above shows a total count of 18246 indicating that the model was able able predict all of the positive class, that URL that are legitgimate. It also correctly predicted a total count 24198 URL that that are threat to an organization. The model did not incorrectly predict any legitimate URL nor Phishing URL. This makes random forest a best model in the model selection for our subsequent analysis.

​![Feature Selection random forest](/plots/rffeat.png)

The graph above shows the most important variables selected in the predictive process of random forest classifier. In order of importance it can be seen that URLSimilarityIndex had the the most predictive power followed by NoOfExternalREf, then LineOfCode and it follows like that from the graph .It can be see that 28 features had more predictive power in the prediction process out of all the 51 features

## SUMMARY OF RESULTS

|Model                            | Accuracy | F1 Score  | Precision |
|---------------------------------|----------|-----------|-----------|
|XGboost                          | 1.00000  | 1.00000   | 1.00000   |
|Decision Tree                    | 1.00000  | 1.00000   | 1.00000   |
|Random Forest                    | 1.00000  | 1.00000   | 1.00000   |
|Regularized Logistic Regression  | 0.99609  | 0.99661   | 0.996529  |
|Support Vector Machine           | 0.89023  | 0.89075   | 0.90380   |


#### Bar Graph of Accuracy among  Different Models

![Bar plot of accuracy](/plots/accuracy.png)

This bar chart visualizes the accuracy of five different machine learning models:
Observations:
Random Forest, Decision Tree, and XGBoost:
These models achieve the highest accuracy of 1.0 (100%), meaning they perfectly classified the test data without errors.
Logistic Regression:
The accuracy is 0.996 (~99.6%), indicating it is highly accurate but not as perfect as the models above.
Support Vector Machine (SVM):
The accuracy is 0.89 (~89%), which is noticeably lower compared to the other models.








