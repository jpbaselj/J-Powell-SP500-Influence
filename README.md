# NLP of Fed. Reserve Speeches To Predict Stock Market Trend
Springboard Data Science Course Capstone 2 Project

Author: John (JP) Baselj

## Introduction

This project harnesses Natural Language Processing (NLP) to analyze public speeches on federal monetary policy and predict stock market trends.

The global economy continues to grapple with the impacts of the Covid-19 pandemic, with the US Federal Reserve largely responsible for shaping economic stability. Federal Reserve Chairman Jerome Powell periodically makes public statements on monetary policy, including federal interest rates, which "The Fed" ultimately controls. Since the inception of Covid-19 in the US, these speeches of federal monetary policy have incited pronounced volatility and trading activity in the stock market.

The objective is to develop a model that, on average, would yield profitable results when applied to invest in S&P500 index or reverse index funds based on predicted market direction. This report outlines the project's methodology and findings covering data wrangling, Exploratory Data Analysis (EDA), feature engineering, and machine learning modeling.

## Dataset
This analysis is based on two key datasets: speech transcripts from Chair Jerome Powell and daily S&P500 stock market data. The speech transcripts, sourced from the Federal Reserve's website, were initially in PDF format and had to be converted to plain text files. This text was then cleaned by preprocessing methods, including converting all letters to lowercase, removing punctuation and common English stop words, and tokenizing the text.

The S&P500 historical records data were sourced from investing.com and required light preprocessing, including the removal of empty fields and conversion of numeric values to suitable data types. Fields representing the Maximum and Minimum price for each trading day were used to create a new field, daily span, which serves as a proxy for daily market volatility.

Additionally, the stock market data was aggregated on a weekly basis to create features that represent the market value, volatility, and returns for any 7-day span. The S&P data was then merged with the data from Chair Powell's speeches so the market features became linked to the weeks preceding and following each speech.

## EDA and Feature Engineering
The analysis aimed to understand how the features of the speech text interact with the features of the stock market data. Sentiment analysis was performed on Chair Jerome Powell's speeches using the VADER module from Python's Natural Language Toolkit (NLTK) suite. The rolling stock market features surrounding each speech were visualized to identify trends, and relevant features for modeling were identified. A binary target variable was created to represent whether the S&P500 index increased in value from the week before to the week after a given speech.

Speech-text data was vectorized into a format compatible with standard Machine Learning models using TF-IDF and Bag of Words vectorization techniques. The datasets were oversampled to create balanced-class training sets.

## Modeling
Logistic Regression, Random Forest Classifier, and Support Vector Machine models were trained and evaluated using cross-validation for different data input combinations. The Random Forest Classifier model was selected as the final model after hyperparameter tuning. The final model achieved an accuracy of about 72% when evaluated using the holdout validation dataset.

A simulation was performed to model how applying a predictor with 72% accuracy may behave for predicting market moves following a speech.

## Conclusion
This project demonstrates the feasibility of applying Natural Language Processing techniques to relevant public speeches to predict future stock market trends. The limitations of this analysis include the relatively small dataset used for model training with only 25 historic public speeches being considered. In a future analysis, it may be beneficial to build upon these results by including larger datasets.

It is recommended to use the re-trained Random Forest model on 100% of the available data for future market predictions, iteratively re-training to include all historical speeches for each prediction moving forward. Continued refinements and expanded data sources hold promise for further enhancing the model's predictive capabilities. This project offers a framework for future investigations in this domain.

#### References
Federal Reserve (speeches): https://www.federalreserve.gov/newsevents/speeches.htm
Stock Market Data Source: https://www.investing.com/indices/us-spx-500-historical-data
