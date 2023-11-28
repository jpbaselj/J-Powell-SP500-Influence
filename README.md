# NLP of Fed. Reserve Speeches To Predict Stock Market Trend
Springboard Data Science Course Capstone Project

![Federal Reserve Logo](https://github.com/jpbaselj/J-Powell-SP500-Influence/blob/main/documentation/Flag_of_the_United_States_Federal_Reserve.svg.png)

## Introduction

This project harnesses Natural Language Processing (NLP) to analyze public speeches on federal monetary policy and predict stock market trends.

The US Federal Reserve is largely responsible for shaping economic stability in the wake of the Covid-19 global economic disruption. Federal Reserve Chairman Jerome Powell periodically makes public statements on monetary policy, including federal interest rates, which "The Fed" ultimately controls. Since the inception of Covid-19 in the US, these speeches of federal monetary policy have incited pronounced volatility and trading activity in the stock market.

The project objective is to develop an NLP model that analyzes speeches from The Fed and, on average, would yield profitable results when applied to guide investment in S&P500 index or reverse index funds based on predicted market direction. 

## Data Wrangling
[Data Wrangling Notebook](https://github.com/jpbaselj/J-Powell-SP500-Influence/blob/main/3_1_data_wrangling.ipynb)

This analysis is based on two key datasets: speech transcripts from Chair Jerome Powell and daily S&P500 stock market data. The speech transcripts, sourced from [The Federal Reserve's website](https://www.federalreserve.gov/newsevents/speeches.htm), were initially in PDF format and were converted to plain text files. This text was then cleaned by preprocessing methods, including converting all letters to lowercase, removing punctuation and common English stop words, and tokenizing the text.

The S&P500 historical records data were sourced from [investing.com](https://www.investing.com/indices/us-spx-500-historical-data) and required light preprocessing, including the removal of empty fields and conversion of numeric values to suitable data types. Fields representing the Maximum and Minimum price for each trading day were used to create a new field, daily span, which serves as a proxy for daily market volatility.

Additionally, the stock market data was aggregated on a weekly basis to create features that represent the mean market value, volatility, and returns for any 7-day span. The S&P data was then merged with the data from Chair Powell's speeches so the market features became linked to the weeks preceding and following each speech.

## EDA and Feature Engineering
[EDA and Feature Engineering Notebook](https://github.com/jpbaselj/J-Powell-SP500-Influence/blob/main/3_2_EDA_Feature_Engineering.ipynb)

The analysis aimed to understand how the features of the speech text interact with the features of the stock market data. Sentiment analysis was performed on speeches using the VADER module from Python's Natural Language Toolkit (NLTK) suite. The rolling stock market features surrounding each speech were visualized to identify trends, and relevant features for modeling were identified. A binary target variable was created to represent whether the S&P500 index increased in value from the week before to the week after a given speech.

![Target Variable Over Time](https://github.com/jpbaselj/J-Powell-SP500-Influence/blob/main/documentation/target_v_over_time.png)

Speech-text data was vectorized into a format compatible with standard Machine Learning models using both TF-IDF and Bag of Words vectorization techniques. The datasets were oversampled to create balanced-class training sets, while also preserving the natural/imbalanced class sets. Each of the 4 data set combinations of balanced OR imbalanced classes and TF-IDF OR BoW vectorization were preserved for model training and performance comparison.

## Modeling
[Modeling Notebook](https://github.com/jpbaselj/J-Powell-SP500-Influence/blob/main/3_3_processing_modeling.ipynb)

Logistic Regression, Random Forest Classifier, and Support Vector Machine models were trained and evaluated using cross-validation for each different data input combination. The Random Forest Classifier model was selected as the final model after hyperparameter tuning. Due to the small data set size, model performance was then evaluated using Cross Validation (CV) techniques. The tuned RF model achieved a mean CV accuracy of about 72% when evaluated using the holdout validation dataset.

#### 

A simulation was performed to model how applying a predictor with 72% accuracy may behave for predicting market moves following a speech. The result of this simulation can be visualized in the following plots:
1) A modified psuedo random walk representing the returns of a model with 72% accuracy on predicting market moves across historical speeches given a starting value of 100USD
![one walk eg](https://github.com/jpbaselj/J-Powell-SP500-Influence/blob/main/documentation/sim_walking_eg.png)
  
2) The distribution of the resulting market return of simulating the psuedo random walk on historical data 1000 times
![simulation distribution](https://github.com/jpbaselj/J-Powell-SP500-Influence/blob/main/documentation/sim_spread_outcomes.png)



## Conclusion
This project demonstrates the feasibility of applying Natural Language Processing techniques to relevant public speeches to predict future stock market trends. The limitations of this analysis include the relatively small dataset used for model training with only 25 historic public speeches being considered. In a future analysis, it may be beneficial to build upon these results by including larger datasets.

It is recommended to re-train the Random Forest model on 100% of the available data for future market predictions, iteratively re-training to include all historical speeches for each prediction moving forward. Continued refinements and expanded data sources hold promise for further enhancing the model's predictive capabilities. This project offers a framework for future investigations in this domain.

#### Project Report
[Project Report](https://github.com/jpbaselj/J-Powell-SP500-Influence/blob/main/Reports/FINAL_REPORT.pdf)

#### References
Federal Reserve (speeches): https://www.federalreserve.gov/newsevents/speeches.htm

Stock Market Data Source: https://www.investing.com/indices/us-spx-500-historical-data
