# Machine Learning Assignment 2

## Problem Statement

The objective is to develop a multiclass classification model that predicts the fuel type of a vehicle based on its characteristics and market-related attributes.
The assignment is to build and deploy classification models using a public dataset and evaluate their performance using standard machine learning metrics.

## Dataset Description

Automobile Market Analytics Dataset (Kaggle Repository https://www.kaggle.com/datasets/deeplumiere/automobile-market-analytics-dataset )

The Automobile Market Analytics dataset contains 5,500 vehicle records described by 18 attributes, including vehicle specifications, ownership history, fuel efficiency, and selling price. 
The target variable is Fuel_Type, which consists of four classes: Petrol, Diesel, Hybrid, and Electric. 
No duplicate records are present in the dataset. 
The class distribution is imbalanced, with Petrol vehicles forming the majority class. 
This dataset is suitable for multiclass classification tasks involving fuel type prediction.


Number of Instances (Rows): 5,500
Number of Features (Columns): 18
Target Variable: Fuel_Type
Number of Classes: 4
Duplicate Records: 0

Fuel_type class distribution:

Fuel Type |	Count
Petrol	  | 4,267
Diesel	  | 551
Hybrid	  | 537
Electric  |	145


## Models Used

1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest

## Evaluation Metrics


================================ MODEL RESULTS ======================================
                 Model  Accuracy       AUC  Precision    Recall        F1       MCC
1  Logistic Regression  0.857273  0.825405   0.771727  0.857273  0.809996  0.574341
2        Decision Tree  0.761818  0.754554   0.771581  0.761818  0.766603  0.383693
3                  KNN  0.791818  0.717330   0.717246  0.791818  0.734890  0.296441
4          Naive Bayes  0.287273  0.649902   0.683922  0.287273  0.365473  0.076881
5        Random Forest  0.855455  0.810317   0.773718  0.855455  0.806094  0.568407
=====================================================================================

## Observations:

Model Name                          Observation about model performance 
=====================================================================================================================================================================================
Logistic Regression                   Achieved the highest accuracy (85.73%) among all models.Highest AUC (0.825) indicates strong class discrimination ability.
									  Best F1-Score (0.810) showing a good balance between precision and recall.
									  Highest MCC (0.574) suggests reliable predictions even under class imbalance.
								      Indicates that the relationship between vehicle attributes and fuel type is largely captured through linear decision boundaries.
									  Conclusion: Logistic Regression is the most suitable model for this dataset.

Random Forest     					  Accuracy is only 0.18% lower than Logistic Regression.
		                              Produced the highest precision (0.774) among all models.
									  Strong F1-score and MCC indicate stable performance.
									  Able to model complex feature interactions and nonlinear relationships.
									  Slightly lower AUC and MCC than Logistic Regression.
				                      Conclusion: Random Forest is a strong alternative and provides robust classification performance.

K-Nearest Neighbors                   Accuracy drops by approximately 6.5% compared to the best models.
                                      Lower MCC indicates weaker agreement between predictions and true labels.
                                      Sensitive to feature scaling and class imbalance.
                                      Performs reasonably well for the majority class but struggles with minority classes such as Electric and Hybrid
									  Conclusion: KNN provides acceptable performance but is less reliable than Logistic Regression and Random Forest.

Decision Tree  				          Lower accuracy than Logistic Regression and Random Forest.
                                      Precision remains relatively high.
                                      Lower MCC suggests reduced overall prediction reliability.
                                      Likely suffers from overfitting due to learning highly specific decision rules.
					                  Conclusion: Decision Tree offers interpretability but sacrifices predictive performance.

 Naïve Bayes                          Extremely poor accuracy.
                                      Very low recall indicates most fuel types are misclassified.
                                      MCC close to zero suggests predictions are only slightly better than random guessing.
                                      Assumption of feature independence does not hold for this dataset because several vehicle attributes are highly correlated (e.g., Engine Size, Horsepower, Torque, Fuel Efficiency).  
									  Conclusion: Naïve Bayes is not suitable for this automobile fuel-type classification problem.	
 
Overall Winner on this dataset?    The results indicate that Logistic Regression and Random Forest are the most effective models for predicting automobile fuel types. 
                                   Logistic Regression achieved the best overall performance across Accuracy, AUC, F1-Score, and MCC, while Random Forest produced comparable results with slightly better precision.
								   Therefore, Logistic Regression is recommended as the final deployment model for the Automobile Market Analytics Fuel-Type Classification system.
 


## GitHub Repository Link

https://github.com/dvrbandi/ML2/tree/main

## Streamlit Link

https://2025ac05934-dvr-bandi.streamlit.app/
