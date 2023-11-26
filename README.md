# ML_Models_Evaluator
This performs evaluation of machine learning models with ease and provides the visualization

# Pre-requisites 
  1. Libraries matplotlib, numpy and pandas
  2. Ensure to have the data splitted into X_train, y_train, X_test, y_test

# Application 

  # Import packages/libraries
  from lib import ml_models as mlm
  # Create an instance 
  evaluate=mlm.evaluator(x_train, y_train, x_test, y_test)

  # Evaluation of Random Forest Classifier
  iters = list(range(10,200,10))  # this can be tweaked individually, depending upon the model chosen  
  model, x, scores, precisions, recalls, f1_scores = evaluate.randomForestClassifer(iters)  
  evaluate.getClassifiersAccuracy(model, x, scores, precisions, recalls, f1_scores)  
  
  # Evaluation of KNN Classifier
  iters = list(range(1,50))  # this can be tweaked individually, depending upon the model chosen  
  model, x, scores, precisions, recalls, f1_scores = evaluate.knnClassifier(iters)  
  evaluate.getClassifiersAccuracy(model, x, scores, precisions, recalls, f1_scores)  

# Check for visualizations 
Check for the directory as below, we can find different graphs stored according to models.
  1. ..reports\ml_stats\




