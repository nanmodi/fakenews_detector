Naïve Bayes classifier

Average F1 Score for MultinomialNB (Validation Set): 0.5098892442157552
 



 ![image](https://github.com/user-attachments/assets/5d5929d1-0094-4064-a5ad-e789ff9133e3)



![image](https://github.com/user-attachments/assets/094ad010-3c44-4f8a-8af3-330b37fcffdf)

 
![image](https://github.com/user-attachments/assets/85ec2d9a-469a-48ad-93cd-faa7482d8037)



Logistic Regression

Average F1 Score for LogisticRegression (Validation Set): 0.5995710837717503

 ![image](https://github.com/user-attachments/assets/1e62b5e3-9efa-495a-a496-6bfc2b0d4dc8)


 ![image](https://github.com/user-attachments/assets/c93e29c6-dc5a-4564-a3cb-3e72ee9b1e17)



 ![image](https://github.com/user-attachments/assets/f6255f94-3093-4b36-8463-ede20aa3de9c)



![image](https://github.com/user-attachments/assets/cc6bdb77-d618-40a0-8a8e-8c45f08ab1bc)

 


Svm
Average F1 Score for LinearSVC (Validation Set): 0.6031450992886522
 

 ![image](https://github.com/user-attachments/assets/c6f9c5e3-1773-4238-a1c3-35989bf0e082)

![image](https://github.com/user-attachments/assets/63aa4fa9-b502-48b2-81a1-0c7070a30807)
![image](https://github.com/user-attachments/assets/70eba6a1-0880-498f-9a1c-ca574c97fdbc)
![image](https://github.com/user-attachments/assets/dc9e120b-c77c-4af5-8738-9585b950222c)

 


 


 


SGD classifier

Average F1 Score for SGDClassifier (Validation Set): 0.4076080143755946
 



 ![image](https://github.com/user-attachments/assets/296fd4e1-c84b-4c07-a853-d239dffc0c62)



 ![image](https://github.com/user-attachments/assets/ae78a719-f6e8-4694-8293-57086c97a590)

![image](https://github.com/user-attachments/assets/2f48be55-4f33-4d2c-a882-587ec5cc60e5)


![image](https://github.com/user-attachments/assets/04eade10-f6b7-4398-8e5f-719bcd193d7f)

Random Forest Classifier
Average F1 Score for RandomForestClassifier (Validation Set): 0.5997234073468678

 
 ![image](https://github.com/user-attachments/assets/d1ffbf45-487c-44f1-ae0e-2295835cf52a)

 ![image](https://github.com/user-attachments/assets/fc3d53f1-d033-4945-9cd0-d17a47fbea8f)

![image](https://github.com/user-attachments/assets/5a480f32-167d-4c2e-aee8-8930146685e2)



 
 
 
![image](https://github.com/user-attachments/assets/996095cf-7f9a-40c3-a736-1d04e1bc246d)




![image](https://github.com/user-attachments/assets/3543b17a-0952-4f61-9b87-fb48b3afd080)



 


 ![image](https://github.com/user-attachments/assets/6361fac0-21f7-45bc-b7b4-90247b952a14)




 


 ![image](https://github.com/user-attachments/assets/5da80daf-ad5f-44da-98ba-3f792c5f5195)



 


By looking at all four classifier the best result is found for svm,random forest and logistic regression
So we will find the best parameter for them then finally select the best model for our data









Best Random Forest Parameters: {'max_depth': 15, 'n_estimators': 100}
Best Score: 0.5662

Best Logistic Regression Parameters: {'C': 1.0, 'penalty': 'l2'}
Best Score: 0.6176

Best SVM Parameters: {'C': 0.1}
Best Score: 0.6176999999999999

Finding best n-gram range for SVM...


Final model: SVM
Best n-gram range: (1, 4)
Best parameters: {'C': 0.1}

Final Model Performance:
              precision    recall  f1-score   support

       False       0.65      0.38      0.48      1169
        True       0.61      0.83      0.70      1382

    accuracy                           0.62      2551
   macro avg       0.63      0.60      0.59      2551
weighted avg       0.63      0.62      0.60      2551


 
