from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import numpy as np
import DataPrep
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
# Set the seed for reproducibility
np.random.seed(42)
# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



    # K-fold cross-validation
    
    
    # Plot the confusion matrix
    
# Define n-gram range, e.g., bigrams (2) and trigrams (3)
ngram_range = (1, 2)  # Unigrams and Bigrams

# Define TF-IDF vectorizer with n-grams
tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)

# Classifiers
nb_clf = MultinomialNB()
logR_clf = LogisticRegression()
svm_clf = svm.LinearSVC()
sgd_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=200)
rf_clf = RandomForestClassifier(n_estimators=50, n_jobs=3)

# Function for K-Fold Cross Validation using a validation set
def build_confusion_matrix_with_validation(clf, vectorizer):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])

    # K-fold cross-validation
    for train_ind, val_ind in k_fold.split(DataPrep.train_news):
        # Split the training data into train and validation sets
        train_text = DataPrep.train_news.iloc[train_ind]['Statement']
        train_y = DataPrep.train_news.iloc[train_ind]['Label']
        val_text = DataPrep.train_news.iloc[val_ind]['Statement']
        val_y = DataPrep.train_news.iloc[val_ind]['Label']
        
        # Apply the vectorizer and fit the classifier on training data
        X_train_tfidf = vectorizer.fit_transform(train_text)
        X_val_tfidf = vectorizer.transform(val_text)
        
        clf.fit(X_train_tfidf, train_y)
        predictions = clf.predict(X_val_tfidf)
        
        # Update confusion matrix and F1 score for validation set
        confusion += confusion_matrix(val_y, predictions)
        score = f1_score(val_y, predictions, average='weighted')
        scores.append(score)
    
    print(f"Average F1 Score for {clf.__class__.__name__} (Validation Set): {sum(scores) / len(scores)}")
    print(f"Confusion matrix for {clf.__class__.__name__}:")
    print(confusion)
    plot_confusion_matrix(confusion, f'{clf.__class__.__name__} Validation Set')
    
    # Plot F1 scores
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 6), scores, marker='o', label='F1 Score')
    plt.title(f'{clf.__class__.__name__} F1 Scores Across K-Folds')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()
    plt.show()

# Run K-Fold Cross Validation with validation set for all classifiers
print("Naive Bayes Classifier (with Validation Set)")
build_confusion_matrix_with_validation(nb_clf, tfidf_vectorizer)

print("\nLogistic Regression Classifier (with Validation Set)")
build_confusion_matrix_with_validation(logR_clf, tfidf_vectorizer)

print("\nLinear SVM Classifier (with Validation Set)")
build_confusion_matrix_with_validation(svm_clf, tfidf_vectorizer)

print("\nSGD Classifier (with Validation Set)")
build_confusion_matrix_with_validation(sgd_clf, tfidf_vectorizer)

print("\nRandom Forest Classifier (with Validation Set)")
build_confusion_matrix_with_validation(rf_clf, tfidf_vectorizer)

# Training classifiers on the full training data and testing on the test set
def evaluate_on_test_data(clf, vectorizer):
    # Fit the model on the entire training data
    X_train_tfidf = vectorizer.fit_transform(DataPrep.train_news['Statement'])
    y_train = DataPrep.train_news['Label']
    
    clf.fit(X_train_tfidf, y_train)
    
    # Predict on the test set
    X_test_tfidf = vectorizer.transform(DataPrep.test_news['Statement'])
    predictions = clf.predict(X_test_tfidf)
    
    # Print classification report
    print(f"Classification Report for {clf.__class__.__name__}:")
    print(classification_report(DataPrep.test_news['Label'], predictions))

# Evaluate all classifiers on the test data
print("\nNaive Bayes Classifier on Test Data")
evaluate_on_test_data(nb_clf, tfidf_vectorizer)

print("\nLogistic Regression Classifier on Test Data")
evaluate_on_test_data(logR_clf, tfidf_vectorizer)

print("\nLinear SVM Classifier on Test Data")
evaluate_on_test_data(svm_clf, tfidf_vectorizer)

print("\nSGD Classifier on Test Data")
evaluate_on_test_data(sgd_clf, tfidf_vectorizer)

print("\nRandom Forest Classifier on Test Data")
evaluate_on_test_data(rf_clf, tfidf_vectorizer)

import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters for grid search
rf_parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

logR_parameters = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l2']
}

svm_parameters = {
    'C': [0.1, 1.0, 10.0]
}

def train_and_evaluate_models(train_data, test_data):
   
    
    # First phase: Grid search with basic TF-IDF
    print("Performing initial grid search...")
   
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['Statement'][:10000])
    
    # Perform grid search for each model
    models = {
        'Random Forest': (rf_clf, rf_parameters),
        'Logistic Regression': (logR_clf, logR_parameters),
        'SVM': (svm_clf, svm_parameters)
    }
    
    best_score = 0
    best_model_name = None
    best_params = None
    best_ngram = None
    
    for name, (model, params) in models.items():
        print(f"\nPerforming {name} Grid Search...")
        grid = GridSearchCV(model, params, cv=5, n_jobs=-1)
        grid.fit(X_train_tfidf, train_data['Label'][:10000])
        print(f"Best {name} Parameters:", grid.best_params_)
        print(f"Best Score:", grid.best_score_)
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model_name = name
            best_params = grid.best_params_
    
    # Second phase: Find best n-gram range for the best model
    print(f"\nFinding best n-gram range for {best_model_name}...")
    ngram_ranges = [(1, 1), (1, 2), (1, 3), (1, 4)]
    best_ngram_score = 0
    
    for ngram_range in ngram_ranges:
        print(f"\nTesting n-gram range {ngram_range}")
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
        X_train_tfidf = tfidf.fit_transform(train_data['Statement'])
        
        if best_model_name == 'Random Forest':
            model = RandomForestClassifier(**best_params, n_jobs=3)
        elif best_model_name == 'Logistic Regression':
            model = LogisticRegression(**best_params)
        else:  # SVM
            model = svm.LinearSVC(**best_params)
        
        model.fit(X_train_tfidf, train_data['Label'])
        score = model.score(X_train_tfidf, train_data['Label'])
        
        if score > best_ngram_score:
            best_ngram_score = score
            best_ngram = ngram_range
    
    # Final phase: Train the best model with best configuration
    print(f"\nTraining final model: {best_model_name}")
    print(f"Best n-gram range: {best_ngram}")
    print(f"Best parameters: {best_params}")
    
    # Create final vectorizer and transform data
    final_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=best_ngram)
    X_train_tfidf_final = final_vectorizer.fit_transform(train_data['Statement'])
    X_test_tfidf_final = final_vectorizer.transform(test_data['Statement'])
    
    # Create and train final model
    if best_model_name == 'Random Forest':
        final_model = RandomForestClassifier(**best_params, n_jobs=3)
    elif best_model_name == 'Logistic Regression':
        final_model = LogisticRegression(**best_params)
    else:  # SVM
        final_model = svm.LinearSVC(**best_params)
    
    final_model.fit(X_train_tfidf_final, train_data['Label'])
    
    # Evaluate final model
    predictions = final_model.predict(X_test_tfidf_final)
    print("\nFinal Model Performance:")
    print(classification_report(test_data['Label'], predictions))
    
    # Save the model and vectorizer
    model_package = {
        'model_name': best_model_name,
        'vectorizer': final_vectorizer,
        'model': final_model,
        'parameters': best_params,
        'ngram_range': best_ngram
    }
    
    joblib.dump(model_package, 'best_fake_news_detector.pkl')
    print("\nModel package saved as 'best_fake_news_detector.pkl'")
    
    # Verify saved model
    test_load = joblib.load('best_fake_news_detector.pkl')
    verify_X = test_load['vectorizer'].transform(test_data['Statement'])
    verify_predictions = test_load['model'].predict(verify_X)
    verify_accuracy = np.mean(verify_predictions == test_data['Label'])
    print(f"Verified accuracy of saved model: {verify_accuracy:.4f}")
    
    return model_package

# Run the training and evaluation
if __name__ == "__main__":
    best_model_package = train_and_evaluate_models(DataPrep.train_news, DataPrep.test_news)

