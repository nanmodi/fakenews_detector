from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import numpy as np
import DataPrep  # Assuming this contains the train and test datasets

# Feature Extraction
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

# Transform training data
train_counts = count_vectorizer.fit_transform(DataPrep.train_news['Statement'])
train_tfidf = tfidf_transformer.fit_transform(train_counts)

# Transform test data
test_counts = count_vectorizer.transform(DataPrep.test_news['Statement'])
test_tfidf = tfidf_transformer.transform(test_counts)

# Define models
nb_clf = MultinomialNB()
log_reg_clf = LogisticRegression(max_iter=1000)
svm_clf = LinearSVC()
rf_clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)

# Train and evaluate models
def train_and_evaluate(model, train_features, train_labels, test_features, test_labels, model_name):
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    
    f1 = f1_score(test_labels, predictions, average='weighted')
    cm = confusion_matrix(test_labels, predictions)
    report = classification_report(test_labels, predictions)
    
    print(f"\nModel: {model_name}")
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    return f1, cm

# Train and evaluate each model
results = {}
results['Naive Bayes'] = train_and_evaluate(nb_clf, train_tfidf, DataPrep.train_news['Label'], 
                                            test_tfidf, DataPrep.test_news['Label'], 'Naive Bayes')

results['Logistic Regression'] = train_and_evaluate(log_reg_clf, train_tfidf, DataPrep.train_news['Label'], 
                                                    test_tfidf, DataPrep.test_news['Label'], 'Logistic Regression')

results['SVM'] = train_and_evaluate(svm_clf, train_tfidf, DataPrep.train_news['Label'], 
                                    test_tfidf, DataPrep.test_news['Label'], 'SVM')

results['Random Forest'] = train_and_evaluate(rf_clf, train_tfidf, DataPrep.train_news['Label'], 
                                              test_tfidf, DataPrep.test_news['Label'], 'Random Forest')

# Perform K-Fold Cross Validation
def k_fold_cross_validation(model, features, labels):
    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.zeros((len(set(labels)), len(set(labels))), dtype=int)

    for train_idx, test_idx in k_fold.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        scores.append(f1_score(y_test, predictions, average='weighted'))
        confusion += confusion_matrix(y_test, predictions)
    
    avg_score = np.mean(scores)
    print(f"\nK-Fold Cross Validation:\nAverage F1 Score: {avg_score}\nConfusion Matrix:\n{confusion}")
    return avg_score, confusion

# Example: Run K-Fold for Naive Bayes
print("\nK-Fold for Naive Bayes:")
k_fold_cross_validation(nb_clf, train_tfidf, DataPrep.train_news['Label'])
print("\nK-Fold for Logistic Regression:")
k_fold_cross_validation(log_reg_clf, train_tfidf, DataPrep.train_news['Label'])
print("\nK-Fold for svm:")
k_fold_cross_validation(svm_clf, train_tfidf, DataPrep.train_news['Label'])
print("\nK-Fold for Random Forest:")
k_fold_cross_validation(rf_clf, train_tfidf, DataPrep.train_news['Label'])
# You can repeat K-Fold for other models as needed

