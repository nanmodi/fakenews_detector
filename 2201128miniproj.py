from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import pandas as pd
import DataPrep
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import joblib


np.random.seed(42)


def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_model_comparison(scores_dict):
    """Plot comparison of model performances."""
    plt.figure(figsize=(12, 6))
    models = list(scores_dict.keys())
    f1_scores = [score['f1'] for score in scores_dict.values()]
    
    bars = plt.bar(models, f1_scores, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f'])
    plt.title('Model Performance Comparison', pad=20)
    plt.xlabel('Models')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    
 
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_roc_curves(models_dict, X_val, y_val):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        else:
            
            y_pred_proba = model.decision_function(X_val)
            
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_precision_recall_curves(models_dict, X_val, y_val):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_val)
            
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        
        plt.plot(recall, precision, label=f'{name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def plot_feature_importance(vectorizer, model, top_n=20):
    """Plot feature importance for models that support it."""
    if hasattr(model, 'feature_importances_'):  
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):  
        importance = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
    else:
        return
    
    feature_names = vectorizer.get_feature_names_out()
    
   
    top_indices = importance.argsort()[-top_n:][::-1]
    top_features = feature_names[top_indices]
    top_importance = importance[top_indices]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(range(top_n), top_importance)
    plt.yticks(range(top_n), top_features)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(model, X_train, y_train, cv=5):
    """Plot learning curves showing training and validation scores."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv, n_jobs=-1, scoring='f1'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('F1 Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def build_confusion_matrix_with_validation(clf, vectorizer):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    confusion = np.array([[0, 0], [0, 0]])
    predictions_all = []
    true_labels_all = []
    
    
    for train_ind, val_ind in k_fold.split(DataPrep.train_news):
        
        train_text = DataPrep.train_news.iloc[train_ind]['Statement']
        train_y = DataPrep.train_news.iloc[train_ind]['Label']
        val_text = DataPrep.train_news.iloc[val_ind]['Statement']
        val_y = DataPrep.train_news.iloc[val_ind]['Label']
        
        
        X_train_tfidf = vectorizer.fit_transform(train_text)
        X_val_tfidf = vectorizer.transform(val_text)
        
        
        clf.fit(X_train_tfidf, train_y)
        predictions = clf.predict(X_val_tfidf)
        
       
        confusion += confusion_matrix(val_y, predictions)
        score = f1_score(val_y, predictions, average='weighted')
        scores.append(score)
        
        predictions_all.extend(predictions)
        true_labels_all.extend(val_y)
    

    plot_confusion_matrix(confusion, f'{clf.__class__.__name__} Validation Set')
    
    
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 6), scores, marker='o', label='F1 Score')
    plt.title(f'{clf.__class__.__name__} F1 Scores Across K-Folds')
    plt.xlabel('Fold')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.legend()
    plt.show()
    
 
    plot_feature_importance(vectorizer, clf)
    
   
    X_train_full = vectorizer.fit_transform(DataPrep.train_news['Statement'])
    y_train_full = DataPrep.train_news['Label']
    plot_learning_curves(clf, X_train_full, y_train_full)
    
    print(f"Average F1 Score for {clf.__class__.__name__} (Validation Set): {sum(scores) / len(scores)}")
    print(f"Confusion matrix for {clf.__class__.__name__}:")
    print(confusion)
    
    return scores, confusion, predictions_all, true_labels_all

def compare_all_models(models_dict, vectorizer):
    """Compare all models with various visualization metrics."""
 
    X_val = vectorizer.transform(DataPrep.test_news['Statement'])
    y_val = DataPrep.test_news['Label']
    
    
    scores_dict = {}
    for name, model in models_dict.items():
        predictions = model.predict(X_val)
        scores_dict[name] = {
            'f1': f1_score(y_val, predictions, average='weighted')
        }
    
 
    plot_model_comparison(scores_dict)
    
    
    plot_roc_curves(models_dict, X_val, y_val)
    
 
    plot_precision_recall_curves(models_dict, X_val, y_val)


ngram_range = (1, 2)  


tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)


nb_clf = MultinomialNB()
logR_clf = LogisticRegression()
svm_clf = svm.LinearSVC()
sgd_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=200)
rf_clf = RandomForestClassifier(n_estimators=50, n_jobs=3)


models_dict = {
    'Naive Bayes': nb_clf,
    'Logistic Regression': logR_clf,
    'SVM': svm_clf,
    'SGD': sgd_clf,
    'Random Forest': rf_clf
}

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
    
    print("Performing initial grid search...")
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['Statement'][:10000])
    
    
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
    
   
    print(f"\nTraining final model: {best_model_name}")
    print(f"Best n-gram range: {best_ngram}")
    print(f"Best parameters: {best_params}")
    
    final_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=best_ngram)
    
    X_train_tfidf_final = final_vectorizer.fit_transform(train_data['Statement'])
    X_test_tfidf_final = final_vectorizer.transform(test_data['Statement'])
    
    
    if best_model_name == 'Random Forest':
        final_model = RandomForestClassifier(**best_params, n_jobs=3)
    elif best_model_name == 'Logistic Regression':
        final_model = LogisticRegression(**best_params)
    else:  # SVM
        final_model = svm.LinearSVC(**best_params)
    
    final_model.fit(X_train_tfidf_final, train_data['Label'])
    
   
    predictions = final_model.predict(X_test_tfidf_final)
    print("\nFinal Model Performance:")
    print(classification_report(test_data['Label'], predictions))
    
  
    model_package = {
        'model_name': best_model_name,
        'vectorizer': final_vectorizer,
        'model': final_model,
        'parameters': best_params,
        'ngram_range': best_ngram
    }
    
    joblib.dump(model_package, 'best_fake_news_detector.pkl')
    print("\nModel package saved as 'best_fake_news_detector.pkl'")
    
    
    test_load = joblib.load('best_fake_news_detector.pkl')
    verify_X = test_load['vectorizer'].transform(test_data['Statement'])
    verify_predictions = test_load['model'].predict(verify_X)
    verify_accuracy = np.mean(verify_predictions == test_data['Label'])
    print(f"Verified accuracy of saved model: {verify_accuracy:.4f}")
    
    return model_package


if __name__ == "__main__":
    print("Starting model training and evaluation...")
    
  
    print("\nEvaluating individual models...")
    for name, clf in models_dict.items():
        print(f"\nEvaluating {name}...")
        scores, confusion, predictions, true_labels = build_confusion_matrix_with_validation(clf, tfidf_vectorizer)
    
    
    print("\nGenerating model comparison visualizations...")
   
    X_train_full = tfidf_vectorizer.fit_transform(DataPrep.train_news['Statement'])
    y_train_full = DataPrep.train_news['Label']
    
    for name, model in models_dict.items():
        print(f"\nTraining {name} on full training data...")
        model.fit(X_train_full, y_train_full)
    
    compare_all_models(models_dict, tfidf_vectorizer)
    
    
    print("\nStarting full model selection and training ...")
    best_model_package = train_and_evaluate_models(DataPrep.train_news, DataPrep.test_news)
    
    print("\nFinal selected model details:")
    print(f"Model type: {best_model_package['model_name']}")
    print(f"Best parameters: {best_model_package['parameters']}")
    print(f"Best n-gram range: {best_model_package['ngram_range']}")
    
    
    print("\nGenerating final visualizations for the best model...")
    X_test = best_model_package['vectorizer'].transform(DataPrep.test_news['Statement'])
    y_test = DataPrep.test_news['Label']
    
    
    print("\nGenerating feature importance plot...")
    plot_feature_importance(best_model_package['vectorizer'], best_model_package['model'])
    
   
    print("\nGenerating learning curves...")
    X_train = best_model_package['vectorizer'].transform(DataPrep.train_news['Statement'])
    y_train = DataPrep.train_news['Label']
    plot_learning_curves(best_model_package['model'], X_train, y_train)
    
    print("\nAnalysis complete! Check the generated visualizations for detailed insights.")



