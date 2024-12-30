# Load the saved model
import joblib
model_package = joblib.load('best_fake_news_detector.pkl')

# Make predictions on new data
def predict_fake_news(text, model_package):
    
    X = model_package['vectorizer'].transform([text])
    
    prediction = model_package['model'].predict(X)
    return prediction[0]


text = input("Enter the news:  ")
prediction = predict_fake_news(text, model_package)
print(prediction)