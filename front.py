import streamlit as st
import requests

# Streamlit app title
st.title("Fake News Detector")

# User input field for news text
news_text = st.text_area("Enter a news article or statement to check its credibility:")

# Submit button
if st.button("Analyze Text"):
    if news_text.strip():  # Ensure that the input is not empty
        # Send a POST request to FastAPI backend
        url = "http://localhost:3000/api/predict"  # URL of the FastAPI backend
        payload = {"text": news_text}
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                # Get prediction and confidence from the response
                data = response.json()
                is_fake = data["is_fake"]
                confidence = data["confidence"]
                if confidence < 0.55:
                 is_fake = not bool(is_fake)
                else:
                 is_fake = bool(is_fake)

                
                # Show result on the frontend
                if is_fake:
                    st.error(f"Potentially Fake News - Confidence: {confidence * 100:.2f}%")
                else:
                    st.success(f"Likely Reliable News - Confidence: {confidence * 100:.2f}%")
            else:
                st.error("Error in getting prediction from the backend.")
        
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter some text to analyze.")
