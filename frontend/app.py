import streamlit as st
import requests

st.write("Hello world")
st.write("This is a simple frontend for our ML model.")

st.header("Make a Prediction")
features = []
for i in range(4):
    feature = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(feature)
if st.button("Predict"):
    payload = {"features": features}
    try:
        r = requests.post("http://host.docker.internal:8000/predict", json=payload, timeout=5)
        r.raise_for_status()
        result = r.json()
        
        st.success("Prediction successful!")
        st.write("**Request:**")
        st.json(payload)
        st.write("**Response:**")
        st.json(result)
        
        # Display in a nicer format
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", result["predicted_class"])
        with col2:
            st.metric("Probability", f"{result['probability']:.4f}")
            
    except requests.Timeout:
        st.error("Request timed out - backend may not be running")
    except requests.ConnectionError:
        st.error("Connection failed - make sure backend is running at http://host.docker.internal:8000")
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

