import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Function to check if the status is Normal or Mental Illness
def check_if_normal(predictions, targets):
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if np.all(predictions == 0):
        return "Normal", []
    else:
        detected_labels = targets.columns[predictions[0] == 1].tolist()
        return f"Mental Illness Detected: {', '.join(detected_labels)}", detected_labels

# Function to provide recommendations based on predictions
def get_recommendations(detected_labels):
    recommendations = {
        "Bipolar disorder": "Consult a psychiatrist for further evaluation and mood management.",
        "Schizophrenia": "Seek immediate help from a mental health professional for proper diagnosis and treatment.",
        "Depression": "Consider talking to a counselor or psychologist and explore therapy or support groups.",
        "Anxiety disorder": "Practice relaxation techniques like meditation or deep breathing and consult a psychologist.",
        "PTSD": "Support from a therapist or support groups can help manage PTSD symptoms.",
    }
    return [recommendations[label] for label in detected_labels]

# Load dataset and model
file_path = 'Mentalillness.csv'
data = pd.read_csv(file_path)
model = joblib.load('mental_health_model.pkl')

# Remove irrelevant columns
data = data.drop(columns=['Sleep disturbance.1', 'Intrusive memories or flashback'])

# Separate features and targets
features = data.drop(columns=['ID', 'Bipolar disorder', 'Schizophrenia', 'Depression', 'Anxiety disorder', 'PTSD'])
targets = data[['Bipolar disorder', 'Schizophrenia', 'Depression', 'Anxiety disorder', 'PTSD']]

questions = features.columns

# Streamlit UI
st.title("Mental Health Prediction")

st.write("### Please answer the following questions by selecting Yes or No:")

answers = {}

# User input for each symptom
for idx, question in enumerate(questions):
    question_translated = question.replace('_', ' ').capitalize()
    question_with_number = f"**{idx+1}. Do you experience {question_translated}?**"
    answer = st.radio(
        question_with_number,
        options=['Not selected', 'Yes', 'No'],
        index=0,
        key=f"q_{idx}"  # Ensure unique keys for Streamlit widgets
    )
    if answer == 'Not selected':
        answers[question] = None
    else:
        answers[question] = 1 if answer == 'Yes' else 0

if st.button("Predict"):
    if None in answers.values():
        st.write("### Please complete all the questions before proceeding!")
    else:
        user_input = np.array([list(answers.values())]).reshape(1, -1)
        prediction = model.predict(user_input)
        result, detected_labels = check_if_normal(prediction, targets)

        # Display prediction results
        st.markdown(f"## **Prediction Result:** \n\n{result}")

        # Visualize user answers
        yes_count = sum(value == 1 for value in answers.values())
        no_count = sum(value == 0 for value in answers.values())
        total = yes_count + no_count

        labels = ['Yes', 'No']
        counts = [yes_count, no_count]
        percentages = [count / total * 100 for count in counts]

        # Adjust the figure size and text size for the chart
        fig, ax = plt.subplots(figsize=(2, 2))  # Smaller size
        ax.pie(
            counts, 
            labels=labels, 
            autopct=lambda p: f'{p:.1f}% ({int(p * total / 100)})', 
            colors=['green', 'red'], 
            startangle=60, 
            textprops={'fontsize': 8}  # Smaller text size
        )
        ax.set_title('Your Answer Distribution', fontsize=10)
        st.pyplot(fig)

        # Display counts and percentages
        st.markdown(f"### Number of 'Yes' answers: {yes_count} ({percentages[0]:.1f}%)")
        st.markdown(f"### Number of 'No' answers: {no_count} ({percentages[1]:.1f}%)")

        # Display recommendations if any mental illness is detected
        if detected_labels:
            st.subheader("Recommendations for You:")
            recommendations = get_recommendations(detected_labels)
            for rec in recommendations:
                st.write(f"- {rec}")
