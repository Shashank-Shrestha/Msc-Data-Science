import pickle
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import random
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Set page configuration at the very top of the script
st.set_page_config(page_title="Breast Cancer Risk Prediction Quiz", layout="centered", page_icon="🩺")

# Define the DNDT class
class DNDT(nn.Module):
    def __init__(self, input_dim, num_classes, max_depth=3):
        super(DNDT, self).__init__()
        self.max_depth = max_depth
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_units = [10, 20, 10]
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_units[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[1], self.hidden_units[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_units[2], self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

# Load the pre-trained models and their accuracies from pickle files
def load_models():
    with open('gradient_boosting_model.pickle', 'rb') as f:
        gb_data = pickle.load(f)
        gb_model = gb_data['model']
        gb_accuracy = gb_data['accuracy']

    with open('LogisticRegression.pickle', 'rb') as f:
        lr_data = pickle.load(f)
        lr_model = lr_data['model']
        lr_accuracy = lr_data['accuracy']

    with open('decision_tree_model.pickle', 'rb') as f:
        dndt_data = pickle.load(f)
        dndt_model = dndt_data['model']
        dndt_accuracy = dndt_data['accuracy']

    with open('random_forest_model.pickle', 'rb') as f:
        rf_data = pickle.load(f)
        rf_model = rf_data['model']
        rf_accuracy = rf_data['accuracy']

    return gb_model, gb_accuracy, lr_model, lr_accuracy, dndt_model, dndt_accuracy, rf_model, rf_accuracy

# Prediction function with probability output for nuanced risk levels
def predict_with_probabilities(model, inputs):
    if isinstance(model, DNDT):
        inputs_tensor = torch.tensor([inputs], dtype=torch.float32)
        with torch.no_grad():
            outputs = model(inputs_tensor)
            probabilities = outputs.numpy()[0]
    else:
        probabilities = model.predict_proba([inputs])[0]

    return probabilities

# Function to generate a summary based on model predictions and inputs
def generate_summary_with_gen_ai(inputs):
    # Initialize the Ollama model
    llm = Ollama(model="monotykamary/medichat-llama3")
    #llm = Ollama(model="gemma/medichat")

    # Create a prompt template with explicit variable placeholders
    prompt = PromptTemplate(
        input_variables=["age", "menopause", "tumor_size", "inv_nodes", "node_caps", "deg_malig", "breast", "breast_quad", "irradiat"],
        template="""
        You are a highly experienced oncologist specializing in breast cancer. Given the patient's data:
        Age: {age},
        Menopausal status: {menopause},
        Tumor size: {tumor_size} mm,
        Number of invaded lymph nodes: {inv_nodes},
        Presence of a capsule: {node_caps},
        Degree of malignancy: {deg_malig},
        Affected breast: {breast},
        Affected breast quadrant: {breast_quad},
        Radiation treatment history: {irradiat}.
        Assess the likelihood of breast cancer recurrence. Identify the key risk factors present in this case, explain their significance, and suggest evidence-based prevention and monitoring strategies tailored to this patient’s condition. Ensure your analysis is thorough yet concise, and provide actionable advice within 100 words."""
    )

    # Prepare the data as a dictionary
    data_dict = {
        "age": inputs[0], 
        "menopause": inputs[1], 
        "tumor_size": inputs[2], 
        "inv_nodes": inputs[3], 
        "node_caps": inputs[4], 
        "deg_malig": inputs[5], 
        "breast": inputs[6], 
        "breast_quad": inputs[7], 
        "irradiat": inputs[8]
    }

    # Run the chain and get the result
    result = prompt.format(**data_dict)
    return llm(result)

# Draw risk meter with a more nuanced risk level
def draw_risk_meter(risk_level):
    fig, ax = plt.subplots(figsize=(5, 3))

    # Draw the colored arcs for risk levels with correct colors
    colors = ['#8B0000', '#FF0000', '#FF4500', '#FF8C00', '#FFA500', '#FFD700', '#FFFF00', '#ADFF2F', '#32CD32', '#008000']
    for i, color in enumerate(colors):
        arc = patches.Wedge((0.5, 0), 0.4, i * 18, (i + 1) * 18, facecolor=color, edgecolor='white')
        ax.add_patch(arc)

    # Calculate angle for the needle
    angle = (risk_level / 100) * 180
    needle_length = 0.35

    # Calculate the needle end point coordinates
    x_needle = 0.5 + needle_length * np.cos(np.radians(180 - angle))
    y_needle = needle_length * np.sin(np.radians(180 - angle))

    # Draw the needle
    ax.plot([0.5, x_needle], [0, y_needle], color='black', lw=2)

    # Add labels for low, medium, and high
    ax.text(0.1, 0.1, 'Low', ha='center', va='center', fontsize=12, color='black')
    ax.text(0.5, 0.45, 'Medium', ha='center', va='center', fontsize=12, color='black')
    ax.text(0.9, 0.1, 'High', ha='center', va='center', fontsize=12, color='black')

    # Set limits and hide axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.axis('off')

    return fig

# Main function for Streamlit app
def main():
    # Apply custom CSS to make the UI more attractive
    st.markdown(
        """
        <style>
        body {
            background-color: #f7f8fc;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: #333;
        }
        .stApp {
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
            padding: 30px;
            max-width: 900px;
            margin: auto;
        }
        h1 {
            color: #2a9df4;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        p {
            color: #5a5a5a;
            font-size: 18px;
            line-height: 1.6;
        }
        .stButton button {
            background-color: #2a9df4;
            color: white;
            padding: 12px 20px;
            border-radius: 5px;
            border: none;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #1a85d4;
            box-shadow: 0px 6px 15px rgba(42, 157, 244, 0.3);
            transform: translateY(-2px);
        }
        .stMarkdown h2, .stMarkdown h3 {
            color: #2a2a2a;
            font-weight: 600;
            margin-top: 25px;
        }
        .stRadio label {
            font-size: 16px;
            color: #5a5a5a;
        }
        .stRadio {
            margin-bottom: 20px;
        }
        .stMarkdown {
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.title('Breast Cancer Risk Prediction Quiz')
    st.markdown("""
        Welcome to the Breast Cancer Risk Prediction Quiz! 🧠
        
        Answer a few questions to help us understand your risk factors, and our models will predict the likelihood of breast cancer recurrence.
    """)

    # Load models
    gb_model, gb_accuracy, lr_model, lr_accuracy, dndt_model, dndt_accuracy, rf_model, rf_accuracy = load_models()

    # Question templates for each feature with icons
    question_templates = {
        "age": [
            "👶 How old are you? (Choose your age range)",
            "👶 Select your age group:",
            "👶 What's your age range?",
            "👶 Can you tell us your age group?"
        ],
        "menopause": [
            "🧬 What's your menopausal status?",
            "🧬 Select your current menopausal status:",
            "🧬 What stage is your menopause?",
            "🧬 Please choose your menopausal status:"
        ],
        "tumor_size": [
            "📏 How large is the tumor? (in mm)",
            "📏 What is the size of the tumor?",
            "📏 Select the tumor size range:",
            "📏 What's the tumor size in millimeters?"
        ],
        "inv_nodes": [
            "🦠 How many lymph nodes are invaded?",
            "🦠 What's the count of invaded lymph nodes?",
            "🦠 Number of invaded lymph nodes:",
            "🦠 Can you specify the number of invaded lymph nodes?"
        ],
        "node_caps": [
            "💊 Is there a node capsule present?",
            "💊 Is a node capsule detected?",
            "💊 Presence of node capsule:",
            "💊 Has a node capsule been identified?"
        ],
        "deg_malig": [
            "📉 What is the degree of malignancy?",
            "📉 Select the malignancy degree:",
            "📉 How malignant is the condition?",
            "📉 What's the malignancy level?"
        ],
        "breast": [
            "🫀 Which breast is affected?",
            "🫀 Choose the affected breast:",
            "🫀 Which side is the affected breast?",
            "🫀 Indicate the breast with the condition:"
        ],
        "breast_quad": [
            "📍 Which quadrant of the breast is affected?",
            "📍 Where in the breast is the issue located?",
            "📍 Select the affected breast quadrant:",
            "📍 Which part of the breast is affected?"
        ],
        "irradiat": [
            "🌟 Have you received radiation treatment?",
            "🌟 Was radiation therapy part of your treatment?",
            "🌟 Did you undergo radiation treatment?",
            "🌟 Is radiation therapy in your treatment history?"
        ]
    }

    # Dictionary for mapping categorical inputs to numeric values
    mappings = {
        'age': {'20-23': 0, '24-27': 1, '28-31': 2, '32-35': 3},
        'menopause': {'ge40': 0, 'premeno': 1, 'lt40': 2},
        'tumor_size': {'0-4': 0, '5-9': 1, '10-14': 2, '15-19': 3, '20-24': 4, '25-29': 5, '30-34': 6, '35-39': 7, '40-44': 8, '45-49': 9},
        'inv_nodes': {'0-2': 0, '3-5': 1, '6-8': 2, '9-11': 3, '12-14': 4, '15-17': 5, '18-20': 6, '21-23': 7, '24-26': 8},
        'node_caps': {'TRUE': 1, 'FALSE': 0},
        'deg_malig': {'1': 0, '2': 1, '3': 2},
        'breast': {'right': 0, 'left': 1},
        'breast_quad': {'right_up': 0, 'left_low': 1, 'left_up': 2, 'right_low': 3, 'central': 4},
        'irradiat': {'TRUE': 1, 'FALSE': 0}
    }

    # Generate random questions once per session with default selections
    if 'selected_questions' not in st.session_state:
        selected_questions = {}
        for key, templates in question_templates.items():
            selected_questions[key] = random.choice(templates)
        st.session_state.selected_questions = selected_questions

    # Collect user input with the randomly selected questions
    inputs = []
    for key, question in st.session_state.selected_questions.items():
        response = st.radio(question, list(mappings[key].keys()), index=0)  # Pre-select first option
        mapped_value = mappings[key][response]
        inputs.append(mapped_value)

    # Add prediction button at the end
    if st.button('Get My Risk Prediction!'):
        try:
            # Perform predictions with each model and get probability outputs
            gb_probabilities = predict_with_probabilities(gb_model, inputs)
            lr_probabilities = predict_with_probabilities(lr_model, inputs)
            dndt_probabilities = predict_with_probabilities(dndt_model, inputs)
            rf_probabilities = predict_with_probabilities(rf_model, inputs)

            # Calculate risk level based on the best model's probability
            gb_risk_level = gb_probabilities[1] * 100  # Probability of class 1 (recurrence)
            lr_risk_level = lr_probabilities[1] * 100
            dndt_risk_level = dndt_probabilities[1] * 100
            rf_risk_level = rf_probabilities[1] * 100

            # Display model accuracies with a title
            st.subheader('📊 Model Accuracy:')
            st.write(f"**Gradient Boosting Accuracy:** {gb_accuracy * 100:.2f}%")
            st.write(f"**Logistic Regression Accuracy:** {lr_accuracy * 100:.2f}%")
            st.write(f"**DNDT Accuracy:** {dndt_accuracy * 100:.2f}%")
            st.write(f"**Random Forest Accuracy:** {rf_accuracy * 100:.2f}%")

            # Show all model predictions and choose the best model based on accuracy
            st.subheader('🔍 Model Predictions:')
            st.write(f'**Gradient Boosting Risk Level:** {gb_risk_level:.2f}%')
            st.write(f'**Logistic Regression Risk Level:** {lr_risk_level:.2f}%')
            st.write(f'**DNDT Risk Level:** {dndt_risk_level:.2f}%')
            st.write(f'**Random Forest Risk Level:** {rf_risk_level:.2f}%')

            best_model_accuracy = max(gb_accuracy, lr_accuracy, dndt_accuracy, rf_accuracy)
            if best_model_accuracy == gb_accuracy:
                best_risk_level = gb_risk_level
                best_model_name = "Gradient Boosting"
            elif best_model_accuracy == lr_accuracy:
                best_risk_level = lr_risk_level
                best_model_name = "Logistic Regression"
            elif best_model_accuracy == dndt_accuracy:
                best_risk_level = dndt_risk_level
                best_model_name = "DNDT"
            else:
                best_risk_level = rf_risk_level
                best_model_name = "Random Forest"

            # Display the best model's prediction
            st.subheader(f'🔍 Best Model Prediction: {best_model_name}')
            st.pyplot(draw_risk_meter(best_risk_level))

            # Final Message based on risk level
            if best_risk_level >= 50:
                st.error("🚨 There is a high risk of breast cancer recurrence.")
            elif best_risk_level >= 20:
                st.warning("⚠️ There is a moderate risk of breast cancer recurrence.")
            else:
                st.success("✅ There is a low risk of breast cancer recurrence.")

            # Generate and display AI-generated summary
            st.subheader('📝 Generative AI Summary:')
            ai_summary = generate_summary_with_gen_ai(inputs)
            st.write(ai_summary)

        except Exception as e:
            st.error(f"An error occurred: {e}")


# Run the main function when the script is executed
if __name__ == '__main__':
    main()
