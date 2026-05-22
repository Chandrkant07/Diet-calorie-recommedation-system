# Diet Calorie Recommendation System
# An AI-powered food calorie tracking and diet recommendation application
# Author: Chandrakanta Jena
# License: MIT

# import streamlit as st
# import google.generativeai as genai
# import pandas as pd
# import plotly.graph_objects as go
# import re  # New import for better text extraction
# from io import BytesIO
# from PIL import Image
# from datetime import datetime

# # Configure Gemini API
# genai.configure(api_key="AIzaSyD9Gx3RTE4ejA-TruwjtrK-mFFnkRL3eVE")

# # Application Constants
# Set daily calorie requirement
# DAILY_CALORIE_GOAL = 2000  # Default goal, can be customized per user

# 
# BMI Calculator Helper
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate BMI and return category."""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": round(bmi, 1), "category": category}

# Function to get food name and calories from Gemini API
# def get_food_details(image_data: list) -> tuple:
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     calorie_prompt = (
#         "Analyze this food image carefully. Identify the food item and estimate its total calorie content accurately."
#         " Format: 'Food: <food name>, Calories: <calories>'. If unsure, estimate."
#     )
    
#     response = model.generate_content([image_data[0], calorie_prompt])
#     raw_text = response.text.strip()  # Capture raw response

#     # Log the API response for debugging
#     print(f"API Response: {raw_text}")

#     # Use regex to extract food name and calories more flexibly
#     match = re.search(r"Food:\s*(.+?),\s*Calories:\s*(\d+)", raw_text, re.IGNORECASE)
    
#     if match:
#         food_name = match.group(1).strip()
#         calories = int(match.group(2).strip())
#         return food_name, calories
#     else:
#         st.error("AI response not in expected format. Please try another image.")
#         return None, None

# # Function to analyze food image
# def analyze_food_image(image_bytes, mime_type):
    """Analyze a food image and return food name with calorie count."""
#     if image_bytes:
#         image_data = [{"mime_type": mime_type, "data": image_bytes}]
#         return get_food_details(image_data)  # Returns food_name, calories

# 
# Daily Nutrition Summary Helper
def get_daily_summary(calorie_history: list) -> dict:
    """Generate a summary of daily nutrition intake."""
    today = datetime.now().date()
    today_items = [item for item in calorie_history if datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S').date() == today]
    total = sum(item['calories'] for item in today_items)
    return {
        "total_calories": total,
        "meal_count": len(today_items),
        "avg_per_meal": round(total / max(len(today_items), 1), 1),
        "remaining": max(0, DAILY_CALORIE_GOAL - total)
    }

# Load calorie history from session state
# if "calorie_history" not in st.session_state:
#     st.session_state.calorie_history = []

# # Streamlit UI
# st.set_page_config(page_title="Food Calorie Tracker", page_icon="🍔", layout="wide")
# st.title("🍕 Food Calorie Tracker Dashboard 🍔")

# # Sidebar - Image Upload
# st.sidebar.header("📸 Upload or Capture Food Image")
# option = st.sidebar.selectbox("Choose an option:", ["Upload a photo", "Capture a photo"])

# image = None
# if option == "Upload a photo":
#     uploaded_image = st.sidebar.file_uploader("Upload a food image (JPG/PNG)", type=["jpg", "jpeg", "png"])
#     if uploaded_image:
#         image = Image.open(uploaded_image)
#         st.sidebar.image(image, caption="Uploaded Food Image", use_column_width=True)

# elif option == "Capture a photo":
#     captured_image = st.sidebar.camera_input("Take a photo")
#     if captured_image:
#         image = Image.open(captured_image)
#         st.sidebar.image(image, caption="Captured Food Image", use_column_width=True)

# # Process image if uploaded or captured
# if image:
#     img_bytes_io = BytesIO()
#     image.save(img_bytes_io, format="JPEG")
#     img_bytes = img_bytes_io.getvalue()
    
#     food_name, calories = analyze_food_image(img_bytes, "image/jpeg")
    
#     if food_name and calories:
#         # Store the food name and calorie count in session state
#         st.session_state.calorie_history.append({
#             "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "food": food_name,
#             "calories": calories
#         })

# # Convert history to DataFrame
# df = pd.DataFrame(st.session_state.calorie_history)

# # Initialize default values
# total_calories = 0
# remaining_calories = DAILY_CALORIE_GOAL

# # Calculate today's total and remaining calories if data exists
# if not df.empty:
#     df["date"] = pd.to_datetime(df["date"])
#     today = datetime.now().date()
#     today_df = df[df["date"].dt.date == today]
    
#     total_calories = today_df["calories"].sum()
#     remaining_calories = max(0, DAILY_CALORIE_GOAL - total_calories)

# # Display calorie metrics
# col1, col2 = st.columns(2)
# with col1:
#     st.metric(label="Total Calories Consumed Today", value=f"🔥 {total_calories} kcal")
# with col2:
#     st.metric(label="Remaining Calories for the Day", value=f"🟢 {remaining_calories} kcal")

# # Progress Pie Chart (Avoid crash when no data is present)
# fig_pie = go.Figure(data=[go.Pie(
#     labels=["Consumed", "Remaining"], 
#     values=[total_calories, remaining_calories], 
#     hole=0.7, 
#     marker=dict(colors=['#FF5733', '#33FF57']),
#     textinfo='percent'
# )])
# fig_pie.update_layout(title_text="Daily Calorie Progress", showlegend=False)
# st.plotly_chart(fig_pie, use_container_width=True)

# # Show Food History Table
# if not df.empty:
#     st.subheader("📜 Today's Food Intake History")
#     st.dataframe(df[::-1])  # Show latest entries first


import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
import re
from io import BytesIO
from PIL import Image
from datetime import datetime
import requests
import json

# Configure Gemini API
genai.configure(api_key="AIzaSyD9Gx3RTE4ejA-TruwjtrK-mFFnkRL3eVE")

# Application Constants
# Set daily calorie requirement
DAILY_CALORIE_GOAL = 2000  # Default goal, can be customized per user

# Hugging Face API configuration - Add your API key here
HUGGINGFACE_API_KEY = "YOUR_HUGGINGFACE_API_KEY"  # Replace with your actual API key
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf"  # You can change to another model if preferred


# BMI Calculator Helper
def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """Calculate BMI and return category."""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    return {"bmi": round(bmi, 1), "category": category}

# Function to get food name and calories from Gemini API
def get_food_details(image_data: list) -> tuple:
    model = genai.GenerativeModel('gemini-1.5-flash')
    calorie_prompt = (
        "Analyze this food image carefully. Identify the food item and estimate its total calorie content accurately."
        " Format: 'Food: <food name>, Calories: <calories>'. If unsure, estimate."
    )
    
    response = model.generate_content([image_data[0], calorie_prompt])
    raw_text = response.text.strip()  # Capture raw response

    # Log the API response for debugging
    print(f"API Response: {raw_text}")

    # Use regex to extract food name and calories more flexibly
    match = re.search(r"Food:\s*(.+?),\s*Calories:\s*(\d+)", raw_text, re.IGNORECASE)
    
    if match:
        food_name = match.group(1).strip()
        calories = int(match.group(2).strip())
        return food_name, calories
    else:
        st.error("AI response not in expected format. Please try another image.")
        return None, None

# New function to get diet recommendations using Hugging Face API
def get_diet_recommendations_huggingface(food_name: str) -> dict:
    """Get diet recommendations using Hugging Face API."""
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    For the food '{food_name}', analyze its nutritional value and provide recommendations. 
    Return a JSON object with the following structure:
    {{
        "nutritional_assessment": "A brief nutritional assessment (50 words max)",
        "healthier_alternatives": ["alternative1", "alternative2", "alternative3"],
        "complementary_foods": ["complementary1", "complementary2"]
    }}
    Keep the response in valid JSON format.
    """
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            # Extract the JSON from the response - might need to clean it up
            response_text = response.json()[0]["generated_text"]
            
            # Extract JSON from the response if it's embedded in other text
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:  # Parse structured JSON from AI response
                response_text = json_match.group(1)
                
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If we couldn't parse the JSON, return the raw text
                return {"raw_response": response_text}
        else:
            return {"error": f"API request failed with status code {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"error": f"Error connecting to Hugging Face API: {str(e)}"}

# Function to analyze food image
def analyze_food_image(image_bytes, mime_type):
    """Analyze a food image and return food name with calorie count."""
    if image_bytes:
        image_data = [{"mime_type": mime_type, "data": image_bytes}]
        return get_food_details(image_data)  # Returns food_name, calories


# Daily Nutrition Summary Helper
def get_daily_summary(calorie_history: list) -> dict:
    """Generate a summary of daily nutrition intake."""
    today = datetime.now().date()
    today_items = [item for item in calorie_history if datetime.strptime(item['date'], '%Y-%m-%d %H:%M:%S').date() == today]
    total = sum(item['calories'] for item in today_items)
    return {
        "total_calories": total,
        "meal_count": len(today_items),
        "avg_per_meal": round(total / max(len(today_items), 1), 1),
        "remaining": max(0, DAILY_CALORIE_GOAL - total)
    }

# Load calorie history from session state
if "calorie_history" not in st.session_state:
    st.session_state.calorie_history = []

# Store current food recommendations
if "current_recommendation" not in st.session_state:
    st.session_state.current_recommendation = None

# Streamlit UI
st.set_page_config(page_title="Food Calorie Tracker", page_icon="🍔", layout="wide")
st.title("🍕 Food Calorie Tracker Dashboard 🍔")

# API Selection in sidebar
st.sidebar.header("🔧 Settings")
api_choice = st.sidebar.radio("Choose AI Provider for Diet Recommendations:", 
                             ["Gemini AI", "Hugging Face"])

# Sidebar - Image Upload
st.sidebar.header("📸 Upload or Capture Food Image")
option = st.sidebar.selectbox("Choose an option:", ["Upload a photo", "Capture a photo"])

image = None
if option == "Upload a photo":
    uploaded_image = st.sidebar.file_uploader("Upload a food image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.sidebar.image(image, caption="Uploaded Food Image", use_column_width=True)

elif option == "Capture a photo":
    captured_image = st.sidebar.camera_input("Take a photo")
    if captured_image:
        image = Image.open(captured_image)
        st.sidebar.image(image, caption="Captured Food Image", use_column_width=True)

# Process image if uploaded or captured
if image:
    img_bytes_io = BytesIO()
    image.save(img_bytes_io, format="JPEG")
    img_bytes = img_bytes_io.getvalue()
    
    food_name, calories = analyze_food_image(img_bytes, "image/jpeg")
    
    if food_name and calories:
        # Store the food name and calorie count in session state
        st.session_state.calorie_history.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "food": food_name,
            "calories": calories
        })
        
        # Get dietary recommendations based on selected API
        if api_choice == "Gemini AI":
            model = genai.GenerativeModel('gemini-1.5-flash')
            recommendation_prompt = f"""
            For the food '{food_name}', provide:
            1. A brief nutritional assessment (50 words max)
            2. Three healthier alternatives if this is not the healthiest option
            3. Two complementary foods that would create a balanced meal
            
            Return as a JSON object:
            {{
                "nutritional_assessment": "...",
                "healthier_alternatives": ["alt1", "alt2", "alt3"],
                "complementary_foods": ["food1", "food2"]
            }}
            """
            try:
                response = model.generate_content(recommendation_prompt)
                # Try to parse as JSON
                try:
                    json_match = re.search(r'({.*})', response.text, re.DOTALL)
                    if json_match:  # Parse structured JSON from AI response
                        clean_json = json_match.group(1)
                        st.session_state.current_recommendation = json.loads(clean_json)
                    else:
                        st.session_state.current_recommendation = {"raw_response": response.text}
                except:
                    st.session_state.current_recommendation = {"raw_response": response.text}
            except Exception as e:
                st.session_state.current_recommendation = {"error": f"Error getting recommendations: {str(e)}"}
        else:  # Hugging Face
            if HUGGINGFACE_API_KEY == "YOUR_HUGGINGFACE_API_KEY":
                st.sidebar.error("⚠️ Please set your Hugging Face API key in the code")
            else:
                st.session_state.current_recommendation = get_diet_recommendations_huggingface(food_name)

# Convert history to DataFrame
df = pd.DataFrame(st.session_state.calorie_history)

# Initialize default values
total_calories = 0
remaining_calories = DAILY_CALORIE_GOAL

# Calculate today's total and remaining calories if data exists
if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    today = datetime.now().date()
    today_df = df[df["date"].dt.date == today]
    
    total_calories = today_df["calories"].sum()
    remaining_calories = max(0, DAILY_CALORIE_GOAL - total_calories)

# Main content area with tabs
tab1, tab2 = st.tabs(["Dashboard", "Diet Recommendations"])

with tab1:
    # Display calorie metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Calories Consumed Today", value=f"🔥 {total_calories} kcal")
    with col2:
        st.metric(label="Remaining Calories for the Day", value=f"🟢 {remaining_calories} kcal")

    # Progress Pie Chart (Avoid crash when no data is present)
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Consumed", "Remaining"], 
        values=[total_calories, remaining_calories], 
        hole=0.7, 
        marker=dict(colors=['#FF5733', '#33FF57']),
        textinfo='percent'
    )])
    fig_pie.update_layout(title_text="Daily Calorie Progress", showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Show Food History Table
    if not df.empty:
        st.subheader("📜 Today's Food Intake History")
        st.dataframe(df[::-1])  # Show latest entries first

with tab2:
    st.subheader("🥗 Diet Recommendations")
    
    if st.session_state.current_recommendation:
        # Display the API provider being used
        st.markdown(f"**AI Provider:** {api_choice}")
        
        # Check if there's an error
        if "error" in st.session_state.current_recommendation:
            st.error(st.session_state.current_recommendation["error"])
        # Check if there's a raw response that couldn't be parsed
        elif "raw_response" in st.session_state.current_recommendation:
            st.info("The AI provided the following recommendations:")
            st.markdown(st.session_state.current_recommendation["raw_response"])
        else:
            # Display structured recommendations
            recommendations = st.session_state.current_recommendation
            
            # Display the nutritional assessment
            st.markdown("### Nutritional Assessment")
            st.info(recommendations.get("nutritional_assessment", "No assessment available"))
            
            # Display healthier alternatives
            st.markdown("### Healthier Alternatives")
            alternatives = recommendations.get("healthier_alternatives", [])
            if alternatives:
                for alt in alternatives:
                    st.markdown(f"- {alt}")
            else:
                st.markdown("No healthier alternatives suggested.")
            
            # Display complementary foods
            st.markdown("### Complementary Foods for a Balanced Meal")
            complementary = recommendations.get("complementary_foods", [])
            if complementary:
                for food in complementary:
                    st.markdown(f"- {food}")
            else:
                st.markdown("No complementary foods suggested.")
    else:
        if not df.empty:
            st.info("Upload or capture a food image to get personalized diet recommendations.")
        else:
            st.info("No food detected yet. Upload or capture a food image to get started.")
    
    # Add a nutrition tips section
    st.markdown("---")
    st.subheader("💡 General Nutrition Tips")
    
    # Random tips that rotate
    tips = [
        "Try to include protein with every meal to stay fuller longer.",
        "Aim for at least 5 servings of fruits and vegetables daily.",
        "Drinking water before meals can help control portion sizes.",
        "Whole grains provide more nutrients and fiber than refined grains.",
        "Healthy fats like avocados and nuts are important for brain health.",
        "Eating slowly helps your body recognize when it's full.",
        "Try to include a variety of colors on your plate for different nutrients.",
        "Portion control is key - use smaller plates to help with this."
    ]
    
    import random
    st.markdown(f"**Tip of the day:** {random.choice(tips)}")



# Updated: 2026-04-04T11:30:00








