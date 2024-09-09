import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from makeDonut import make_donut

# Page configurations
st.set_page_config(
    page_title="Exploring the Impact of Player Stats on Game Outcomes: A Scottie Barnes Case Study",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Title
st.title("🏀 Exploring the Impact of Player Stats on Game Outcomes: A Scottie Barnes Case Study")

# Introduction
st.markdown("""
The aim of this project is to investigate the relationship between a player statistics and the outcomes of basketball games, with a specific focus on Scottie Barnes. The dataset utilized for this analysis, sourced from basketballreference.com, contains comprehensive information about Barnes' performance in various games, including points scored (PTS), assists (AST), rebounds (TRB), blocks (BLK), steals (STL), and plus/minus (+/-).

Given the rich dataset, the primary question of interest revolves around understanding how Scottie Barnes' individual performance metrics influence the likelihood of a win or loss for the Toronto Raptors. Exploring this relationship could provide valuable insights into the significance of different player statistics in determining game outcomes.

This section will delve into the data, the motivation behind the analysis, and the key questions guiding the exploration.
""")

# Load CSV file from local directory
current_directory = os.path.dirname(__file__)

# Define the relative path to the CSV file
csv_path = os.path.join(current_directory, 'scottieBarnesData.csv')

# Read the CSV file using the relative path
data = pd.read_csv(csv_path)
data['WL'] = data['WL'].astype(str)
data['Outcome'] = data['WL'].apply(lambda x: 1 if x[0] == 'W' else 0)

# Data Preprocessing
features = data[['PTS', 'AST', 'TRB', 'BLK', 'STL', '+/-']]
target = data['Outcome']

# Label encoding for the outcome column
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)
features.fillna(features.mean(), inplace=True)

# Displaying dataset
st.subheader("Dataset Overview")
st.write(data.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

with st.sidebar:
    # Model Accuracy
    st.subheader("Model Accuracy")
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Feature Importance
    st.subheader("Feature Importances")
    feature_importances = mlp.coefs_[0]
    importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances.mean(axis=1)})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    st.write(importance_df)

    # Define the win percentage
    wins_percentage = 63.2
    input_colour = 'green'

    # Create the donut chart
    donut_chart = make_donut(wins_percentage, input_colour)

    # Display the chart
    st.altair_chart(donut_chart)

    # Define the loss percentage percentage
    wins_percentage = 74.6
    input_colour = 'red'

    # Create the donut chart
    donut_chart = make_donut(wins_percentage, input_colour)

    # Display the chart
    st.altair_chart(donut_chart)


# Pair Plot
st.subheader("Pair Plot: Stats vs Outcomes")
pair_plot_data = pd.concat([features, data['WL']], axis=1)
pair_plot_data['Outcome'] = pair_plot_data['WL'].apply(lambda x: 'Win' if x[0] == 'W' else 'Loss')

# Use Seaborn to create the pair plot
fig = sns.pairplot(pair_plot_data, hue='Outcome')

# Pass the figure to Streamlit
st.pyplot(fig.fig)
