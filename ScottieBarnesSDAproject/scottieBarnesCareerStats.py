import streamlit as st
import pandas as pd

# Hardcoded Scottie Barnes stats
scottie_stats = {
    'Season': ['2021-22', '2022-23', '2023-24'],
    'Team': ['TOR', 'TOR', 'TOR'],
    'Games Played': [74, 77, 60],
    'Games Started': [74, 76, 60],
    'Minutes': [35.4, 34.8, 34.9],
    'Points': [15.3, 15.3, 19.9],
    'Rebounds': [7.5, 6.6, 8.2],
    'Assists': [3.5, 4.8, 6.1],
    'Steals': [1.1, 1.1, 1.3],
    'Blocks': [0.7, 0.8, 1.5],
    'Field Goals Made': [6.2, 6.0, 7.5],
    'Field Goals Attempted': [12.6, 13.2, 15.7],
    'Field Goal %': [49.2, 45.6, 47.5],
    '3PM': [0.8, 0.8, 1.7],
    '3PA': [2.6, 2.9, 4.9],
    '3P%': [30.1, 28.1, 34.1],
    'Free Throws Made': [2.1, 2.5, 3.3],
    'Free Throws Attempted': [2.9, 3.2, 4.2],
    'Free Throw %': [73.5, 77.2, 78.1],
    'Effective FG%': [55.2, 52.4, 56.6],
    'Offensive Rebounds': [2.6, 2.3, 2.4],
    'Defensive Rebounds': [4.9, 4.3, 5.9],
    'Personal Fouls': [1.8, 2.0, 2.8],
    'Turnovers': [2.6, 2.2, 2.0]
}
