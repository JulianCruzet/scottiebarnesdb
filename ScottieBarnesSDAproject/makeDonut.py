import altair as alt
import pandas as pd

def make_donut(wins_percentage, input_colour):
    # Ensure percentage is valid
    assert 0 <= wins_percentage <= 100, "Percentage should be between 0 and 100"

    losses_percentage = 100 - wins_percentage  # Remaining percentage for the donut

    # Define chart colors
    if input_colour == 'green':
        chart_color = ['#27AE60', '#12783D']  # Green for wins, Gray for the rest
    if input_colour == 'red':
        chart_color = ['#E74C3C', '#781F16']  # Green for wins, Gray for the rest



    # Data preparation
    source = pd.DataFrame({
        "Outcome": ['Wins', 'Other'],
        "% value": [wins_percentage, losses_percentage]
    })
    source_bg = pd.DataFrame({
        "Outcome": ['Wins', 'Other'],
        "% value": [100, 0]
    })
    
    # Create the donut chart
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Outcome:N",
                        scale=alt.Scale(
                            domain=['Wins', 'Other'],
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    # Add text to the chart
    text = plot.mark_text(align='center', color="#000000", font="Lato", fontSize=16, fontWeight=700).encode(
        text=alt.value(f'{wins_percentage:.1f}% ')
    )

    # Background chart
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color=alt.Color("Outcome:N",
                        scale=alt.Scale(
                            domain=['Wins', 'Other'],
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    # Combine the background, plot, and text
    return plot_bg + plot + text
