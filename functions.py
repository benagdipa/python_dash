import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plotly.subplots import make_subplots
from dash import dash_table
from dash.dash_table.Format import Group
import os

def load_data(filename):
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Construct full file path
    filepath = os.path.join(script_dir, filename)
    df = pd.read_csv(filepath)
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df

data = load_data('data/5G_NR_data.csv')

# Extract unique cell_ids, kpi_categories, and pis
cell_ids = data['cell_id'].unique()
kpi_categories = data['kpi_category'].unique()
pis = data['pi'].unique()

# Extract the minimum and maximum dates
min_date = data['date_time'].min()
max_date = data['date_time'].max()

# Generate statistical summary
summary = data.describe(include=[np.number]).transpose().round(2)

# Function to convert summary DataFrame to a list of dictionaries for DataTable
def df_to_table(df):
    return [{column: row[i] for i, column in enumerate(df.columns)} for row in df.values]

bright_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

line_colors = bright_colors * (len(pis) // len(bright_colors)) + bright_colors[:len(pis) % len(bright_colors)]

def resample_data(selected_cell, selected_pis, date_range, frequency):
    """Resample data for selected cell and PIs over specified date range."""
    start_date, end_date = date_range
    # Ensure selected_pis is a list
    if not isinstance(selected_pis, list):
        selected_pis = [selected_pis]
    # Now proceed with filtering and resampling
    resampled_data = data[
        (data['cell_id'] == selected_cell) &
        (data['pi'].isin(selected_pis)) &
        (data['date_time'] >= pd.to_datetime(start_date)) &
        (data['date_time'] <= pd.to_datetime(end_date))
    ]
    # Convert the 'date_time' column to datetime
    resampled_data['date_time'] = pd.to_datetime(resampled_data['date_time'])
    # Set 'date_time' as the index
    resampled_data.set_index('date_time', inplace=True)
    # Group by 'pi' and resample
    resampled_data = resampled_data.groupby('pi').resample(frequency).mean(numeric_only=True)
    # Reset the index
    resampled_data.reset_index(inplace=True)
    return resampled_data

# Function to generate date marks for range slider
def get_date_marks(start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq='MS')
    date_marks = {int((date - start_date).days): str(date.date()) for date in dates}
    date_marks[(end_date - start_date).days] = str(end_date.date())
    return date_marks


# Functions for the different chart types
def update_line_chart(selected_cell, selected_pis, date_range, frequency):
    """Updates the line chart"""
    resampled_data = resample_data(selected_cell, selected_pis, date_range, frequency)
    resampled_data = resampled_data.sort_values(by=['date_time'])
    fig = go.Figure()
    for i, pi in enumerate(selected_pis):
        pi_data = resampled_data[resampled_data['pi'] == pi]
        pi_data = pi_data.dropna()
        fig.add_trace(go.Scatter(x=pi_data['date_time'].dt.strftime('%Y-%m-%d %H'), y=pi_data['value'], mode='lines+markers+text', name=pi, line=dict(color=line_colors[i % len(line_colors)]),
                                 text=["{:.2f}".format(val) for val in pi_data['value']],
                                 textposition='top center',
                                 textfont=dict(color="#FFFFFF", size=10)))
    fig.update_layout(
        title=f'Time Series Line Chart for {selected_cell}',
        xaxis_title='Time',
        yaxis_title='Value',
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def update_bar_chart(selected_cell, selected_pis, date_range, frequency):
    """Updates the bar chart"""
    resampled_data = resample_data(selected_cell, selected_pis, date_range, frequency)
    fig = go.Figure()

    for i, pi in enumerate(selected_pis):
        pi_data = resampled_data[resampled_data['pi'] == pi]
        pi_data = pi_data.dropna()  # make sure to drop NaN values
        fig.add_trace(go.Bar(x=pi_data['date_time'].dt.strftime('%Y-%m-%d %H'), y=pi_data['value'], 
                             name=pi, marker=dict(color=line_colors[i % len(line_colors)]),
                             text=["{:.2f}".format(val) for val in pi_data['value']],
                             textposition='outside',
                             textfont=dict(color="#FFFFFF", size=10)))
    
    fig.update_layout(
        title=f'Time Series Bar Chart for {selected_cell}', 
        xaxis_title='Time', 
        yaxis_title='Value',
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig



def update_scatter_chart(selected_cell, selected_pis, date_range, frequency):
    """Updates the scatter chart"""
    resampled_data = resample_data(selected_cell, selected_pis, date_range, frequency)
    resampled_data = resampled_data.dropna()
    fig = go.Figure()

    if len(selected_pis) < 2:
        fig.update_layout(title='Scatter Plot', template='plotly_dark')
        return fig

    pi_data_x = resampled_data[resampled_data['pi'] == selected_pis[0]]
    pi_data_y = resampled_data[resampled_data['pi'] == selected_pis[1]]

    # Normalize the size data for better visualization
    size = (pi_data_y['value'] - pi_data_y['value'].min()) / (pi_data_y['value'].max() - pi_data_y['value'].min()) * 20

    fig.add_trace(go.Scatter(
        x=pi_data_x['value'], 
        y=pi_data_y['value'], 
        mode='markers',
        name=f'{selected_pis[0]} vs {selected_pis[1]}',
        marker=dict(
            size=size,  # Use normalized size
            color=pi_data_x['value'],  # Use the value of the first PI for color
            colorscale='Viridis',  # Set color scale
            colorbar=dict(title="Value of PI")  # Set color bar title
        )
    ))

    fig.update_layout(
        title=f'Scatter Plot for {selected_cell} between {selected_pis[0]} and {selected_pis[1]}', 
        xaxis_title=selected_pis[0], 
        yaxis_title=selected_pis[1],
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def update_heatmap(selected_cell, selected_pis, date_range, frequency):
    """Updates the heatmap"""
    resampled_data = resample_data(selected_cell, selected_pis, date_range, frequency)
    resampled_data = resampled_data.dropna()

    # Scale the vertical spacing based on the number of rows
    vertical_spacing = 1.0 / (len(selected_pis) * 10)

    # Create subplots, using 'domain' type for y-axes
    fig = make_subplots(rows=len(selected_pis), cols=1, vertical_spacing=vertical_spacing)

    for i, pi in enumerate(selected_pis):
        pi_data = resampled_data[resampled_data['pi'] == pi]
        pi_data = pi_data.pivot(index='pi', columns='date_time', values='value')
        heatmap = go.Heatmap(
            z=pi_data.values,
            x=pi_data.columns.strftime('%Y-%m-%d %H:%M'),  # include hours and minutes in the x-axis
            # y=pi_data.index,
            colorscale='YlOrRd',  # using a yellow-orange-red gradient
            colorbar=dict(title="Value", len=1/len(selected_pis), y=(i*1/len(selected_pis)) + 1/len(selected_pis)/2),
            hoverongaps=False,
            hoverinfo='x+y+z',
            name=pi
        )
        fig.add_trace(heatmap, i+1, 1)

        # Adding y-axes title with vertical alignment
        fig.update_yaxes(title_text=pi, row=i+1, col=1, title_standoff=0)

    fig.update_layout(
        title=f'Heatmap for {selected_cell}',
        template='plotly_dark',
        showlegend=False,
        autosize=True,
        height=360* len(selected_pis),  # Adjust the height based on the number of PIs
    )

    return fig

def update_box_plot(selected_cell, selected_pis, date_range, frequency):
    """Updates the box plot"""
    resampled_data = resample_data(selected_cell, selected_pis, date_range, frequency)
    resampled_data = resampled_data.dropna()
    fig = go.Figure()
    for i, pi in enumerate(selected_pis):
        pi_data = resampled_data[resampled_data['pi'] == pi]
        fig.add_trace(go.Box(y=pi_data['value'], name=pi, hoverinfo='y', marker_color=line_colors[i % len(line_colors)]))
    fig.update_layout(
        title=f'Box Plot for {selected_cell}',
        xaxis_title='Performance Indicator',
        yaxis_title='Value',
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def update_histogram(selected_cell, selected_pis, date_range, frequency):
    """Updates the histogram chart"""
    resampled_data = resample_data(selected_cell, selected_pis, date_range, frequency)
    resampled_data = resampled_data.dropna()

    # Create subplots, using 'domain' type for x-axes
    fig = make_subplots(rows=1, cols=len(selected_pis), 
                        shared_yaxes=True)

    for i, pi in enumerate(selected_pis):
        pi_data = resampled_data[resampled_data['pi'] == pi]
        fig.add_trace(go.Histogram(x=pi_data['value'], 
                                   name=pi, 
                                   nbinsx=20, 
                                   marker_color=line_colors[i % len(line_colors)]), 
                      row=1, col=i+1)
        fig.update_xaxes(title_text='Value', row=1, col=i+1)

    fig.update_layout(title=f'{selected_cell} Histogram for each PI', 
                      yaxis_title='Count', 
                      template='plotly_dark', 
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1
                      ))

    return fig

# Dictionary to map tab values to chart functions
chart_func_dict = {
    'tab-line': update_line_chart,
    'tab-bar': update_bar_chart,
    'tab-scatter': update_scatter_chart,
    'tab-heatmap': update_heatmap,
    'tab-box': update_box_plot,
    'tab-hist': update_histogram,
}