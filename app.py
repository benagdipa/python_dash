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
from functions import *


# Initialize Dash app with Bootstrap
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY]) 

# Define Dash app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("5G NR KPI Dashboard", 
                     className="text-center my-4 title"),  # add a new class 'title'
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Control Panel", className="mb-3"),
                dbc.CardBody([
                    html.H6("Select Cell", className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='cell_dropdown',
                                options=[{'label': cell, 'value': cell} for cell in cell_ids],
                                value=cell_ids[0],
                                multi=False,
                                clearable=False,
                                className="mb-4"
                            ),
                        ]),
                    ]),
                    html.H6("Select KPI", className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='pi_dropdown',
                                options=[{'label': pi, 'value': pi} for pi in pis],
                                value=pis[0],
                                multi=True,
                                clearable=False,
                                className="mb-4"
                            ),
                        ]),
                    ]),
                    html.H6("Select time range", className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dcc.DatePickerRange(
                                id='date_picker',
                                min_date_allowed=min_date,
                                max_date_allowed=max_date,
                                start_date=min_date,
                                end_date=max_date,
                                className="mb-4"
                            ),
                        ]),
                    ]),
                    html.H6("Select date resampling", className="mb-4"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='resample_dropdown',
                                options=[{'label': 'Hourly', 'value': 'H'},
                                         {'label': 'Daily', 'value': 'D'},
                                         {'label': 'Weekly', 'value': 'W'}],
                                value='D',
                                className="mb-4"
                            ),
                        ]),
                    ]),
                ]),
            ], className="mb-4"),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dbc.Tabs(
                        id="tabs",
                        active_tab='tab-line',
                        children=[
                            dbc.Tab(label='Line Chart', tab_id='tab-line'),
                            dbc.Tab(label='Bar Chart', tab_id='tab-bar'),
                            dbc.Tab(label='Scatter Chart', tab_id='tab-scatter'),
                            dbc.Tab(label='Heatmap', tab_id='tab-heatmap'),
                            dbc.Tab(label='Box Plot', tab_id='tab-box'),
                            dbc.Tab(label='Histogram', tab_id='tab-hist'),
                        ]),

                    className='card-header'
                ),
                dbc.CardBody([
                    html.Div(id='tabs-content',
                             children=dcc.Loading(
                                 type="default",
                                 children=dcc.Graph(
                                     figure=update_line_chart(cell_ids[0], [pis[0]], (min_date, max_date), 'D'),
                                     config={'displayModeBar': False},
                                     id='graph'
                                 ))
                             )
                ]),
                dbc.CardBody([
                    html.Div(id='summary-table-container'),
                ]),
            ], className="mb-4"),
        ], width=9),
    ]),
], fluid=True)


@app.callback(
    Output('graph', 'figure'),
    [Input('cell_dropdown', 'value'),
    Input('pi_dropdown', 'value'),
    Input('date_picker', 'start_date'),
    Input('date_picker', 'end_date'),
    Input('tabs', 'active_tab'),  # change 'value' to 'active_tab'
    Input('resample_dropdown', 'value')])
def update_graph(cell_id, pis, start_date, end_date, tab, resample_freq):
    if tab == 'tab-line':
        return update_line_chart(cell_id, pis, (start_date, end_date), resample_freq)
    elif tab == 'tab-bar':
        return update_bar_chart(cell_id, pis, (start_date, end_date), resample_freq)
    elif tab == 'tab-scatter':
        return update_scatter_chart(cell_id, pis, (start_date, end_date), resample_freq)
    elif tab == 'tab-heatmap':
        return update_heatmap(cell_id, pis, (start_date, end_date), resample_freq)
    elif tab == 'tab-box':
        return update_box_plot(cell_id, pis, (start_date, end_date), resample_freq)
    elif tab == 'tab-hist':
        return update_histogram(cell_id, pis, (start_date, end_date), resample_freq)

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'active_tab'),  # change 'value' to 'active_tab'
    Input('cell_dropdown', 'value'),
    Input('pi_dropdown', 'value'),
    Input('date_picker', 'start_date'),
    Input('date_picker', 'end_date'),
    Input('resample_dropdown', 'value'))
def render_tab_content(tab, selected_cells, selected_pis, start_date, end_date, resample_freq):
    # Convert date strings to datetime objects
    start_date = pd.to_datetime(start_date).replace(hour=0, minute=0)
    end_date = pd.to_datetime(end_date).replace(hour=23, minute=59)
    # Check if any selected item is None or if selected_pis is an empty list
    if None in (selected_cells, selected_pis, start_date, end_date) or not selected_pis:
        # If yes, return an empty figure
        return [dcc.Graph(figure=go.Figure())]
    else:
        # Ensure selected_cells is a list
        if not isinstance(selected_cells, list):
            selected_cells = [selected_cells]
        # Create a graph for each selected cell
        content = []
        for selected_cell in selected_cells:
            fig = chart_func_dict[tab](selected_cell, selected_pis, (start_date, end_date), resample_freq)  # Include resample_freq
            content.append(dcc.Graph(figure=fig))
            content.append(html.Hr())
        return content

    
# Callback for updating the summary table
@app.callback(
    Output('summary-table-container', 'children'),
    [Input('cell_dropdown', 'value'),
    Input('pi_dropdown', 'value'),
    Input('date_picker', 'start_date'),
    Input('date_picker', 'end_date'),
    Input('resample_dropdown', 'value')])
def update_summary_tables(selected_cells, selected_pis, start_date, end_date, resample_freq):
    # Ensure selected_pis and selected_cells are lists
    if not isinstance(selected_pis, list):
        selected_pis = [selected_pis]
    if not isinstance(selected_cells, list):
        selected_cells = [selected_cells]

    table_list = []
    for pi in selected_pis:
        for cell in selected_cells:
            # Filter the DataFrame for the current 'pi' and 'cell'
            filtered_data = data[(data['pi'] == pi) & (data['cell_id'] == cell)]
            filtered_data = filtered_data[(filtered_data['date_time'] >= pd.to_datetime(start_date)) & (filtered_data['date_time'] <= pd.to_datetime(end_date))]

            # set 'date_time' as the index
            filtered_data.set_index('date_time', inplace=True)

            # Resample filtered data before generating the summary
            filtered_data_resampled = filtered_data.resample(resample_freq).mean()


            summary = filtered_data_resampled['value'].describe().round(2)
            summary_table = dash_table.DataTable(
                id=f'summary-table-{pi}-{cell}',
                columns=[{"name": i, "id": i} for i in summary.index],
                data=summary.to_frame().T.to_dict('records'),
                style_cell={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white', 'textAlign': 'center'},
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'textAlign': 'center'},
                style_table={'width': '100%'},  # make the table width same as chart
            )
            table_list.append(html.H5(f'Statistical Summary for {pi} in {cell}'))
            table_list.append(summary_table)
            table_list.append(html.Br())  # add a gap between tables

    return table_list

if __name__ == '__main__':
    app.run_server(debug=True)
