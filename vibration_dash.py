#https://towardsdatascience.com/dash-for-beginners-create-interactive-python-dashboards-338bfcb6ffa4
import pandas as pd
from Func_V2 import *
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px

from dash.dependencies import Input, Output

DATA_PATH = 'D:\\기계시설물 고장 예지 센서\\Training'
os.chdir(DATA_PATH)
tp = 'vibration'
kw = '15'
machine = 'R-CAHU-03S'
state = '정상'
path,file_names = detect_file_name(tp, kw, machine, state)
file = file_names[0]

normal = load_vibration_data(path,file)

ab_state = '벨트느슨함'
path,file_names = detect_file_name(tp, kw, machine,ab_state)
file = file_names[0]
abnormal = load_vibration_data(path,file)

app = dash.Dash()   #initialising dash app

def vibration_plot(normal_df,tp,dropdown1_value,machine,state,n):
    path,file_names = detect_file_name(tp, dropdown1_value, machine,state)
    file = file_names[n]
    file_len = len(file_names)
    normal_df = load_vibration_data(path,file)

    # Function for creating line chart showing Google stock prices over time 
    fig1 = go.Figure([go.Scatter(x = normal_df['time'], y = normal_df['vibration'],\
                     line = dict(color = 'firebrick', width = 4), name = f'{tp}_{dropdown1_value}_{machine}_{state}')
                     ])
    fig1.update_layout(title = f'{tp}_{dropdown1_value}_{machine}_{state}',
                     xaxis_title = 'time',
                      yaxis_title = 'vibration'
                      )
    return fig1


app.layout = html.Div(
    id = 'parent', 
    children = [
        html.H1(id = 'H1', children = f'Vibration Data Plot 3sec n=12000', 
        style = {'textAlign':'center','marginTop':40,'marginBottom':40}),

        #dcc.Dropdown1( id = 'dropdown1', options = []),
        dcc.Graph(id = 'line_plot1', figure = graph_update(dropdown1_value)),#vibration_plot(normal,kw,machine,state,10)),
        dcc.Dropdown( id = 'dropdown1', 
        options = [
            {'label':'2.2kW','value':'2.2'},
            {'label':'3.7kW','value':'3.7'},
            {'label':'3.75kW','value':'3.75'},
            {'label':'5.5kW','value':'5.5'},
            {'label':'7.5kW','value':'7.5'},
            {'label':'11kW','value':'11'},
            {'label':'15kW','value':'15'},
            {'label':'18.5kW','value':'18.5'},
            {'label':'22kW','value':'22'},
        ],
         placeholder="Select kW"),
        #dcc.Graph(id = 'line_plot2', figure = vibration_plot(abnormal,tp,dropdown1_value,machine,ab_state,10))
        ])
@app.callback(Output(component_id='line_plot1', component_property= 'figure'),[Input(component_id='dropdown1', component_property= 'value')])

def graph_update(dropdown1_value):
    path,file_names = detect_file_name(tp, dropdown1_value, machine,state)
    file = file_names[1]
    file_len = len(file_names)
    normal_df = load_vibration_data(path,file)
    print(dropdown1_value)
    fig1 = go.Figure([go.Scatter(x = normal_df['time'], y = normal_df['vibration'],\
                     line = dict(color = 'firebrick', width = 4), name = f'{tp}_{dropdown1_value}_{machine}_{state}')
                     ])
    fig1.update_layout(title = f'{tp}_{dropdown1_value}_{machine}_{state}',
                      xaxis_title = 'time',
                      yaxis_title = 'vibration'
                      )
    return fig1  



#vibration_plot(normal,tp,kw,machine,state)


if __name__ == '__main__': 
    app.run_server()