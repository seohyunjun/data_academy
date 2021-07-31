import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output


app = dash.Dash()

df = normal


app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'Styling using html components', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),

        dcc.Dropdown( id = 'dropdown',
        options = [
            {'label':'10', 'value':10 },
            {'label': '9', 'value':9},
            {'label': '8', 'value':8},
            ],
        value = '0'),
        dcc.Graph(id = 'bar_plot')
    ])
    
    
@app.callback(Output(component_id='bar_plot', component_property= 'figure'),
              [Input(component_id='dropdown', component_property= 'value')])
def graph_update(path,file_names,dropdown_value):
    print(dropdown_value)
    file = file_names[int(dropdown_value)]
    normal = load_vibration_data(path,file)

    fig = go.Figure([go.Scatter(x = df['time'], y = df['vibration'],\
                     line = dict(color = 'firebrick', width = 4))
                     ])
    
    fig.update_layout(title = 'Stock prices over time',
                      xaxis_title = 'Dates',
                      yaxis_title = 'Prices'
                      )
    return fig  



if __name__ == '__main__': 
    app.run_server()