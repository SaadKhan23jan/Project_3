from datetime import datetime as dt
import time
import os
import dash
from dateutil.relativedelta import relativedelta
from dash import Dash, html, dcc, dash_table, DiskcacheManager, CeleryManager
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from crypto_list import crypto_list
from functions import sarimax_pred, decomposition

import warnings
warnings.filterwarnings("ignore")

if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

css_sheet = [dbc.themes.SKETCHY]
BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"

app = Dash(__name__, external_stylesheets=css_sheet, background_callback_manager=background_callback_manager)
server = app.server
app.title = 'Live Market Analysis'
p_alert = dbc.Alert("Invalid inputs, please check the Starting p,i,d and Maximum p, i, d values"
                    "(maximums should be greater than the starting values)",
                    id="alert",
                    ),

app.layout = html.Div([


    dbc.Button('Contact Me: LinkedIn', href='https://www.linkedin.com/in/saad-khan-167704163/', target='_blank',
               style={'position': 'center'}),

    dbc.Button('Click Here for Stock Markets', href='http://127.0.0.1:8050/stock_markets.py', target='_blank',
               disabled=True),
    html.Div(
        html.H1("Welcome to Live Crypto/Stock Market Data and Predictions", style={'textAlign': 'center',
                                                                                   'backgroundColor': 'Lightgreen'})
    ),

    html.Div([
        html.Label('Select Crypto-Pair/Stock Market'),
        dcc.Dropdown(id='crypto-pair', options=crypto_list,
                     style={'width': '50%'}, value='BTC-USD'),
    ]),

    html.Div([
        html.Label('Select Time Frame'),
        dcc.Dropdown(id='time-frame', options=[{'label': '10 Year', 'value': '10y'},
                                               {'label': '1 Year', 'value': '1y'},
                                               {'label': '6 Months', 'value': '6mo'},
                                               {'label': '3 Months', 'value': '3mo'},
                                               {'label': '1 Months', 'value': '1mo'},
                                               {'label': '1 Week', 'value': '1wk'},
                                               {'label': '1 Day', 'value': '24h'}],
                     style={'width': '50%'}, value='1y'),
    ]),

    html.Div(
        dcc.Graph(id='graph-candlestick')
    ),

    html.Div(
        dcc.Graph(id='graph-line'),
    ),

    html.Div(
        dcc.Graph(id='decomposed-graphs')
    ),

    html.Div(
        dcc.Graph(id='volume-graph-line')
    ),

    html.Div([
        html.H2("SARIMAX Models Predictions"),
        html.Br(),

        html.Label('Select one of SARIMAX Model:   '),
        dcc.Dropdown(id='sarimax-model', options=[{'label': 'MA', 'value': 'MA'},
                                                  {'label': 'AR', 'value': 'AR'},
                                                  {'label': 'ARMA', 'value': 'ARMA'},
                                                  {'label': 'ARIMA', 'value': 'ARIMA'},
                                                  {'label': 'SARIMAX', 'value': 'SARIMAX'},
                                                  {'label': 'Auto ARIMA', 'value': 'Auto ARIMA'}
                                                  ],
                     style={'width': '50%'},
                     value='ARIMA'
                     ),
        html.Br(),

        html.Div(
            id='auto-arima-container',
            children=html.Div([
                html.Div([
                    html.Div([
                        html.Label('Starting value of p:', style={'paddingRight': '20px'}),
                        dcc.Input(id='start-p-order', type='number', value=0, inputMode='numeric', min=0, )
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                    html.Div([
                        html.Label('Starting value of i:', style={'paddingRight': '20px'}),
                        dcc.Input(id='start-i-order', type='number', value=0, inputMode='numeric', min=0)
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                    html.Div([
                        html.Label('Starting value of q:', style={'paddingRight': '20px'}),
                        dcc.Input(id='start-q-order', type='number', value=0, inputMode='numeric', min=0)
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),
                    html.Div([
                        html.Label('Maximum value of p:', style={'paddingRight': '20px'}),
                        dcc.Input(id='max-p-order', type='number', value=1, inputMode='numeric', min=1),
                        html.Div(id='error-max-p', children=[]),
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                    html.Div([
                        html.Label('Maximum value of i:', style={'paddingRight': '20px'}),
                        dcc.Input(id='max-i-order', type='number', value=1, inputMode='numeric', min=1),
                        html.Div(id='error-max-i'),
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                    html.Div([
                        html.Label('Maximum value of q:', style={'paddingRight': '20px'}),
                        dcc.Input(id='max-q-order', type='number', value=1, inputMode='numeric', min=1),
                        html.Div(id='error-max-q'),
                    ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},
                    ),

                ], style={'display': 'flex'}),
                html.Br(),
                html.Div([
                    html.Div([

                        html.Label('Select the Seasonal Factor', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto-arima-seasonal-factor',
                                     options=[{'label': 'Quarterly', 'value': 3},
                                              {'label': '4-Monthly', 'value': 4},
                                              {'label': 'Bi-Yearly', 'value': 6},
                                              {'label': 'Yearly', 'value': 12}, ],
                                     value=12),
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False),

                    html.Div([
                        html.Label('Seasonal', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto-arima-seasonal',
                                     options=[{'label': 'True', 'value': 'True'},
                                              {'label': 'False', 'value': 'False'}, ],
                                     value='False'),
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},),
                    html.Div([
                        html.Label('Stationary', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto-arima-stationary',
                                     options=[{'label': 'True', 'value': 'True'},
                                              {'label': 'False', 'value': 'False'}, ],
                                     value='False'),
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},),

                    html.Div([
                        html.Label('Information Criteria', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto-arima-information-criterion',
                                     options=[{'label': 'AIC', 'value': 'aic'},
                                              {'label': 'BIC', 'value': 'bic'},
                                              {'label': 'HQIC', 'value': 'hqic'},
                                              {'label': 'OOB', 'value': 'oob'}, ],
                                     value='aic'),
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},),

                    html.Div([
                        html.Label('Method to be used', style={'paddingRight': '20px'}),
                        dcc.Dropdown(id='auto-arima-method',
                                     options=[{'label': 'Newton-Raphson', 'value': 'newton'},
                                              {'label': 'Nelder-Mead', 'value': 'nm'},
                                              {'label': ' Broyden-Fletcher-Goldfarb-Shanno (BFGS)', 'value': 'bfgs'},
                                              {'label': 'limited-memory BFGS with optional box constraints', 'value': 'lbfgs'},
                                              {'label': 'modified Powell’s method', 'value': 'powell'},
                                              {'label': 'conjugate gradient', 'value': 'cg'},
                                              {'label': 'Newton-conjugate gradient', 'value': 'ncg'},
                                              {'label': 'global basin-hopping solver', 'value': 'basinhopping'}, ],
                                     value='lbfgs')
                    ], style={'width': '300px', 'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                              'paddingRight': '10px', 'paddingTop': '15px'},)
                ], style={'display': 'flex'}),

            ]),
            hidden=True,
        ),

        html.Div(
            id='sarimax-container',
            children=html.Div([
                html.Div([
                    html.Label('Enter the order of P:', style={'paddingRight': '20px'}),
                    dcc.Input(id='sarimax-p-order', type='number', value=0, inputMode='numeric', min=0, )
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'},
                ),

                html.Div([
                    html.Label('Enter the order of I:', style={'paddingRight': '20px'}),
                    dcc.Input(id='sarimax-i-order', type='number', value=0, inputMode='numeric', min=0)
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'},
                ),

                html.Div([
                    html.Label('Enter the order of Q:', style={'paddingRight': '20px'}),
                    dcc.Input(id='sarimax-q-order', type='number', value=0, inputMode='numeric', min=0, )
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'},
                ),

                html.Div([
                    html.Label('Select the Seasonal Factor'),
                    dcc.Dropdown(id='seasonal-factor', options=[{'label': 'Quarterly', 'value': 3},
                                                                {'label': '4-Monthly', 'value': 4},
                                                                {'label': 'Bi-Yearly', 'value': 6},
                                                                {'label': 'Yearly', 'value': 12}, ],
                                 value=12),
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False),

            ], style={'display': 'flex'}),
            hidden=True,
        ),

        html.Br(),
        html.Br(),

        html.Div([
            html.Div(
                id='p-order-container',
                children=[
                    html.Label('Enter the order of p:', style={'paddingRight': '20px'}),
                    dcc.Input(id='p-order', type='number', placeholder='Enter the order of P', value=0,
                              inputMode='numeric', min=0, required=True),
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False,
            ),

            html.Div(
                id='i-order-container',
                children=[
                    html.Label('Enter the order of i:', style={'paddingRight': '20px'}),
                    dcc.Input(id='i-order', type='number', placeholder='Enter the order of I', value=0,
                              inputMode='numeric', min=0),
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False,
            ),

            html.Div(
                id='q-order-container',
                children=[
                    html.Label('Enter the order of q:', style={'paddingRight': '20px'}),
                    dcc.Input(id='q-order', type='number', placeholder='Enter the order of Q', value=0,
                              inputMode='numeric', min=0),
                ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                          'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False,
            ),
            html.Div([
                html.Label('Days for Forecast:', style={'paddingRight': '20px'}),
                dcc.Input(id='days', type='number', placeholder='Enter the order of Q', value=14,
                          inputMode='numeric', min=0,),
            ], style={'backgroundColor': '#f3f2f5', 'borderRadius': '10px', 'marginLeft': '10px',
                      'paddingRight': '10px', 'paddingTop': '15px'}, hidden=False
            ),

        ], style={'display': 'flex'}),

        html.Br(),

        html.Div([
            html.Div([html.P(id='run-calc', children=['Calculations are not yet Started'])]),
            html.Button(id='run-pred', children='Run Forecast',
                        style={'weight': 'bold', 'height': '100px', 'borderRadius': '30px', 'paddingRight': '20px',
                               'fontWeight': 'bold', 'fontSize': '20px'}),
            html.Button(id='stop-pred', children='Stop Forecast',
                        style={'weight': 'bold', 'height': '100px', 'borderRadius': '30px', 'paddingLeft': '20px',
                               'fontWeight': 'bold', 'fontSize': '20px'}),

        ]
        ),


        html.Br(),
        html.Br(),

        # dash_table.DataTable(id='sarimax-results',columns =  [{"name": i, "id": i,} for i in (df.columns)],),
        html.Div(id='sarimax-results'),

    ]),


    html.Br(),
    html.Br(),
    html.Br(),
    html.Div([
        html.H1([f'Predictions Through ', html.Label(id='model_used', style={'color': 'blue', 'weight': 'bold'}),
                 ' Model']),

    ], style={'borderRadius': '20px', 'width': '800px',
              'backgroundColor': '#e6f5f4', 'padding': '30px'}),


    dcc.Graph(id='fig-pred'),

    html.P('***Note***: This is for demo purpose and the results are never claimed to be correct, nor can be used'
           ' for real time prediction of the actual data', style={'color': 'white', 'backgroundColor': 'black'}),



], style={'background-color': 'Lightgreen'})


@dash.callback(
    output=Output('run-calc', 'children'),
    inputs=Input('run-pred', 'n_clicks'),
    background=True,
    running=[
        (Output('run-pred', 'disabled'), True, False),
        (Output('stop-pred', 'disabled'), False, True),
    ],
    cancel=[Input('stop-pred', 'n_clicks')],
)
def run_calculations(n_clicks):
    time.sleep(2.0)
    return [f"Click on Run to Start Calculations or Stop to Cancel the Calculations or Refresh the page"]


@app.callback([Output('error-max-p', 'children'),
               Output('error-max-i', 'children'),
               Output('error-max-q', 'children'), ],
              [Input('max-p-order', 'value'),
               Input('max-i-order', 'value'),
               Input('max-q-order', 'value'),
               Input('start-p-order', 'value'),
               Input('start-i-order', 'value'),
               Input('start-q-order', 'value'), ]
              )
def error_alert(max_p_order, max_i_order, max_q_order, start_p_order,  start_i_order,  start_q_order):

    # This code is short form, else it can be written in 6 lines of if-else statements
    alert_p, alert_i, alert_q, = ('', '', '')

    if max_p_order <= start_p_order:
        alert_p = 'Should be greater than Starting p value'
    if max_i_order <= start_i_order:
        alert_i = 'Should be greater than Starting i value'
    if max_q_order <= start_q_order:
        alert_q = 'Should be greater than Starting q value'

    return alert_p, alert_i, alert_q


@app.callback([Output('auto-arima-container', 'hidden'),
               Output('sarimax-container', 'hidden'),
               Output('p-order-container', 'hidden'),
               Output('i-order-container', 'hidden'),
               Output('q-order-container', 'hidden'), ],
              Input('sarimax-model', 'value'),
              prevent_initial_call=True)
def update_output(sarimax_model):
    if sarimax_model == 'SARIMAX':
        return True, False, False, False, False
    elif sarimax_model == 'MA':
        return True, True, True, True, False
    elif sarimax_model == 'AR':
        return True, True, False, True, True
    elif sarimax_model == 'ARMA':
        return True, True, False, True, False
    elif sarimax_model == 'ARIMA':
        return True, True, False, False, False
    elif sarimax_model == 'Auto ARIMA':
        return False, True, True, True, True
    else:
        return False, False, True, True, True


@app.callback([Output('graph-candlestick', 'figure'),
               Output('graph-line', 'figure'),
               Output('volume-graph-line', 'figure'),
               Output('decomposed-graphs', 'figure')],
              [Input('crypto-pair', 'value'),
               Input('time-frame', 'value'),
               ])
def update_graph(crypto, time_frame):

    if time_frame in ['10y']:
        interval = '1mo'
        start = (dt.now()-relativedelta(years=10))
    elif time_frame in ['1y']:
        interval = '1wk'
        start = (dt.now() - relativedelta(years=1))
    elif time_frame in ['6mo']:
        interval = '5d'
        start = (dt.now() - relativedelta(months=6))
    elif time_frame in ['3mo']:
        interval = '1d'
        start = (dt.now() - relativedelta(months=3))
    elif time_frame in ['1mo']:
        interval = '90m'
        start = (dt.now() - relativedelta(months=1))
    elif time_frame in ['1wk']:
        interval = '30m'
        start = (dt.now() - relativedelta(weeks=1))
    elif time_frame in ['24h']:
        interval = '1m'
        start = (dt.now() - relativedelta(hours=23))
    else:
        interval = '1m'
        start = dt.now() - relativedelta(days=1)

    end = dt.now()

    # df = yf.download(tickers=crypto, period=time_frame, interval=interval)
    df = yf.download(tickers=crypto, start=start, end=end, interval=interval)

    fig1 = go.Figure(data=[go.Candlestick(x=df.index,
                                          open=df.Open,
                                          high=df.High,
                                          low=df.Low,
                                          close=df.Close)])
    fig1.update_layout(title=f'Candle Chart of {crypto}', xaxis_title='Time', yaxis_title=f'{crypto}')

    fig2 = px.line(data_frame=df, x=df.index, y=[df['Open'], df['High'], df['Low'], df['Close']])
    fig2.update_layout(title=f'History of Price {crypto}', xaxis_title='Time', yaxis_title=f'Price History of {crypto}')

    fig3 = px.line(data_frame=df, x=df.index, y=df['Volume'], markers='o')
    fig3.update_layout(title=f'History of Volume {crypto}', xaxis_title='Time', yaxis_title=f'Volume of {crypto}')

    fig_seasonality_decompose = decomposition(df)

    return fig1, fig2, fig3, fig_seasonality_decompose


@app.callback([Output('sarimax-results', 'children'),
               Output('fig-pred', 'figure'),
               Output('model_used', 'children'), ],
              [Input('run-pred', 'n_clicks'),
               State('time-frame', 'value'),
               State('crypto-pair', 'value'),
               State('p-order', 'value'),
               State('i-order', 'value'),
               State('q-order', 'value'),
               State('sarimax-model', 'value'),
               State('days', 'value'),
               State('sarimax-p-order', 'value'),
               State('sarimax-i-order', 'value'),
               State('sarimax-q-order', 'value'),
               State('seasonal-factor', 'value'),
               State('start-p-order', 'value'),
               State('start-i-order', 'value'),
               State('start-q-order', 'value'),
               State('max-p-order', 'value'),
               State('max-i-order', 'value'),
               State('max-q-order', 'value'),
               State('auto-arima-seasonal-factor', 'value'),
               State('auto-arima-seasonal', 'value'),
               State('auto-arima-stationary', 'value'),
               State('auto-arima-information-criterion', 'value'),
               State('auto-arima-method', 'value'),

               ])
def predictions(n_clicks, time_frame, crypto, p, i, q, sarimax_model, days, sp, si, sq, seasonal_factor,
                start_p_order, start_i_order, start_q_order, max_p_order, max_i_order, max_q_order,
                auto_arima_seasonal_factor, auto_arima_seasonal, auto_arima_stationary,
                auto_arima_information_criterion, auto_arima_method):

    if time_frame == '10y':
        interval = '90m'
        start = (dt.now()-relativedelta(years=10))
    elif time_frame in ['1y']:
        interval = '1wk'
        start = (dt.now() - relativedelta(years=1))
    elif time_frame in ['6mo']:
        interval = '5d'
        start = (dt.now() - relativedelta(months=6))
    elif time_frame in ['3mo']:
        interval = '1d'
        start = (dt.now() - relativedelta(months=3))
    elif time_frame in ['1mo']:
        interval = '90m'
        start = (dt.now() - relativedelta(months=1))
    elif time_frame in ['1wk']:
        interval = '30m'
        start = (dt.now() - relativedelta(weeks=1))
    elif time_frame in ['24h']:
        interval = '1m'
        start = (dt.now() - relativedelta(hours=23))
    else:
        interval = '1m'
        start = dt.now() - relativedelta(days=1)

    end = dt.now()

    # df = yf.download(tickers=crypto, period=time_frame, interval=interval)
    df = yf.download(tickers=crypto, start=start, end=end, auto_adjust=True)

    # Here we will call our function for SARIMAX Model

    if auto_arima_seasonal == 'True':
        auto_arima_seasonal = True
    else:
        auto_arima_seasonal = False

    if auto_arima_stationary == 'True':
        auto_arima_stationary = True
    else:
        auto_arima_stationary = False

    results, pred_fig = sarimax_pred(df, crypto, p, i, q, sarimax_model, days, sp, si, sq, seasonal_factor,
                                     start_p_order, start_i_order, start_q_order, max_p_order, max_i_order, max_q_order,
                                     auto_arima_seasonal_factor, auto_arima_seasonal, auto_arima_stationary,
                                     auto_arima_information_criterion, auto_arima_method)

    return results, pred_fig, sarimax_model


if __name__ == '__main__':
    app.run_server(debug=True)
