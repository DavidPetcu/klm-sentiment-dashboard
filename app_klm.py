import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import json

with open('all_tweets_hybrid_sentiment.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

def score_to_label(score):
    if score > 0.2:
        return 'positive'
    elif score < -0.2:
        return 'negative'
    else:
        return 'neutral'

if 'sentiment' not in df.columns or df['sentiment'].isnull().all():
    df['sentiment'] = df['score'].apply(score_to_label)

app = Dash(__name__)
app.title = "KLM Sentiment Dashboard"

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src="/assets/klmlogo.png", style={'height': '60px', 'marginRight': '20px'}),
            html.Img(src="/assets/twitterlogo.png.png", style={'height': '60px'})
        ], style={'display': 'flex', 'align-items': 'center'}),

        html.Div([
            html.H1("KLM Tweet Sentiment Dashboard", style={
                'margin-top': '10px',
                'fontSize': '30px',
                'fontWeight': 'bold',
                'color': '#1a1a1a',
                'marginRight': '30px'
            }),
            html.Div("Developed By SkySense", style={
                'fontSize': '14px',
                'color': '#444',
                'fontWeight': '600',
                'letterSpacing': '0.5px',
                'alignSelf': 'center'
            })
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'})
    ], style={
        'padding': '20px',
        'backgroundColor': '#e3eaf2',
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'width': '100%',
        'zIndex': '1000',
        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'
    }),

    html.Div([
        html.Label("Filter by Airline:", style={
            'fontWeight': 'bold',
            'marginBottom': '10px',
            'display': 'block',
            'fontSize': '16px'
        }),
        dcc.Dropdown(
            options=[
                {"label": "All Airlines", "value": "all"},
                {"label": "KLM", "value": "klm"}
            ],
            value='all',
            id='airline-filter',
            clearable=False,
            style={'width': '300px'}
        )
    ], style={'padding': '20px'}),

    html.Div([
        html.Label("Display Mode:"),
        dcc.Checklist(
            options=[{'label': 'Dark Mode', 'value': 'dark'}],
            value=[],
            id='mode-toggle',
            inputStyle={'margin-right': '5px'}
        )
    ], style={'padding': '0 20px 20px 20px'}),

    html.Div([
        dcc.Graph(id='bar-chart', config={'toImageButtonOptions': {'format': 'png'}}, style={'boxShadow': '0 4px 8px rgba(0,0,0,0.05)', 'border': '1px solid #ccc', 'padding': '10px'}),
        dcc.Graph(id='score-hist', config={'toImageButtonOptions': {'format': 'png'}}, style={'boxShadow': '0 4px 8px rgba(0,0,0,0.05)', 'border': '1px solid #ccc', 'padding': '10px'}),
        dcc.Graph(id='pie-chart', config={'toImageButtonOptions': {'format': 'png'}}, style={'boxShadow': '0 4px 8px rgba(0,0,0,0.05)', 'border': '1px solid #ccc', 'padding': '10px'})
    ], style={'padding': '0 20px 40px 20px', 'backgroundColor': '#ffffff', 'borderRadius': '8px'})
], style={'backgroundColor': '#f7f9fc', 'fontFamily': 'Verdana, Helvetica, sans-serif', 'marginTop': '120px'})

@app.callback(
    [Output('bar-chart', 'figure'),
     Output('score-hist', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('airline-filter', 'value'),
     Input('mode-toggle', 'value')]
)
def update_graphs(selected_airline, toggle_mode):
    if selected_airline == 'klm':
        filtered_df = df[df['text'].str.contains('klm', case=False, na=False)]
    else:
        filtered_df = df

    color_seq = ['#0072B2', '#E69F00', '#009E73']
    hist_color = ['#0072B2']

    df_filtered_for_counts = filtered_df[filtered_df['sentiment'].isin(['positive', 'neutral', 'negative'])]

    bar = px.histogram(df_filtered_for_counts, x='sentiment', title='Sentiment Distribution',
                       color_discrete_sequence=color_seq)

    hist = px.histogram(filtered_df, x='score', nbins=20,
                        title=f'Score Distribution ({selected_airline.upper()})',
                        color_discrete_sequence=hist_color)

    pie = px.pie(df_filtered_for_counts, names='sentiment', title='Sentiment Breakdown',
                 color_discrete_sequence=color_seq)

    dark_mode = 'dark' in toggle_mode
    plot_bgcolor = '#1e1e1e' if dark_mode else '#ffffff'
    paper_bgcolor = '#1e1e1e' if dark_mode else '#ffffff'
    font_color = '#f2f2f2' if dark_mode else '#000000'
    grid_color = '#aaaaaa' if dark_mode else '#444444'

    for fig in [bar, hist, pie]:
        fig.update_layout(
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor,
            font_color=font_color
        )
        if 'xaxis' in fig.layout:
            fig.update_layout(xaxis=dict(gridcolor=grid_color))
        if 'yaxis' in fig.layout:
            fig.update_layout(yaxis=dict(gridcolor=grid_color))

    return bar, hist, pie

if __name__ == '__main__':
    app.run_server(debug=True)
