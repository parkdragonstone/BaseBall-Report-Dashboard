import plotly.graph_objs as go
import numpy as np
import streamlit as st
import pandas as pd
import subprocess

username_passward = {
    "kookmin" : ["640511"],
    "yongseok" : ["1234"],
    "Kangmingu" : ["1234",["Kangmingu"]],
    "songseok" : ["1234", ["songseokhyun"]],
    "leejeongwoo" : ["1q2w3e4r", ["leejoengwoo"]]
}
master_ID = ['kookmin','yongseok']

def transform_list(nums):
    indexed_nums = list(enumerate(nums))
    indexed_nums.sort(key=lambda x: x[1])
    transformed = [0] * len(nums)
    current_rank = 1
    for i in range(len(nums)):
        if i > 0 and indexed_nums[i][1] != indexed_nums[i-1][1]:
            current_rank += 1
        transformed[indexed_nums[i][0]] = current_rank
    return transformed

def grf_plotly(data, cols, time, kh_time, fc_time, mer_time, br_time, axis):
    if axis == 'ap':
        title = 'GROUND REACTION FORCE (AP-AXIS)'
        ylb = 'Force [%BW]'
    elif axis == 'result':
        title = 'GROUND REACTION FORCE (RESULTANT)'
        ylb = 'Force [%BW]'
    elif axis == 'vt':
        title = 'GROUND REACTION FORCE (Vertical)'
        ylb = 'Force [%BW]'
    elif axis == 'freemoment':
        title = 'TORQUE'
        ylb = 'Torque [N*m]'

    y_values = {
        'max'       : {},
        'max_time' : {},
        'kh_time'   : {},
        'fc_time'   : {},
        'mer_time'  : {},
        'br_time'   : {},
    }
    
    # Create traces
    traces = []
    for col, info in cols.items():
        df = data[col]
        trace = go.Scatter(x=time, y=df, mode='lines', name=info[0], line=dict(color=info[-1],width=4))
        traces.append(trace)
        
        # Perform and store the calculations for max, min and specific times
        y_values['kh_time'][col] = round(df.iloc[kh_time], 2)
        y_values['fc_time'][col] = round(df.iloc[fc_time], 2)
        y_values['mer_time'][col] = round(df.iloc[mer_time], 2)
        y_values['br_time'][col] = round(df.iloc[br_time], 2)
        if col == 'LEAD_FORCE_Y':
            y_values['max'][col] = round(df.min(), 2)
            y_values['max_time'][col] = np.where(df == df.min())[0][0]
        else:
            y_values['max'][col] = round(df.max(), 2)
            y_values['max_time'][col] = np.where(df == df.max())[0][0]

    event_times = [kh_time, fc_time, mer_time, br_time]
    event_names = ['KH', 'FC', 'MER', 'BR']
    shapes = [
        {
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': time[event_time],
            'y0': 0,
            'x1': time[event_time],
            'y1': 1,
            'opacity' : 0.5,
            'line': {
                'color': 'cyan',
                'width': 4,
                'dash': 'dash',
            }
        } for event_time in event_times
    ]
    annotations = [
        {
            'x': time[event_time + 12],
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': label,
            'showarrow': False,
            'font': {
                'color': 'cyan',
                'size' : 16
            },
            'textangle': -90
        } for event_time, label in zip(event_times, event_names)
    ]

    # Update the layout with additional elements
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Time [s]',
                    showgrid=False),
        yaxis=dict(
                    title=ylb,
                    showgrid=True,         # This will show the horizontal gridlines
                    gridcolor='lightgrey',
                    gridwidth=1,
                    zeroline=False
                ),
        showlegend=True,
        shapes = shapes,
        legend=dict(
                    # x=1, # Adjust this value to move the legend left or right
                    # y=1, # Adjust this value to move the legend up or down
                    # xanchor='right', # Anchor the legend's right side at the x position
                    # yanchor='top', # Anchor the legend's top at the y position
                    # bgcolor='rgb(43,48,61)', # Set a background color with a bit of transparency
                    orientation = 'h',
                    ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600,
        hovermode='closest',
        plot_bgcolor='rgb(43,48,61)',
        annotations=annotations
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    return fig, y_values

def one_angle_plotly(data, cols, time, k_kh_time, k_fc_time, k_mer_time, k_br_time):
    ang = {
        'max'       : {},
        'max_time' : {},
        'kh_time'   : {},
        'fc_time'   : {},
        'mer_time'  : {},
        'br_time'   : {},
    }
    
    figures = {}
    
    for col in cols:
        df = data[col]
        if 'VELOCITY' in col:
            y_label = 'Angular Velocity [deg/s]'
        elif 'ANGLE' in col:
            y_label = 'Angle [deg]'
        else:
            y_label = 'Distance [CM]'
        # Create the trace for the main data line
        trace = go.Scatter(x=time, y=df, mode='lines', name=cols[col], line=dict(color='firebrick', width=4))
        traces = [trace]
        
        ang['kh_time'][col]   = round(df[k_kh_time], 2)
        ang['fc_time'][col]   = round(df[k_fc_time], 2)
        ang['mer_time'][col]  = round(df[k_mer_time], 2)
        ang['br_time'][col]   = round(df[k_br_time], 2)
        ang['max'][col]       = round(df.max(), 2)
        ang['max_time'][col] = np.where(df == df.max())[0][0]
        
        if col in ['TORSO_ANGLE_Y','LEAD_ELBOW_ANGLE_X','LEAD_SHOULDER_ANGLE_Y','LEAD_SHOULDER_ANGLE_Z','LEAD_KNEE_ANGULAR_VELOCITY_X']:
            ang['max'][col]  = round(df[k_fc_time-40:k_br_time+15].max(), 2)
            ang['max_time'][col] = np.where(df == df[k_fc_time-40:k_br_time+15].max())[0][0]

        elif col in ['LEAD_KNEE_ANGLE_X', 'HAND_ELBOW_HEIGHT']:
            ang['max'][col]  = round(df[k_fc_time:k_br_time+1].max(), 2)
            ang['max_time'][col] = np.where(df == df[k_fc_time:k_br_time+1].max())[0][0]
        
        elif col in ['TORSO_PELVIS_ANGLE_Z','LEAD_SHOULDER_ANGLE_X','PELVIS_ANGLE_Z','TORSO_ANGLE_Z']:
            ang['max'][col]  = round(df.min(), 2)
            ang['max_time'][col] = np.where(df == df.min())[0][0]
        
        event_times = [k_kh_time, k_fc_time, k_mer_time, k_br_time]
        event_names = ['KH', 'FC', 'MER', 'BR']
        shapes = [
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'paper',
                'x0': time[event_time],
                'y0': 0,
                'x1': time[event_time],
                'y1': 1,
                'opacity' : 0.5,
                'line': {
                    'color': 'cyan',
                    'width': 4,
                    'dash': 'dash',
                }
            } for event_time in event_times
        ]
        annotations = [
            {
                'x': time[event_time + 2],
                'y': 1,
                'xref': 'x',
                'yref': 'paper',
                'text': label,
                'showarrow': False,
                'font': {
                    'color': 'cyan',
                    'size' : 16
                },
                'textangle': -90
            } for event_time, label in zip(event_times, event_names)
        ]
        
        # Define the layout
        layout = go.Layout(
            title=f'{cols[col]}',
            xaxis=dict(title='Time [s]',
                       showgrid=False),
            yaxis=dict(title=y_label,
                       autorange = True,
                       rangemode='tozero',
                        showgrid=True,         # This will show the horizontal gridlines
                        gridcolor='lightgrey',
                        gridwidth=1,
                        zeroline=False,
                        ),                        
            showlegend=False,
            shapes =shapes,
            margin=dict(l=40, r=40, t=40, b=40),
            height=600,
            plot_bgcolor='rgb(43,48,61)',
            annotations=annotations
        )
        
        # Create the figure and add the traces to it
        fig = go.Figure(data=traces, layout=layout)
        
        # Store the figure in the dictionary
        figures[col] = fig
        
    return ang, figures

def kinematic_sequence_plotly(data, ks_cols, time, k_kh_time, k_fc_time, k_mer_time, k_br_time):
    ks = {
        'peak' : {},
        'time' : {},
    }
    
    # Create traces for each data series
    traces = []
    for col in ks_cols:
        trace = go.Scatter(
            x=time, 
            y=data[col], 
            mode='lines', 
            name=ks_cols[col][0],
            opacity=0.9,
            line=dict(color=ks_cols[col][-1],width= 3), 
        )
        traces.append(trace)
        ks['peak'][col] = round(data[col].max(), 2)
        ks['time'][col] = np.where(data[col] == data[col].max())[0][0]
    
    event_times = [k_kh_time, k_fc_time, k_mer_time, k_br_time]
    event_names = ['KH', 'FC', 'MER', 'BR']
    shapes = [
        {
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': time[event_time],
            'y0': 0,
            'x1': time[event_time],
            'y1': 1,
            'opacity' : 0.5,
            'line': {
                'color': 'cyan',
                'width': 3,
                'dash': 'dash',
            }
        } for event_time in event_times
    ]
    annotations = [
        {
            'x': time[event_time + 2],
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': label,
            'showarrow': False,
            'font': {
                'color': 'cyan',
                'size' : 16
            },
            'textangle': -90
        } for event_time, label in zip(event_times, event_names)
    ]

    # Define the layout with annotations and shapes
    layout = go.Layout(
        title='KINEMATIC SEQUENCE',
        xaxis=dict(title='Time [s]',
                   showgrid=False),
        yaxis=dict(title='Angular Velocity [Deg/s]', 
                   autorange=True,           
                    rangemode='tozero',
                    showgrid=True,         # This will show the horizontal gridlines
                    gridcolor='lightgrey',
                    gridwidth=1,
                    zeroline=False,),
        annotations=annotations,
        shapes=shapes,
        showlegend=True,
        legend=dict(orientation='h'),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='rgb(43,48,61)'
    )

    # Create the figure and add traces to it
    fig = go.Figure(data=traces, layout=layout)
    
    return ks, fig

def energy_flow_plotly(data, ks_cols, time, k_kh_time, k_fc_time, k_mer_time, k_br_time):
    ks = {
        'max' : {},
        'max_time' : {},
        'fc_time' : {},
        'mer_time' : {},
        'br_time' : {}
    }
    
    # Create traces for each data series
    traces = []
    for col in ks_cols:
        trace = go.Scatter(
            x=time, 
            y=data[col], 
            mode='lines', 
            name=ks_cols[col][0],
            opacity=0.9,
            line=dict(color=ks_cols[col][-1],width= 3), 
        )
        traces.append(trace)
        ks['max'][col] = round(data[col].max(), 2)
        ks['max_time'][col] = np.where(data[col] == data[col].max())[0][0]
        ks['fc_time'][col] = round(data[col][k_fc_time],2)
        ks['mer_time'][col] = round(data[col][k_mer_time],2)
        ks['br_time'][col] = round(data[col][k_br_time],2)
    
    event_times = [k_kh_time, k_fc_time, k_mer_time, k_br_time]
    event_names = ['KH', 'FC', 'MER', 'BR']
    shapes = [
        {
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': time[event_time],
            'y0': 0,
            'x1': time[event_time],
            'y1': 1,
            'opacity' : 0.5,
            'line': {
                'color': 'cyan',
                'width': 3,
                'dash': 'dash',
            }
        } for event_time in event_times
    ]
    annotations = [
        {
            'x': time[event_time + 2],
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': label,
            'showarrow': False,
            'font': {
                'color': 'cyan',
                'size' : 16
            },
            'textangle': -90
        } for event_time, label in zip(event_times, event_names)
    ]

    # Define the layout with annotations and shapes
    layout = go.Layout(
        title='SEGMENT POWER',
        xaxis=dict(title='Time [s]',
                   showgrid=False),
        yaxis=dict(title='POWER [W/kg]', 
                   autorange=True,           
                    rangemode='tozero',
                    showgrid=True,         # This will show the horizontal gridlines
                    gridcolor='lightgrey',
                    gridwidth=1,
                    zeroline=False,),
        annotations=annotations,
        shapes=shapes,
        showlegend=True,
        legend=dict(orientation='h'),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='rgb(43,48,61)'
    )

    # Create the figure and add traces to it
    fig = go.Figure(data=traces, layout=layout)
    
    return ks, fig

def linear_momentum_plotly(data, ks_cols, time, k_kh_time, k_fc_time, k_mer_time, k_br_time):
    ks = {
        'max' : {},
        'max_time' : {},
        'fc_time' : {},
        'mer_time' : {},
        'br_time' : {}
    }
    
    # Create traces for each data series
    traces = []
    for col in ks_cols:
        trace = go.Scatter(
            x=time, 
            y=data[col], 
            mode='lines', 
            name=ks_cols[col][0],
            opacity=0.9,
            line=dict(color=ks_cols[col][-1],width= 3), 
        )
        traces.append(trace)
        ks['max'][col] = round(data[col].max(), 2)
        ks['max_time'][col] = np.where(data[col] == data[col].max())[0][0]
        ks['fc_time'][col] = round(data[col][k_fc_time],2)
        ks['mer_time'][col] = round(data[col][k_mer_time],2)
        ks['br_time'][col] = round(data[col][k_br_time],2)
    
    event_times = [k_kh_time, k_fc_time, k_mer_time, k_br_time]
    event_names = ['KH', 'FC', 'MER', 'BR']
    shapes = [
        {
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': time[event_time],
            'y0': 0,
            'x1': time[event_time],
            'y1': 1,
            'opacity' : 0.5,
            'line': {
                'color': 'cyan',
                'width': 3,
                'dash': 'dash',
            }
        } for event_time in event_times
    ]
    annotations = [
        {
            'x': time[event_time + 2],
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': label,
            'showarrow': False,
            'font': {
                'color': 'cyan',
                'size' : 16
            },
            'textangle': -90
        } for event_time, label in zip(event_times, event_names)
    ]

    # Define the layout with annotations and shapes
    layout = go.Layout(
        title='LINEAR MOMENTUM',
        xaxis=dict(title='Time [s]',
                   showgrid=False),
        yaxis=dict(title='Momentum [kg*m²/s]', 
                   autorange=True,           
                    rangemode='tozero',
                    showgrid=True,         # This will show the horizontal gridlines
                    gridcolor='lightgrey',
                    gridwidth=1,
                    zeroline=False,),
        annotations=annotations,
        shapes=shapes,
        showlegend=True,
        legend=dict(orientation='h'),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='rgb(43,48,61)'
    )

    # Create the figure and add traces to it
    fig = go.Figure(data=traces, layout=layout)
    
    return ks, fig

def angular_momentum_plotly(data, ks_cols, time, k_kh_time, k_fc_time, k_mer_time, k_br_time):
    ks = {
        'max' : {},
        'max_time' : {},
        'fc_time' : {},
        'mer_time' : {},
        'br_time' : {}
    }
    
    # Create traces for each data series
    traces = []
    for col in ks_cols:
        trace = go.Scatter(
            x=time, 
            y=data[col], 
            mode='lines', 
            name=ks_cols[col][0],
            opacity=0.9,
            line=dict(color=ks_cols[col][-1],width= 3), 
        )
        traces.append(trace)
        ks['max'][col] = round(data[col].max(), 2)
        ks['max_time'][col] = np.where(data[col] == data[col].max())[0][0]
        ks['fc_time'][col] = round(data[col][k_fc_time],2)
        ks['mer_time'][col] = round(data[col][k_mer_time],2)
        ks['br_time'][col] = round(data[col][k_br_time],2)
    
    event_times = [k_kh_time, k_fc_time, k_mer_time, k_br_time]
    event_names = ['KH', 'FC', 'MER', 'BR']
    shapes = [
        {
            'type': 'line',
            'xref': 'x',
            'yref': 'paper',
            'x0': time[event_time],
            'y0': 0,
            'x1': time[event_time],
            'y1': 1,
            'opacity' : 0.5,
            'line': {
                'color': 'cyan',
                'width': 3,
                'dash': 'dash',
            }
        } for event_time in event_times
    ]
    annotations = [
        {
            'x': time[event_time + 2],
            'y': 1,
            'xref': 'x',
            'yref': 'paper',
            'text': label,
            'showarrow': False,
            'font': {
                'color': 'cyan',
                'size' : 16
            },
            'textangle': -90
        } for event_time, label in zip(event_times, event_names)
    ]

    # Define the layout with annotations and shapes
    layout = go.Layout(
        title='ANGULAR MOMENTUM',
        xaxis=dict(title='Time [s]',
                   showgrid=False),
        yaxis=dict(title='Momentum [kg*m²/(s*rad)]', 
                   autorange=True,           
                    rangemode='tozero',
                    showgrid=True,         # This will show the horizontal gridlines
                    gridcolor='lightgrey',
                    gridwidth=1,
                    zeroline=False,),
        annotations=annotations,
        shapes=shapes,
        showlegend=True,
        legend=dict(orientation='h'),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='rgb(43,48,61)'
    )

    # Create the figure and add traces to it
    fig = go.Figure(data=traces, layout=layout)
    
    return ks, fig

def energy_plotly(data, cols, time, k_kh_time, k_fc_time, k_mer_time, k_br_time):
    ang = {
        'max'       : {},
        'max_time'  : {},
        'min'       : {},
        'min_time'  : {},
        'kh_time'   : {},
        'fc_time'   : {},
        'mer_time'  : {},
        'br_time'   : {},
    }
    
    figures = {}
    
    for col in cols:
        df = data[col]
        if 'LINEAR_MOMENTUM' in col:
            y_label = 'Momentum [kg•m²/s]'
        elif 'ANGULAR_MOMENTUM' in col:
            y_label = 'Momentum [kg•m²/(s•rad)]'
        elif 'NET' in col:
            y_label = 'Power [W/kg]'
        # Create the trace for the main data line
        trace = go.Scatter(x=time, y=df, mode='lines', name=cols[col][0], line=dict(color='firebrick', width=4))
        traces = [trace]
        
        ang['kh_time'][col]   = round(df[k_kh_time], 2)
        ang['fc_time'][col]   = round(df[k_fc_time], 2)
        ang['mer_time'][col]  = round(df[k_mer_time], 2)
        ang['br_time'][col]   = round(df[k_br_time], 2)
        ang['max'][col]       = round(df.max(), 2)
        ang['max_time'][col] = np.where(df == df.max())[0][0]
        ang['min'][col]       = round(df.min(), 2)
        ang['min_time'][col] = np.where(df == df.min())[0][0]
        
        event_times = [k_kh_time, k_fc_time, k_mer_time, k_br_time]
        event_names = ['KH', 'FC', 'MER', 'BR']
        shapes = [
            {
                'type': 'line',
                'xref': 'x',
                'yref': 'paper',
                'x0': time[event_time],
                'y0': 0,
                'x1': time[event_time],
                'y1': 1,
                'opacity' : 0.5,
                'line': {
                    'color': 'cyan',
                    'width': 4,
                    'dash': 'dash',
                }
            } for event_time in event_times
        ]
        annotations = [
            {
                'x': time[event_time + 2],
                'y': 1,
                'xref': 'x',
                'yref': 'paper',
                'text': label,
                'showarrow': False,
                'font': {
                    'color': 'cyan',
                    'size' : 16
                },
                'textangle': -90
            } for event_time, label in zip(event_times, event_names)
        ]
        
        # Define the layout
        layout = go.Layout(
            title=f'{cols[col][0]}',
            xaxis=dict(title='Time [s]',
                       showgrid=False),
            yaxis=dict(title=y_label,
                       autorange = True,
                       rangemode='tozero',
                        showgrid=True,         # This will show the horizontal gridlines
                        gridcolor='lightgrey',
                        gridwidth=1,
                        zeroline=False,
                        ),                        
            showlegend=False,
            shapes =shapes,
            margin=dict(l=40, r=40, t=40, b=40),
            height=600,
            plot_bgcolor='rgb(43,48,61)',
            annotations=annotations
        )
        
        # Create the figure and add the traces to it
        fig = go.Figure(data=traces, layout=layout)
        
        # Store the figure in the dictionary
        figures[col] = fig
        
    return ang, figures

def check_credentials(username, passward):
    if username in username_passward and passward == username_passward[username][0]:
        st.session_state['authenticated'] = True
        if username in master_ID:
            st.session_state['selected_name'] = username
        else:
            st.session_state['selected_name'] = username_passward[username][1]
    else:
        st.session_state['authenticated'] = False
        st.error('ID나 PASSWORD가 잘못되었습니다')

# 인증되지 않았을 때 로그인 폼을 보여주는 함수
def show_login_form():
    with st.container():
        st.write("Login")
        username = st.text_input("ID", key='login_username')
        password = st.text_input("PASSWORD", type="password", key='login_password')
        login_button = st.button("login", on_click=check_credentials, args=(username, password))
        
def save_feedback(df, csv_file, name, date, trial, feedback):
    new_feedback = pd.DataFrame({
        'name' : [name],
        'date' : [date],
        'trial' : [trial],
        'feedback' : [feedback]})
    
    df_updated = pd.concat([df,new_feedback], ignore_index=True)
    df_updated.to_csv(csv_file, index=False)
        # GitHub에 변경 사항 푸시
    
    return df_updated
    