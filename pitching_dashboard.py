import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import numpy as np
from glob import glob
import data_concat

st.set_page_config(layout="wide")
@st.cache_data
def load_data():
    kdf, fdf = data_concat.data_concat()
    return kdf, fdf

kdf, fdf = load_data()

kdf['trial'] = kdf['trial'].astype(int)
fdf['trial'] = fdf['trial'].astype(int)

# 스트림릿 사이드바 설정
unique_names = kdf['player'].unique()
selected_name = st.sidebar.selectbox('Select Name', unique_names)
filtered_df_by_name = kdf[kdf['player'] == selected_name]
unique_dates = sorted(filtered_df_by_name['day'].unique())
selected_date = st.sidebar.selectbox('Select Date', unique_dates)

filtered_df_by_name_datas = kdf[(kdf['player'] == selected_name) &
                                (kdf['day'] == selected_date)]
unique_trial = sorted(filtered_df_by_name_datas['trial'].unique())
selected_trial = st.sidebar.selectbox('Select Date', unique_trial)

kine_filtered = kdf[(kdf['player'] == selected_name) & 
                    (kdf['day'] == selected_date) &
                    (kdf['trial'] == selected_trial)]

force_filtered = fdf[(fdf['player'] == selected_name) & 
                    (fdf['day'] == selected_date) &
                    (fdf['trial'] == selected_trial)]

kine_filtered.reset_index(inplace=True, drop=True)
force_filtered.reset_index(inplace=True, drop=True)

k_sr = 180
k_kh_time  = kine_filtered['kh_time'][0]
k_kh_time1 = kine_filtered['kh_time'][0] - k_kh_time
k_fc_time  = kine_filtered['fc_time'][0] - k_kh_time
k_mer_time = kine_filtered['mer_time'][0] - k_kh_time
k_br_time  = kine_filtered['br_time'][0] - k_kh_time
stride_length = round(kine_filtered['stride_length'][0])

f_sr = 1080
f_kh_time  = force_filtered['kh_time'][0] 
f_kh_time1 = force_filtered['kh_time'][0]  - f_kh_time
f_fc_time  = force_filtered['fc_time'][0]  - f_kh_time
f_mer_time = force_filtered['mer_time'][0] - f_kh_time
f_br_time  = force_filtered['br_time'][0]  - f_kh_time

k_df = kine_filtered.iloc[k_kh_time:int(k_br_time + k_kh_time + (k_sr * 0.2)),:].reset_index(drop=True)
f_df = force_filtered.iloc[f_kh_time:int(f_br_time + f_kh_time + (f_sr * 0.2)),:].reset_index(drop=True)

f_lead_peak_z_time = f_df['lead_peak_z'][0] - f_kh_time
f_rear_peak_z_time = np.where(f_df['REAR_FORCE_Z'] == f_df['REAR_FORCE_Z'].max())[0][0]
force_peak_time = round((f_lead_peak_z_time - f_rear_peak_z_time) / 1080 , 4)

f_rear_peak_y_time = np.where(f_df['REAR_FORCE_Y'] == f_df['REAR_FORCE_Y'].max())[0][0]
f_lead_min_y_time  = f_df['lead_valley_y'][0] - f_kh_time

k_df.drop(['kh_time','fc_time','mer_time','br_time','mir_time'], axis=1, inplace=True)
f_df.drop(['kh_time','fc_time','mer_time','br_time','mir_time'], axis=1, inplace=True)

k_len = len(k_df)
k_time = np.arange(0,k_len/k_sr, 1/k_sr)
k_time = k_time.round(3)

f_len = len(f_df)
f_time = np.arange(0,f_len/f_sr, 1/f_sr)
f_time = f_time.round(3)
# ===================================================================================
# ============================= Using Data ==========================================
ap_cols = {
    'REAR_FORCE_Y' : ['Trail Leg' , 'blue'],
    'LEAD_FORCE_Y' : ['Stride Leg', 'red'],
}
vt_cols = {
    'REAR_FORCE_Z' : ['Trail Leg' , 'blue'],
    'LEAD_FORCE_Z' : ['Stride Leg', 'red']
}
ks_cols = {
    'PELVIS_ANGLUAR_VELOCITY_Z'        : ['PELVIS'   , 'red'],
    'TORSO_ANGLUAR_VELOCITY_Z'         : ['TORSO'    , 'green'],
    'LEAD_ELBOW_ANGULAR_VELOCITY_X'    : ['ELBOW'    , 'blue'],
    'LEAD_SHOULDER_ANGULAR_VELOCITY_Z' : ['SHOULDER' , 'yellow'],
}
ang_cols = {
    'TORSO_PELVIS_ANGLE_Z'            : 'HIP-SHOULDER SEPARATION',
    'LEAD_ELBOW_ANGLE_X'              : 'ELBOW FLEXION',
    'LEAD_SHOULDER_ANGLE_Z'           : 'SHOULDER EXTERNAL ROTATION',          
    'LEAD_SHOULDER_ANGLE_X'           : 'SHOULDER HORIZONTAL ABDUCTION',
    'LEAD_KNEE_ANGLE_X'               : 'LEAD LEG KNEE FLEXION',
    'LEAD_KNEE_ANGULAR_VELOCITY_X'    : 'LEAD LEG KNEE EXTENSION ANGULAR VELOCITY',
    'LEAD_SHOULDER_ANGLE_Y'           : 'SHOULDER ABDUCTION', 
    'TORSO_ANGLE_X'                   : 'TRUNK FORWARD TILT',
    'TORSO_ANGLE_Y'                   : 'TRUNK LATERAL TILT',
}

# ===================================================================================
# ============================= DashBoard ===========================================
st.title('KUM BASEBALL PITCHING REPORT')

# ============================ 그래프 함수 정의 =========================================
def grf_plotly(data, cols, time, kh_time, fc_time, mer_time, br_time, axis):
    title = 'GROUND REACTION FORCE (AP-AXIS)' if axis == 'ap' else 'GROUND REACTION FORCE (Vertical)'
    
    y_values = {
        'max'       : {},
        'max_frame' : {},
        'min'       : {},
        'min_frame' : {},
        'kh_time'   : {},
        'fc_time'   : {},
        'mer_time'  : {},
        'br_time'   : {},
    }
    
    # Create traces
    traces = []
    for col, info in cols.items():
        df = data[col]
        trace = go.Scatter(x=time, y=df, mode='lines', name=info[0], line=dict(color=info[-1]))
        traces.append(trace)
        
        # Perform and store the calculations for max, min and specific times
        y_values['kh_time'][col] = round(df.iloc[kh_time], 2)
        y_values['fc_time'][col] = round(df.iloc[fc_time], 2)
        y_values['mer_time'][col] = round(df.iloc[mer_time], 2)
        y_values['br_time'][col] = round(df.iloc[br_time], 2)
        y_values['max'][col] = round(df.max(), 2)
        y_values['max_frame'][col] = df.idxmax()
        y_values['min'][col] = round(df.min(), 2)
        y_values['min_frame'][col] = df.idxmin()

    # Adding reference lines and annotations
    reference_lines = []
    annotations = []

    # Add vertical lines and annotations for key events
    for key_time, description in zip([kh_time, fc_time, mer_time, br_time],
                                     ['KH', 'FC', 'MER', 'BR']):
        reference_lines.append(
            go.Scatter(x=[time[key_time], time[key_time]], y=[data[cols.keys()].min().min(), data[cols.keys()].max().max()],
                       mode='lines', line=dict(color='black', width=2, dash='dash'),
                       showlegend=False)
        )
        annotations.append(
            dict(x=time[key_time + 2], y=0.95, xref='x', yref='paper', showarrow=False,
                 text=description,textangle=-90, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', borderwidth=0,
    
                 )
        )

    # Update the layout with additional elements
    layout = go.Layout(
        title=title,
        xaxis=dict(title='Time [s]',
                    showgrid=False),
        yaxis=dict(
                    title='Force [% BW]',
                    showgrid=True,         # This will show the horizontal gridlines
                    gridcolor='lightgrey',
                    gridwidth=1,
                    zeroline=False
                ),
        showlegend=True,
        legend=dict(
                    x=1, # Adjust this value to move the legend left or right
                    y=1, # Adjust this value to move the legend up or down
                    xanchor='right', # Anchor the legend's right side at the x position
                    yanchor='top', # Anchor the legend's top at the y position
                    bgcolor='rgb(43,48,61)' # Set a background color with a bit of transparency
                    ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=600,
        hovermode='closest',
        plot_bgcolor='rgb(43,48,61)',
        annotations=annotations
    )

    # Create the figure
    fig = go.Figure(data=traces + reference_lines, layout=layout)

    return fig, y_values
def one_angle_plotly(data, cols, time, k_kh_time, k_fc_time, k_mer_time, k_br_time):
    ang = {
        'max'       : {},
        'max_frame' : {},
        'min'       : {},
        'min_frame' : {},
        'kh_time'   : {},
        'fc_time'   : {},
        'fp_time'   : {},
        'mer_time'  : {},
        'br_time'   : {},
    }
    
    figures = {}
    
    for col in cols:
        df = data[col]
        if 'VELOCITY' in col:
            y_label = 'Angular Velocity [deg/s]'
        else:
            y_label = 'Angle [deg]'
        
        # Create the trace for the main data line
        trace = go.Scatter(x=time, y=df, mode='lines', name=cols[col], line=dict(color='firebrick'))
        traces = [trace]
        
        ang['kh_time'][col]   = round(df[k_kh_time], 2)
        ang['fc_time'][col]   = round(df[k_fc_time], 2)
        ang['mer_time'][col]  = round(df[k_mer_time], 2)
        ang['br_time'][col]   = round(df[k_br_time], 2)
        ang['max'][col]       = round(df.max(), 2)
        ang['max_frame'][col] = np.where(df == df.max())[0][0]
        ang['min'][col]       = round(df.min(), 2)
        ang['min_frame'][col] = np.where(df == df.min())[0][0]
        
        if col in ['TORSO_ANGLE_Y','LEAD_ELBOW_ANGLE_X','LEAD_SHOULDER_ANGLE_Y','LEAD_SHOULDER_ANGLE_Z','LEAD_KNEE_ANGULAR_VELOCITY_X']:
            ang['max'][col]  = round(df[k_fc_time-40:k_br_time+15].max(), 2)
            ang['max_frame'][col] = np.where(df == df[k_fc_time-40:k_br_time+15].max())[0][0]

        elif col in ['LEAD_KNEE_ANGLE_X']:
            ang['max'][col]  = round(df[k_fc_time:k_br_time+1].max(), 2)
            ang['max_frame'][col] = np.where(df == df[k_fc_time:k_br_time+1].max())[0][0]
        
        reference_lines =[]
        annotations = []
        
        for key_time, description in zip([k_kh_time, k_fc_time, k_mer_time, k_br_time],
                                     ['KH', 'FC', 'MER', 'BR']):
            
            reference_lines.append(
            go.Scatter(x=[time[key_time], time[key_time]], y=[df.min(), df.max()],
                       mode='lines', line=dict(color='black', width=2, dash='dash'),
                       showlegend=False)
        )
            annotations.append(
            dict(x=time[key_time + 2], y=0.95, xref='x', yref='paper', showarrow=False,
                 text=description,textangle=-90, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', borderwidth=0,
                 )
        )
        
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
            margin=dict(l=40, r=40, t=40, b=40),
            height=600,
            plot_bgcolor='rgb(43,48,61)',
            annotations=annotations
        )
        
        # Create the figure and add the traces to it
        fig = go.Figure(data=traces + reference_lines, layout=layout)
        
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
            line=dict(color=ks_cols[col][-1])
        )
        traces.append(trace)
        ks['peak'][col] = round(data[col].max(), 2)
        ks['time'][col] = time[data[col].idxmax()]
    
    
    # Standard event lines
    reference_lines =[]
    annotations = []
    
    for key_time, description in zip([k_kh_time, k_fc_time, k_mer_time, k_br_time],
                                    ['KH', 'FC', 'MER', 'BR']):
        
        reference_lines.append(
        go.Scatter(x=[time[key_time], time[key_time]], y=[data[ks_cols.keys()].min().min(), data[ks_cols.keys()].max().max()],
                    mode='lines', line=dict(color='black', width=2, dash='dash'),
                    showlegend=False)
    )
        annotations.append(
        dict(x=time[key_time + 2], y=0.95, xref='x', yref='paper', showarrow=False,
                text=description,textangle=-90, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', borderwidth=0,
                )
    )
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
        showlegend=True,
        legend=dict(orientation='h'),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='rgb(43,48,61)'
    )

    # Create the figure and add traces to it
    fig = go.Figure(data=traces + reference_lines, layout=layout)
    
    return ks, fig

# ============================ 그래프 및 시점 수치 =======================================
force_ap_fig, force_ap_values = grf_plotly(f_df, ap_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='ap')
force_vt_fig, force_vt_values = grf_plotly(f_df, vt_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='vt')
kine_values, kine_fig = one_angle_plotly(k_df, ang_cols, k_time, k_kh_time1, k_fc_time, k_mer_time, k_br_time)
kinematic_values, kinematic_fig = kinematic_sequence_plotly(k_df, ks_cols, k_time, k_kh_time1, k_fc_time, k_mer_time, k_br_time)

force_ap_fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
)
force_vt_fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
)
for col in kine_fig:
    fig = kine_fig[col]
    fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
    )
kinematic_fig.update_layout(
    width=800,
    height=400
)

st.header('분석 구간')
st.image('image/analysis.PNG', use_column_width=True)

st.subheader('KINEMATICS PARAMETERS')

st.write('KINEMATIC SEQUENCE')
col1, col2 = st.columns([1,2.8])
with col1:
    st.image('image/GRF_Y.png', use_column_width=True)
with col2:
    st.plotly_chart(kinematic_fig, use_container_width=True)

for col in ang_cols:
    st.write(ang_cols[col])
    col1, col2 = st.columns([1,2.8])
    with col1:
        st.image('image/GRF_Y.png', use_column_width=True)
    with col2:
        st.plotly_chart(kine_fig[col], use_container_width=True)
            
        
st.subheader('KINETICS PARAMETERS')
st.write("GROUND REACTION FORCE AP")
col1, col2 = st.columns([1,2.8])
with col1:
    st.image('image/GRF_Y.png', use_column_width=True)
with col2:
    st.plotly_chart(force_ap_fig, use_container_width=True)
force_ap_fig.update_layout(height=600) 

st.write("GROUND REACTION FORCE VERTICAL")
col1, col2 = st.columns([1,2.8])
with col1:
    st.image('image/GRF_Z.png', use_column_width=True)
with col2:
    st.plotly_chart(force_vt_fig, use_container_width=True)

