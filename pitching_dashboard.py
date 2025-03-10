import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import numpy as np
from glob import glob
import data_concat
from graph_data import show_login_form,transform_list, grf_plotly, one_angle_plotly,x_factor_plotly
from graph_data import kinematic_sequence_plotly, segment_power_plotly, linear_momentum_plotly,angular_momentum_plotly

csv_file = 'feedback.csv'
feedback_df = pd.read_csv(csv_file)
master_ID = ['kookmin','yongseok']

st.set_page_config(page_title = "KMU BASEBALL PITCHING REPORT", 
                layout="wide"
                    )
@st.cache_data
def load_data():
    kdf, fdf = data_concat.data_concat()
    return kdf, fdf

# if 'authenticated' not in st.session_state:
#     st.session_state['authenticated'] = False

# # 사용자가 인증되지 않았다면 로그인 폼을 보여줌
# if not st.session_state['authenticated']:
#     show_login_form()
    
# else:
kdf, fdf = load_data()  
kdf['trial'] = kdf['trial'].astype(int)
fdf['trial'] = fdf['trial'].astype(int)

# # 스트림릿 사이드바 설정
# if st.session_state['selected_name'] in master_ID:
#     unique_names = kdf['player'].unique()
# else:
#     unique_names = st.session_state['selected_name']

unique_names = kdf['player'].unique()
selected_name = st.sidebar.selectbox('Select Name', unique_names)
filtered_df_by_name = kdf[kdf['player'] == selected_name]
unique_dates = sorted(filtered_df_by_name['day'].unique())
selected_date = st.sidebar.selectbox('Select Date', unique_dates)

filtered_df_by_name_datas = kdf[(kdf['player'] == selected_name) &
                                (kdf['day'] == selected_date)]
unique_trial = sorted(filtered_df_by_name_datas['trial'].unique())
selected_trial = st.sidebar.selectbox('Select Trial', unique_trial)

kine_filtered = kdf[(kdf['player'] == selected_name) & 
                    (kdf['day'] == selected_date) &
                    (kdf['trial'] == selected_trial)]

force_filtered = fdf[(fdf['player'] == selected_name) & 
                    (fdf['day'] == selected_date) &
                    (fdf['trial'] == selected_trial)]

kine_filtered.reset_index(inplace=True, drop=True)
force_filtered.reset_index(inplace=True, drop=True)

k_sr = 180
multiple = 6
f_sr = k_sr * multiple

k_kh_time  = kine_filtered['kh_time'][0]
k_kh_time1 = kine_filtered['kh_time'][0] - k_kh_time
k_fc_time  = kine_filtered['fc_time'][0] - k_kh_time
k_mer_time = kine_filtered['mer_time'][0] - k_kh_time
k_br_time  = kine_filtered['br_time'][0] - k_kh_time

stride_length = round(float(kine_filtered['stride_length'][0]))
ball_speed = round(float(kine_filtered['ball_speed'][0]) * 1.6)
pit_type = kine_filtered['pit_type'][0]
k_total_time = k_br_time+1 - k_fc_time

f_kh_time  = force_filtered['kh_time'][0] 
f_kh_time1 = force_filtered['kh_time'][0]  - f_kh_time
f_fc_time  = force_filtered['fc_time'][0]  - f_kh_time
f_mer_time = force_filtered['mer_time'][0] - f_kh_time
f_br_time  = force_filtered['br_time'][0]  - f_kh_time
f_total_time = f_br_time+1 - f_fc_time

k_df = kine_filtered.iloc[k_kh_time:int(k_br_time + k_kh_time + (k_sr * 0.2)),:].reset_index(drop=True)
f_df = force_filtered.iloc[f_kh_time:int(f_br_time + f_kh_time + (f_sr * 0.2)),:].reset_index(drop=True)

k_df.drop(['kh_time','fc_time','mer_time','br_time','mir_time'], axis=1, inplace=True)
f_df.drop(['kh_time','fc_time','mer_time','br_time','mir_time'], axis=1, inplace=True)

k_time = k_df['TIME']
f_time = f_df['TIME']
# ===================================================================================
# ============================= Using Data ==========================================
ap_cols = {
    'REAR_FORCE_Y' : ['Trail Leg' , 'yellow'],
    'LEAD_FORCE_Y' : ['Stride Leg', 'red'],
}
vt_cols = {
    'REAR_FORCE_Z' : ['Trail Leg' , 'yellow'],
    'LEAD_FORCE_Z' : ['Stride Leg', 'red']
}
result_cols = {
    'REAR_RESULT' : ['Trail Leg' , 'yellow'],
    'LEAD_RESULT' : ['Stride Leg', 'red']
}
momentum_cols = {
    'REAR_FREEMOMENT_Z' : ['Trail Leg' , 'yellow'],
    'LEAD_FREEMOMENT_Z' : ['Stride Leg', 'red']
}
ks_cols = {
    'PELVIS_SEG_ANGULAR_VELOCITY_Z'        : ['PELVIS'   , 'red'],
    'TORSO_SEG_ANGULAR_VELOCITY_Z'         : ['TORSO'    , 'green'],
    'LEAD_ELBOW_ANGULAR_VELOCITY_X'    : ['ELBOW'    , 'blue'],
    'LEAD_SHOULDER_ANGULAR_VELOCITY_Z' : ['SHOULDER' , 'yellow'],
}
ang_cols = {
    'TORSO_PELVIS_ANGLE_Z'            : 'HIP-SHOULDER SEPARATION',
    'LEAD_ELBOW_ANGLE_X'              : 'ELBOW FLEXION',
    'LEAD_SHOULDER_ANGLE_Z'           : 'SHOULDER EXTERNAL ROTATION',          
    'LEAD_SHOULDER_ANGLE_X'           : 'SHOULDER HORIZONTAL ABDUCTION',
    'LEAD_KNEE_ANGLE_X'               : 'LEAD KNEE FLEXION',
    'LEAD_KNEE_ANGULAR_VELOCITY_X'    : 'LEAD KNEE EXTENSION ANGULAR VELOCITY',
    'LEAD_SHOULDER_ANGLE_Y'           : 'SHOULDER ABDUCTION', 
    'TORSO_ANGLE_X'                   : 'TRUNK FORWARD TILT',
    'TORSO_ANGLE_Y'                   : 'TRUNK LATERAL TILT',
    'HAND_ELBOW_HEIGHT'               : 'HAND ELBOW HEIGHT',
    'TORSO_ANGLE_Z'                   : 'TRUNK ROTATION',
    'PELVIS_ANGLE_Z'                  : 'PELVIS ROTATION',
    'PELVIS_TORSO_AP_DISTANCE'        : 'PELVIS-HEAD DISTANCE',
    'ANKLE_HAND_AP_DISTANCE'          : 'ANKLE-HAND DISTANCE',
}
xfactor_cols = {
    'TORSO_PELVIS_ANGLE_Z'            : ['X-Factor', 'yellow'],
    'PELVIS_ANGLE_Z'                  : ['PELVIS', 'red'],
    'TORSO_ANGLE_Z'                   : ['TORSO', 'green'],
}
sp_cols = {
    "PELVIS_NET_SP" : ["PELVIS","red"],
    "TORSO_NET_SP" : ["TORSO","green"],
    "LEAD_ARM_NET_SP" : ["UPPER ARM","blue"],
    "LEAD_FOREARM_NET_SP" : ["FOREARM","yellow"],
}

lm_cols = {
    "PELVIS_LINEAR_MOMENTUM": ["PELVIS","red"],
    "TORSO_LINEAR_MOMENTUM": ["TORSO","green"],
    "LEAD_UPA_LINEAR_MOMENTUM": ["UPPER ARM","blue"],
    "LEAD_FA_LINEAR_MOMENTUM": ["FOREARM","yellow"],
}

am_cols = {
    "PELVIS_ANGULAR_MOMENTUM": ["PELVIS","red"],
    "TORSO_ANGULAR_MOMENTUM": ["TORSO","green"],
    "LEAD_UPA_ANGULAR_MOMENTUM": ["UPPER ARM","blue"],
    "LEAD_FA_ANGULAR_MOMENTUM": ["FOREARM","yellow"],
}

energy_cols = {
    # "LEAD_SHANK_LINEAR_MOMENTUM": ["SHANK LINEAR MOMENTUM","shank_energy"],
    # "LEAD_THIGH_LINEAR_MOMENTUM": ["THIGH LINEAR MOMENTUM","thigh_energy"],
    "PELVIS_LINEAR_MOMENTUM": ["PELVIS LINEAR MOMENTUM","pelvis_energy"],
    "TORSO_LINEAR_MOMENTUM": ["TORSO LINEAR MOMENTUM","torso_energy"],
    "LEAD_UPA_LINEAR_MOMENTUM": ["ARM LINEAR MOMENTUM","arm_energy"],
    "LEAD_FA_LINEAR_MOMENTUM": ["FOREARM LINEAR MOMENTUM","forearm_energy"],
    # "LEAD_SHANK_ANGULAR_MOMENTUM": ["SHANK ANGULAR MOMENTUM","shank_energy"],
    # "LEAD_THIGH_ANGULAR_MOMENTUM": ["THIGH ANGULAR MOMENTUM","thigh_energy"],
    "PELVIS_ANGULAR_MOMENTUM_Z": ["PELVIS ANGULAR MOMENTUM","pelvis_energy"],
    "TORSO_ANGULAR_MOMENTUM_Z": ["TORSO ANGULAR MOMENTUM","torso_energy"],
    "LEAD_UPA_ANGULAR_MOMENTUM": ["ARM ANGULAR MOMENTUM","arm_energy"],
    "LEAD_FA_ANGULAR_MOMENTUM": ["FOREARM ANGULAR MOMENTUM","forearm_energy"],
    # "LEAD_SHANK_NET_SP" : ["SHANK POWER","shank_energy"],
    # "LEAD_THIGH_NET_SP" : ["THIGH POWER","thigh_energy"],
    "PELVIS_NET_SP" : ["PELVIS POWER","pelvis_energy"],
    "TORSO_NET_SP" : ["TORSO POWER","torso_energy"],
    "LEAD_ARM_NET_SP" : ["ARM POWER","arm_energy"],
    "LEAD_FOREARM_NET_SP" : ["FOREARM POWER","forearm_energy"],
}
# ============================ 그래프 및 시점 수치 =======================================
force_ap_fig, force_ap_values = grf_plotly(f_df, ap_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='ap')
force_vt_fig, force_vt_values = grf_plotly(f_df, vt_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='vt')
force_result_fig, force_result_values = grf_plotly(f_df, result_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='result')
force_momentum_fig, force_momentum_values = grf_plotly(f_df, momentum_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='freemoment')
kine_values, kine_fig = one_angle_plotly(k_df, ang_cols, k_time, k_kh_time1, k_fc_time, k_mer_time, k_br_time)
kinematic_values, kinematic_fig = kinematic_sequence_plotly(k_df, ks_cols, k_time, k_kh_time1, k_fc_time, k_mer_time, k_br_time)
segment_power_values, segment_power_fig = segment_power_plotly(k_df, sp_cols, k_time, k_kh_time1, k_fc_time, k_mer_time, k_br_time)
linear_momentum_values, linear_momentum_fig = linear_momentum_plotly(k_df, lm_cols, k_time, k_kh_time1, k_fc_time, k_mer_time, k_br_time)
angular_momentum_values, angular_momentum_fig = angular_momentum_plotly(k_df, am_cols, k_time, k_kh_time1, k_fc_time, k_mer_time, k_br_time)
xfactor_values, xfactor_fig = angular_momentum_plotly(k_df, xfactor_cols, k_time, k_kh_time1, k_fc_time, k_mer_time, k_br_time)
# energy_values, energy_fig = energy_plotly(k_df, energy_cols, k_time, k_kh_time1, k_fc_time, k_mer_time, k_br_time)


force_ap_fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
)
force_vt_fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
)
force_result_fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
)
force_momentum_fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
)
segment_power_fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
)
linear_momentum_fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
)
angular_momentum_fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
)
for col in kine_fig:
    fig = kine_fig[col]
    fig.update_layout(
    width=800,  # Set the width to your preference
    height=400  # Set the height to your preference
    )


peak_pel = round(kinematic_values['peak']['PELVIS_SEG_ANGULAR_VELOCITY_Z']); time_pel = kinematic_values['time']['PELVIS_SEG_ANGULAR_VELOCITY_Z']
peak_tor = round(kinematic_values['peak']['TORSO_SEG_ANGULAR_VELOCITY_Z']);time_tor = kinematic_values['time']['TORSO_SEG_ANGULAR_VELOCITY_Z']
peak_elb = round(kinematic_values['peak']['LEAD_ELBOW_ANGULAR_VELOCITY_X']);time_elb = kinematic_values['time']['LEAD_ELBOW_ANGULAR_VELOCITY_X']
peak_sho = round(kinematic_values['peak']['LEAD_SHOULDER_ANGULAR_VELOCITY_Z']);time_sho = kinematic_values['time']['LEAD_SHOULDER_ANGULAR_VELOCITY_Z']

pel_time = round(100 * (time_pel - k_fc_time) / k_total_time)
tor_time = round(100 * (time_tor - k_fc_time) / k_total_time)
elb_time = round(100 * (time_elb - k_fc_time) / k_total_time)
sho_time = round(100 * (time_sho - k_fc_time) / k_total_time)

tor_gain = round(peak_tor / peak_pel,2)
upper_gain = round(peak_elb / peak_tor,2)
fore_gain = round(peak_sho / peak_elb,2)

sq_time = [pel_time, tor_time, elb_time, sho_time]
expected_order = transform_list(sq_time)

data_as_dict = {
    "Segment": ["Pelvic", "Torso", "Elbow", "Shoulder"],
    "Peak": [f"{peak_pel} °/s", f"{peak_tor} °/s", f"{peak_elb} °/s", f"{peak_sho} °/s"],
    "Timing [FC-BR%]": [f"{pel_time} %", f"{tor_time} %", f"{elb_time} %", f"{sho_time} %"],
    "Sequence": expected_order,
    "Speed Gain": [0, tor_gain,upper_gain, fore_gain],
    "Pro": ["649 ~ 840", "987 ~ 1174", "2211 ~ 2710", "4331 ~ 4884"]
}
kinematic_sq = pd.DataFrame(data_as_dict)
kinematic_sq = kinematic_sq.set_index('Segment')
kinematic_sq['Speed Gain'] = kinematic_sq['Speed Gain'].astype(float).map('{:.2f}'.format)
kinematic_sq = kinematic_sq.style.set_properties(**{'text-align': 'center'})

sp_data_dict = {
    "Segment" : ["Pelvis", "Torso", "Upper Arm" , "ForeArm"],
    "at FC": [f"{round(segment_power_values['fc_time']['PELVIS_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['fc_time']['TORSO_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['fc_time']['LEAD_ARM_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['fc_time']['LEAD_FOREARM_NET_SP'],2)} W/kg"],
    "at MER": [f"{round(segment_power_values['mer_time']['PELVIS_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['mer_time']['TORSO_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['mer_time']['LEAD_ARM_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['mer_time']['LEAD_FOREARM_NET_SP'],2)} W/kg"],
    "at BR": [f"{round(segment_power_values['br_time']['PELVIS_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['br_time']['TORSO_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['br_time']['LEAD_ARM_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['br_time']['LEAD_FOREARM_NET_SP'],2)} W/kg"],
    "at MAX": [f"{round(segment_power_values['max']['PELVIS_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['max']['TORSO_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['max']['LEAD_ARM_NET_SP'],2)} W/kg", 
                f"{round(segment_power_values['max']['LEAD_FOREARM_NET_SP'],2)} W/kg"],
    "at MAX TIMING [FC-BR%]": [f"{round(100*(segment_power_values['max_time']['PELVIS_NET_SP'] - k_fc_time) / k_total_time)} %", 
                                f"{round(100*(segment_power_values['max_time']['TORSO_NET_SP'] - k_fc_time)/k_total_time)} %", 
                                f"{round(100*(segment_power_values['max_time']['LEAD_ARM_NET_SP'] - k_fc_time) / k_total_time)} %", 
                                f"{round(100*(segment_power_values['max_time']['LEAD_FOREARM_NET_SP'] - k_fc_time)/ k_total_time)} %"],
}
lm_data_dict = {
    "Segment" : ["Pelvis", "Torso", "Upper Arm" , "ForeArm"],
    "at FC": [f"{round(linear_momentum_values['fc_time']['PELVIS_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['fc_time']['TORSO_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['fc_time']['LEAD_UPA_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['fc_time']['LEAD_FA_LINEAR_MOMENTUM'],2)} kg•m/s"],
    "at MER": [f"{round(linear_momentum_values['mer_time']['PELVIS_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['mer_time']['TORSO_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['mer_time']['LEAD_UPA_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['mer_time']['LEAD_FA_LINEAR_MOMENTUM'],2)} kg•m/s"],
    "at BR": [f"{round(linear_momentum_values['br_time']['PELVIS_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['br_time']['TORSO_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['br_time']['LEAD_UPA_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['br_time']['LEAD_FA_LINEAR_MOMENTUM'],2)} kg•m/s"],
    "at MAX": [f"{round(linear_momentum_values['max']['PELVIS_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['max']['TORSO_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['max']['LEAD_UPA_LINEAR_MOMENTUM'],2)} kg•m/s", 
                f"{round(linear_momentum_values['max']['LEAD_FA_LINEAR_MOMENTUM'],2)} kg•m/s"],
    "at MAX TIMING [FC-BR%]": [f"{round(100 * (linear_momentum_values['max_time']['PELVIS_LINEAR_MOMENTUM'] - k_fc_time) / k_total_time)} %", 
                                f"{round(100 * (linear_momentum_values['max_time']['TORSO_LINEAR_MOMENTUM'] - k_fc_time)/k_total_time)} %", 
                                f"{round(100 * (linear_momentum_values['max_time']['LEAD_UPA_LINEAR_MOMENTUM'] - k_fc_time) / k_total_time)} %", 
                                f"{round(100 * (linear_momentum_values['max_time']['LEAD_FA_LINEAR_MOMENTUM'] - k_fc_time)/ k_total_time)} %"],
}
am_data_dict = {
    "Segment" : ["Pelvis", "Torso", "Upper Arm" , "ForeArm"],
    "at FC": [f"{round(angular_momentum_values['fc_time']['PELVIS_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['fc_time']['TORSO_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['fc_time']['LEAD_UPA_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['fc_time']['LEAD_FA_ANGULAR_MOMENTUM'],2)} kg•m²/s"],
    "at MER": [f"{round(angular_momentum_values['mer_time']['PELVIS_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['mer_time']['TORSO_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['mer_time']['LEAD_UPA_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['mer_time']['LEAD_FA_ANGULAR_MOMENTUM'],2)} kg•m²/s"],
    "at BR": [f"{round(angular_momentum_values['br_time']['PELVIS_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['br_time']['TORSO_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['br_time']['LEAD_UPA_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['br_time']['LEAD_FA_ANGULAR_MOMENTUM'],2)} kg•m²/s"],
    "at MAX": [f"{round(angular_momentum_values['max']['PELVIS_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['max']['TORSO_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['max']['LEAD_UPA_ANGULAR_MOMENTUM'],2)} kg•m²/s", 
                f"{round(angular_momentum_values['max']['LEAD_FA_ANGULAR_MOMENTUM'],2)} kg•m²/s"],
    "at MAX TIMING [FC-BR%]": [f"{round(100*(angular_momentum_values['max_time']['PELVIS_ANGULAR_MOMENTUM'] - k_fc_time) / k_total_time)} %", 
                                f"{round(100*(angular_momentum_values['max_time']['TORSO_ANGULAR_MOMENTUM'] - k_fc_time) / k_total_time)} %", 
                                f"{round(100*(angular_momentum_values['max_time']['LEAD_UPA_ANGULAR_MOMENTUM'] - k_fc_time) / k_total_time)} %", 
                                f"{round(100*(angular_momentum_values['max_time']['LEAD_FA_ANGULAR_MOMENTUM'] - k_fc_time) / k_total_time)} %"],
}

sp_df = pd.DataFrame(sp_data_dict)
sp_df = sp_df.set_index('Segment')
sp_df = sp_df.style.set_properties(**{'text-align': 'center'})
lm_df = pd.DataFrame(lm_data_dict)
lm_df = lm_df.set_index('Segment')
lm_df = lm_df.style.set_properties(**{'text-align': 'center'})
am_df = pd.DataFrame(am_data_dict)
am_df = am_df.set_index('Segment')
am_df = am_df.style.set_properties(**{'text-align': 'center'})
# ===================================================================================
# ============================= DashBoard ===========================================
page_tab1, page_tab2 = st.tabs(["데이터 보기", "피드백"])

with page_tab1:
    st.markdown('<a name="top"></a>', unsafe_allow_html=True)
    st.write('PARAMETERS')
    st.markdown("""
    <style>
    .fixed-top a {
        color: #10d5c2; /* A bright, sporty color for the links */
        padding: 8px 15px; /* Space around the links */
        text-decoration: none;
        font-size: 18px;
        line-height: 24px;
        border-radius: 5px; /* Rounded corners for links */
        transition: background-color 0.3s ease; /* Smooth transition for hover effect */
    }

    .fixed-top a:hover {
        background-color: #10d5c2; /* Background color on hover */
        color: #fff; /* Text color on hover */
        text-decoration: none; /* No underline on hover */
    }
    </style>
        """, unsafe_allow_html=True)

    # 고정된 상단 바에 넣을 링크들입니다.
    st.markdown("""
    <div class="fixed-top">
        <a href="#energy-flow">Energy Flow</a>
        <a href="#kinematic-sequence">Kinematic Sequence</a>
        <a href="#ball-release">Ball Release</a>
        <a href="#arm-acceleration">Arm Acceleration</a>
        <a href="#arm-cocking">Arm Cocking</a>
        <a href="#stride">Stride</a>
        <a href="#kinetic-parameters">Kinetic Parameters</a>
    </div>
    """, unsafe_allow_html=True)

    st.title("KMU BASEBALL PITCHING REPORT")
    cols = st.columns([1,1,1])
    with cols[0]:
        st.metric(label="Ball Speed", value=f"{ball_speed} km/h", delta=None)
    with cols[1]:
        st.metric(label="Pitching Type", value=f"{pit_type}", delta=None)
    with cols[2]:
        st.metric(label="Stride Length", value=f"{stride_length} %Height", delta=None)    
    
    ## 분석 구간
    st.markdown("""
        <style>
        .event_phase-header {
            background-color: #26282F; /* 박스의 배경 색상 */
            color: white; /* 텍스트 색상 */
            padding: 2; /* 안쪽 여백 */
            border-radius: 0px; /* 모서리 둥글기 */
            padding-left: 20px;
        }
        </style>
        <div class="event_phase-header">
            <h2>EVENTS & PHASES</h2>
        </div>
    """, unsafe_allow_html=True)
    st.image('image/analysis.png', use_container_width=True)
    
    st.empty()  # 상단에 빈 공간 추가
    ## ENERGY FLOW
    st.markdown("""
        <style>
        .energy_flow-parameters {
            background-color: #26282F; /* 박스의 배경 색상 */
            color: white; /* 텍스트 색상 */
            padding: 2px; /* 안쪽 여백 */
            border-radius: 0px; /* 모서리 둥글기 */
            padding-left: 20px;
        }
        </style>
        <div class="energy_flow-parameters">
            <h2>ENERGY FLOW</h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<a name="energy-flow"></a>', unsafe_allow_html=True)
    ef_taps = st.tabs(['LINEAR MOMENTUM', 'ANGULAR MOMENTUM','SEGMENT POWER'])
    with ef_taps[0]:
        col1, col2 = st.columns([1,2.8])
        with col1:
            st.image(f'image/energy_flow.png', use_container_width=True)
        with col2:
            st.plotly_chart(linear_momentum_fig, use_container_width=True,key='linear_momentum_fig')
        st.dataframe(lm_df, use_container_width=True)       

    with ef_taps[1]:
        col1, col2 = st.columns([1,2.8])
        with col1:
            st.image(f'image/energy_flow.png', use_container_width=True)
        with col2:
            st.plotly_chart(angular_momentum_fig, use_container_width=True,key='angular_momentum_fig')
        st.dataframe(am_df, use_container_width=True)       

    with ef_taps[2]:
        col1, col2 = st.columns([1,2.8])
        with col1:
            st.image(f'image/energy_flow.png', use_container_width=True)
        with col2:
            st.plotly_chart(segment_power_fig, use_container_width=True, key='segment_power_fig')
        st.dataframe(sp_df, use_container_width=True)       
    
    st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)
    # ## LINEAR MOMENTUM
    # st.subheader('Linear Momentum')
    # st.markdown('<a name="linear-momentum"></a>', unsafe_allow_html=True)
    # tabs_keys = ['LEAD_SHANK_LINEAR_MOMENTUM','LEAD_THIGH_LINEAR_MOMENTUM', 'PELVIS_LINEAR_MOMENTUM', 
    #              'TORSO_LINEAR_MOMENTUM', 'LEAD_UPA_LINEAR_MOMENTUM','LEAD_FA_LINEAR_MOMENTUM']
    # lm_taps = st.tabs(['SHANK', 'THIGH', 'PELVIS','TORSO', 'ARM', 'FOREARM'])
    # for tab, key in zip(lm_taps, tabs_keys):
    #     with tab:
    #         col1, col2 = st.columns([1,2.8])
    #         with col1:
    #             st.image(f'image/{energy_cols[key][1]}.png', use_container_width=True)
    #         with col2:
    #             st.plotly_chart(energy_fig[key], use_container_width=True)
    #         cols = st.columns([1,1,1,1,1])
    #         metrics = ['fc_time','mer_time','br_time','max','max_time']
    #         labels = ['At FC', 'At MER', 'At BR', 'Max', 'Max Time']
    #         values = [energy_values[m][key] for m in metrics]
    #         for i, (col, label, value) in enumerate(zip(cols, labels, values)):
    #             with col:
    #                 if metrics[i] in ['fc_time','br_time','mer_time','max']:
    #                     # Use Streamlit's metric for the rest of the values
    #                     st.metric(label=label, value=f"{value} kg•m²/s", delta=None)
    #                 elif metrics[i] in ['max_time']:
    #                     st.metric(label=label, value=f"{round(100 * (value - k_fc_time) / k_total_time)} %", delta=None) 
    # st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)
    
    # ## ANGULAR MOMENTUM
    # st.subheader('Angular Momentum')
    # st.markdown('<a name="angular-momentum"></a>', unsafe_allow_html=True)
    # tabs_keys = ['LEAD_SHANK_ANGULAR_MOMENTUM','LEAD_THIGH_ANGULAR_MOMENTUM', 'PELVIS_ANGULAR_MOMENTUM_Z', 
    #              'TORSO_ANGULAR_MOMENTUM_Z', 'LEAD_UPA_ANGULAR_MOMENTUM','LEAD_FA_ANGULAR_MOMENTUM']
    # am_taps = st.tabs(['SHANK', 'THIGH', 'PELVIS','TORSO', 'ARM', 'FOREARM'])
    # for tab, key in zip(am_taps, tabs_keys):
    #     with tab:
    #         col1, col2 = st.columns([1,2.8])
    #         with col1:
    #             st.image(f'image/{energy_cols[key][1]}.png', use_container_width=True)
    #         with col2:
    #             st.plotly_chart(energy_fig[key], use_container_width=True)
    #         cols = st.columns([1,1,1,1,1])    
    #         metrics = ['fc_time','mer_time','br_time','max','max_time']
    #         labels = ['At FC', 'At MER', 'At BR', 'Max', 'Max Time']
    #         values = [energy_values[m][key] for m in metrics]
    #         for i, (col, label, value) in enumerate(zip(cols, labels, values)):
    #             with col:
    #                 if metrics[i] in ['fc_time','br_time','mer_time','max']:
    #                     # Use Streamlit's metric for the rest of the values
    #                     st.metric(label=label, value=f"{value} kg•m²/(s•rad)", delta=None)
    #                 elif metrics[i] in ['max_time']:
    #                     st.metric(label=label, value=f"{round(100 * (value - k_fc_time) / k_total_time)} %", delta=None)
    # st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)
    
    # ## Segment Power
    # st.subheader('Segment Power')
    # st.markdown('<a name="segment-power"></a>', unsafe_allow_html=True)    
    # tabs_keys = ['LEAD_SHANK_NET_SP','LEAD_THIGH_NET_SP', 'PELVIS_NET_SP', 
    #              'TORSO_NET_SP', 'LEAD_ARM_NET_SP','LEAD_FOREARM_NET_SP']
    # sp_taps = st.tabs(['SHANK', 'THIGH', 'PELVIS','TORSO', 'ARM', 'FOREARM'])
    # for tab, key in zip(sp_taps, tabs_keys):
    #     with tab:
    #         col1, col2 = st.columns([1,2.8])
    #         with col1:
    #             st.image(f'image/{energy_cols[key][1]}.png', use_container_width=True)
    #         with col2:
    #             st.plotly_chart(energy_fig[key], use_container_width=True)
    #         cols = st.columns([1,1,1,1,1,1,1])    
    #         metrics = ['fc_time','mer_time','br_time','max','min','max_time','min_time']
    #         labels = ['At FC', 'At MER', 'At BR', 'Max', 'Min', 'Max Time','Min Time']
    #         values = [energy_values[m][key] for m in metrics]
    #         for i, (col, label, value) in enumerate(zip(cols, labels, values)):
    #             with col:
    #                 if metrics[i] in ['fc_time','br_time','mer_time','max','min']:
    #                     # Use Streamlit's metric for the rest of the values
    #                     st.metric(label=label, value=f"{value}", delta=None)
    #                 elif metrics[i] in ['max_time','min_time']:
    #                     st.metric(label=label, value=f"{round(100 * (value - k_fc_time) / k_total_time)} %", delta=None) 
    # st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)        

    # KINEMATICS PARAMETERS
    st.markdown("""
        <style>
        .kinematics-parameters {
            background-color: #26282F; /* 박스의 배경 색상 */
            color: white; /* 텍스트 색상 */
            padding: 2px; /* 안쪽 여백 */
            border-radius: 0px; /* 모서리 둥글기 */
            padding-left: 20px;
        }
        </style>
        <div class="kinematics-parameters">
            <h2>KINEMATICS PARAMETERS</h2>
        </div>
    """, unsafe_allow_html=True)
    
    ## KINEMATIC SEQUENCE
    st.markdown('<a name="kinematic-sequence"></a>', unsafe_allow_html=True)
    st.subheader('Kinematic Sequence')
    col1, col2 = st.columns([1,2.8])
    with col1:
        st.image('image/kinematic.png', use_container_width=True)
    with col2:
        st.plotly_chart(kinematic_fig, use_container_width=True,key='kinematic_fig')
    # Streamlit에서 테이블 형태로 DataFrame 표시.
    st.dataframe(kinematic_sq, use_container_width=True)
    st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)

    ## Ball Release
    st.markdown('<a name="ball-release"></a>', unsafe_allow_html=True)
    st.subheader('Ball Release')
    tabs_keys = ['ANKLE_HAND_AP_DISTANCE','LEAD_SHOULDER_ANGLE_Y','TORSO_ANGLE_X', 'TORSO_ANGLE_Y', 'LEAD_KNEE_ANGLE_X', 'LEAD_KNEE_ANGULAR_VELOCITY_X']
    br_taps = st.tabs(['Early Release','SHOULDER ABDUCTION', 'TRUNK FORWARD TILT', 'TRUNK LATERAL TILT','LEAD KNEE FLEXION','LEAD KNEE EXTENSION VELOCITY'])
    for tab, key in zip(br_taps, tabs_keys):
        if 'ANGLE' in key:
            unit = '°'
        elif 'ANGULAR' in key:
            unit = '°/s'
        else:
            unit = 'cm'    
        with tab:
            col1, col2 = st.columns([1,2.8])
            with col1:
                st.image(f'image/{ang_cols[key]}.png', use_container_width=True)
            with col2:
                st.plotly_chart(kine_fig[key], use_container_width=True, key=f"{key}.png")
            
            cols = st.columns([1,1,1,1,1])
            metrics = ['fc_time','mer_time','br_time','max','max_time']
            labels = ['At FC', 'At MER', 'At BR', 'At Max', 'At Max Time']
            values = [kine_values[m][key] for m in metrics]
            for i, (col, label, value) in enumerate(zip(cols, labels, values)):
                with col:
                    if metrics[i] == 'br_time':  # Highlight the 'At BR' value in red
                        # Customize as per your actual styling needs
                        st.markdown(
                            f"<div style='text-align: left;'><span style='font-size: 15px; color:red;'>{label}</span><br><span style='color: red; font-size: 36px;'>{value} {unit}</span></div>",
                            unsafe_allow_html=True
                        )
                    elif metrics[i] in ['fc_time','mer_time','max']:
                        # Use Streamlit's metric for the rest of the values
                        st.metric(label=label, value=f"{value} {unit}", delta=None) 
                    elif metrics[i] == 'max_time':
                        st.metric(label=label, value=f"{round(100 * (value - k_fc_time) / k_total_time)} %", delta=None) 
    st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)

    ## ARM ACCELERATION
    st.markdown('<a name="arm-acceleration"></a>', unsafe_allow_html=True)
    st.subheader('ARM ACCELERATION')
    tabs_keys = ['HAND_ELBOW_HEIGHT','LEAD_KNEE_ANGLE_X', 'LEAD_ELBOW_ANGLE_X']
    aa_taps = st.tabs(['HIGH HAND at MAX LAYBACK','LEAD KNEE FLEXION','ELBOW FLEXION'])       
    for tab, key in zip(aa_taps, tabs_keys):
        if 'ANGLE' in key:
            unit = '°'
        elif 'ANGULAR' in key:
            unit = '°/s'
        else:
            unit = 'cm'
            
        with tab:
            col1, col2 = st.columns([1,2.8])
            with col1:
                st.image(f'image/{ang_cols[key]}.png', use_container_width=True)
            with col2:
                st.plotly_chart(kine_fig[key], use_container_width=True,key=f"{key}")
            
            cols = st.columns([1,1,1,1,1])
            metrics = ['fc_time','mer_time','br_time','max','max_time']
            labels = ['At FC', 'At MER', 'At BR', 'At Max', 'At Max Time']
            values = [kine_values[m][key] for m in metrics]
            for i, (col, label, value) in enumerate(zip(cols, labels, values)):
                with col:
                    if metrics[i] in ['fc_time','mer_time','max','br_time']:
                        # Use Streamlit's metric for the rest of the values
                        st.metric(label=label, value=f"{value} {unit}", delta=None) 
                    elif metrics[i] == 'max_time':
                        st.metric(label=label, value=f"{round(100 * (value - k_fc_time) / k_total_time)} %", delta=None) 
    st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)

    ## ARM COCKING
    st.markdown('<a name="arm-cocking"></a>', unsafe_allow_html=True)
    st.subheader('ARM COCKING')
    tabs_keys = ['HAND_ELBOW_HEIGHT','PELVIS_TORSO_AP_DISTANCE','TORSO_PELVIS_ANGLE_Z','LEAD_SHOULDER_ANGLE_Z','LEAD_SHOULDER_ANGLE_X', 'LEAD_ELBOW_ANGLE_X','LEAD_KNEE_ANGLE_X']
    ac_taps = st.tabs(['LATE RISE','Getting Out in Front','X FACTOR','SHOULDER EXTERNAL ROTATION','SHOULDER HORIZONTAL ABDUCTION','ELBOW FLEXION','KNEE EXTENSION'])  
    for tab, key in zip(ac_taps, tabs_keys):
        if key != 'TORSO_PELVIS_ANGLE_Z':
            if 'ANGLE' in key:
                unit = '°'
            elif 'ANGULAR' in key:
                unit = '°/s'
            else:
                unit = 'cm'
            with tab:
                col1, col2 = st.columns([1,2.8])
                with col1:
                    st.image(f'image/{ang_cols[key]}.png', use_container_width=True)
                with col2:
                    st.plotly_chart(kine_fig[key], use_container_width=True, key=f"{key}.jpg")
                
                cols = st.columns([1,1,1,1,1])
                metrics = ['fc_time','mer_time','br_time','max','max_time']
                labels = ['At FC', 'At MER', 'At BR', 'At Max', 'At Max Time']
                values = [kine_values[m][key] for m in metrics]
                for i, (col, label, value) in enumerate(zip(cols, labels, values)):
                    with col:
                        if metrics[i] == 'fc_time':  # Highlight the 'At BR' value in red
                            # Customize as per your actual styling needs
                            st.markdown(
                                f"<div style='text-align: left;'><span style='font-size: 15px; color:red;'>{label}</span><br><span style='color: red; font-size: 36px;'>{value} {unit}</span></div>",
                                unsafe_allow_html=True
                            )
                        elif metrics[i] in ['br_time','mer_time','max']:
                            # Use Streamlit's metric for the rest of the values
                            st.metric(label=label, value=f"{value} {unit}", delta=None)
                        elif metrics[i] == 'max_time':
                            st.metric(label=label, value=f"{round(100 * (value - k_fc_time) / k_total_time)} %", delta=None) 
        else:
            unit = '°'
            with tab:
                col1, col2 = st.columns([1,2.8])
                with col1:
                    st.image(f'image/{ang_cols[key]}.png', use_container_width=True)
                with col2:
                    st.plotly_chart(xfactor_fig, use_container_width=True, key='xfactor_fig')
                cols = st.columns([1,1,1,1,1])
                metrics = ['fc_time','mer_time','br_time','max','max_time']
                labels = ['At FC', 'At MER', 'At BR', 'At Max', 'At Max Time']
                values = [kine_values[m][key] for m in metrics] 
                for i, (col, label, value) in enumerate(zip(cols, labels, values)):
                    with col:
                        if metrics[i] == 'fc_time':  # Highlight the 'At BR' value in red
                            # Customize as per your actual styling needs
                            st.markdown(
                                f"<div style='text-align: left;'><span style='font-size: 15px; color:red;'>{label}</span><br><span style='color: red; font-size: 36px;'>{value} {unit}</span></div>",
                                unsafe_allow_html=True
                            )
                        elif metrics[i] in ['br_time','mer_time','max']:
                            # Use Streamlit's metric for the rest of the values
                            st.metric(label=label, value=f"{value} {unit}", delta=None)
                        elif metrics[i] == 'max_time':
                            st.metric(label=label, value=f"{round(100 * (value - k_fc_time) / k_total_time)} %", delta=None)
                                
    st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)

    ## STRIDE
    st.markdown('<a name="stride"></a>', unsafe_allow_html=True)
    st.subheader('STRIDE')
    tabs_keys = ['PELVIS_ANGLE_Z','TORSO_ANGLE_Z','HAND_ELBOW_HEIGHT']
    st_taps = st.tabs(['PELVIS ROTATION', 'TRUNK ROTATION','LATE RISE'])  
    for tab, key in zip(st_taps, tabs_keys):
        if 'ANGLE' in key:
            unit = '°'
        elif 'ANGULAR' in key:
            unit = '°/s'
        else:
            unit = 'cm'
        with tab:
            col1, col2 = st.columns([1,2.8])
            with col1:
                st.image(f'image/{ang_cols[key]}.png', use_container_width=True)
            with col2:
                st.plotly_chart(kine_fig[key], use_container_width=True, key=f'{key}_stride')
            
            cols = st.columns([1,1,1,1,1])
            metrics = ['fc_time','mer_time','br_time','max','max_time']
            labels = ['At FC', 'At MER', 'At BR', 'At Max', 'At Max Time']
            values = [kine_values[m][key] for m in metrics]
            for i, (col, label, value) in enumerate(zip(cols, labels, values)):
                with col:
                    if metrics[i] in ['fc_time','br_time','mer_time','max']:
                        # Use Streamlit's metric for the rest of the values
                        st.metric(label=label, value=f"{value} {unit}", delta=None)
                    elif metrics[i] == 'max_time':
                        st.metric(label=label, value=f"{round(100 * (value - k_fc_time) / k_total_time)} %", delta=None) 
    st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)

    ## KINETICS PARAMETERS
    st.markdown('<a name="kinetic-parameters"></a>', unsafe_allow_html=True)
    st.markdown("""
        <style>
        .kinetics-parameters {
            background-color: #26282F; /* 박스의 배경 색상 */
            color: white; /* 텍스트 색상 */
            padding: 2px; /* 안쪽 여백 */
            border-radius: 0px; /* 모서리 둥글기 */
            padding-left: 20px;
        }
        </style>
        <div class="kinetics-parameters">
            <h2>KINETICS PARAMETERS</h2>
        </div>
    """, unsafe_allow_html=True)
    kinetics_tab = st.tabs(['GRF [AP AXIS]', 'GRF [VERTICAL AXIS]','GRF [RESULTANT]','TORQUE'])
    metrics = ['kh_time','fc_time','mer_time','br_time','max','max_time']
    labels = ['At KH','At FC', 'At MER', 'At BR', 'At Max', 'At Max Time']
    
    with kinetics_tab[0]:
        col1, col2 = st.columns([1,2.8])
        with col1:
            st.image('image/GRF_Y.png', use_container_width=True)
        with col2:
            st.plotly_chart(force_ap_fig, use_container_width=True, key=f"{force_ap_fig}")
        
        leg = ['Trail Leg', 'Stride Leg']
        force_key = ['REAR_FORCE_Y','LEAD_FORCE_Y']
        
        for key in force_key:    
            values = [force_ap_values[m][key] for m in metrics]
            color = ap_cols[key][-1]

            cols = st.columns([1,1,1,1,1,1,1])
            with cols[0]:
                st.markdown(f"<h1 style='font-size: 24px;'>{leg[0]} (%BW)</h1>", unsafe_allow_html=True)  # Change 24px as needed
                leg.pop(0)
                
            for i, (col, label, value) in enumerate(zip(cols[1:], labels, values)):
                with col:
                    if metrics[i] == 'max':  # Highlight the 'At BR' value in red
                        # Customize as per your actual styling needs
                        st.markdown(
                            f"<div style='text-align: left;'><span style='font-size: 15px; color:{color};'>{label}</span><br><span style='color: {color}; font-size: 36px;'>{value}</span></div>",
                            unsafe_allow_html=True
                        )
                    elif metrics[i] in ['kh_time','fc_time','mer_time','br_time']:
                        # Use Streamlit's metric for the rest of the values
                        st.metric(label=label, value=f"{value}", delta=None) 
                    elif metrics[i] == 'max_time':
                        st.metric(label=label, value=f"{round(100 * (value - f_fc_time)/(f_br_time+1 - f_fc_time))} %", delta=None)   
    with kinetics_tab[1]:
        col1, col2 = st.columns([1,2.8])
        with col1:
            st.image('image/GRF_Z.png', use_container_width=True)
        with col2:
            st.plotly_chart(force_vt_fig, use_container_width=True, key="force_vt_fig")
        
        
        force_key = ['REAR_FORCE_Z','LEAD_FORCE_Z']
        leg = ['Trail Leg', 'Stride Leg']
        for key in force_key:    
            values = [force_vt_values[m][key] for m in metrics]
            color = vt_cols[key][-1]

            cols = st.columns([1,1,1,1,1,1,1])
            with cols[0]:
                st.markdown(f"<h1 style='font-size: 24px;'>{leg[0]} (%BW)</h1>", unsafe_allow_html=True)  # Change 24px as needed
                leg.pop(0)
                
            for i, (col, label, value) in enumerate(zip(cols[1:], labels, values)):
                with col:
                    if metrics[i] == 'max':  # Highlight the 'At BR' value in red
                        # Customize as per your actual styling needs
                        st.markdown(
                            f"<div style='text-align: left;'><span style='font-size: 15px; color:{color};'>{label}</span><br><span style='color: {color}; font-size: 36px;'>{value}</span></div>",
                            unsafe_allow_html=True
                        )
                    elif metrics[i] in ['kh_time','fc_time','mer_time','br_time']:
                        # Use Streamlit's metric for the rest of the values
                        st.metric(label=label, value=f"{value}", delta=None) 
                    elif metrics[i] == 'max_time':
                        st.metric(label=label, value=f"{round(100 * (value - f_fc_time)/(f_br_time+1 - f_fc_time))} %", delta=None) 
    with kinetics_tab[2]:
        col1, col2 = st.columns([1,2.8])
        with col1:
            st.image('image/GRF_R.png', use_container_width=True)
        with col2:
            st.plotly_chart(force_result_fig, use_container_width=True, key="force_result_fig")
        
        
        force_key = ['REAR_RESULT','LEAD_RESULT']
        leg = ['Trail Leg', 'Stride Leg']
        for key in force_key:    
            values = [force_result_values[m][key] for m in metrics]
            color = result_cols[key][-1]

            cols = st.columns([1,1,1,1,1,1,1])
            with cols[0]:
                st.markdown(f"<h1 style='font-size: 24px;'>{leg[0]} (%BW)</h1>", unsafe_allow_html=True)  # Change 24px as needed
                leg.pop(0)
                
            for i, (col, label, value) in enumerate(zip(cols[1:], labels, values)):
                with col:
                    if metrics[i] == 'max':  # Highlight the 'At BR' value in red
                        # Customize as per your actual styling needs
                        st.markdown(
                            f"<div style='text-align: left;'><span style='font-size: 15px; color:{color};'>{label}</span><br><span style='color: {color}; font-size: 36px;'>{value}</span></div>",
                            unsafe_allow_html=True
                        )
                    elif metrics[i] in ['kh_time','fc_time','mer_time','br_time']:
                        # Use Streamlit's metric for the rest of the values
                        st.metric(label=label, value=f"{value}", delta=None) 
                    elif metrics[i] == 'max_time':
                        st.metric(label=label, value=f"{round(100 * (value - f_fc_time)/(f_br_time+1 - f_fc_time))} %", delta=None) 
    with kinetics_tab[3]:
        col1, col2 = st.columns([1,2.8])
        with col1:
            st.image('image/GRF_FREE.png', use_container_width=True)
        with col2:
            st.plotly_chart(force_momentum_fig, use_container_width=True, key='force_momentum_fig')
        
        
        force_key = ['REAR_FREEMOMENT_Z','LEAD_FREEMOMENT_Z']
        leg = ['Trail Leg', 'Stride Leg']
        for key in force_key:    
            values = [force_momentum_values[m][key] for m in metrics]
            color = momentum_cols[key][-1]

            cols = st.columns([1,1,1,1,1,1,1])
            with cols[0]:
                st.markdown(f"<h1 style='font-size: 24px;'>{leg[0]} (N*m)</h1>", unsafe_allow_html=True)  # Change 24px as needed
                leg.pop(0)
                
            
            for i, (col, label, value) in enumerate(zip(cols[1:], labels, values)):
                with col:
                    if metrics[i] == 'max':  # Highlight the 'At BR' value in red
                        # Customize as per your actual styling needs
                        st.markdown(
                            f"<div style='text-align: left;'><span style='font-size: 15px; color:{color};'>{label}</span><br><span style='color: {color}; font-size: 36px;'>{value}</span></div>",
                            unsafe_allow_html=True
                        )
                    elif metrics[i] in ['kh_time','fc_time','mer_time','br_time']:
                        # Use Streamlit's metric for the rest of the values
                        st.metric(label=label, value=f"{value}", delta=None) 
                    elif metrics[i] == 'max_time':
                        st.metric(label=label, value=f"{round(100 * (value - f_fc_time)/(f_br_time+1 - f_fc_time))} %", delta=None) 

    st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)

with page_tab2:  
    # 피드백 제출 버튼
    # st.subheader("피드백 남기기")
    # feedback_input = st.text_area("피드백을 남겨주세요:")
    # if st.session_state['selected_name'] in ['kookmin','yongseok']:
    #     if st.button('제출'):
    #         feedback_df = save_feedback(feedback_df, csv_file, selected_name, selected_date, feedback_input)
    
    st.subheader('저장된 피드백')
    try:
        filtered_feedback = [i for i in feedback_df[(feedback_df['name'] == selected_name) & 
                                    (feedback_df['date'] == int(selected_date))]['feedback'].values]
    except IndexError:
        filtered_feedback = ''
        
    st.write(filtered_feedback)