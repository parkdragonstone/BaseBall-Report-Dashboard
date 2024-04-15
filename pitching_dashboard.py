import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import numpy as np
from glob import glob
import data_concat
from graph_data import check_credentials,show_login_form,transform_list, grf_plotly, one_angle_plotly, kinematic_sequence_plotly


st.set_page_config(page_title = "KMU BASEBALL PITCHING REPORT", 
                layout="wide"
                    )
@st.cache_data
def load_data():
    kdf, fdf = data_concat.data_concat()
    return kdf, fdf

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# 사용자가 인증되지 않았다면 로그인 폼을 보여줌
if not st.session_state['authenticated']:
    show_login_form()
    
else:
    kdf, fdf = load_data()  
    kdf['trial'] = kdf['trial'].astype(int)
    fdf['trial'] = fdf['trial'].astype(int)

    # 스트림릿 사이드바 설정
    if st.session_state['selected_name'] in ['kookmin']:
        unique_names = kdf['player'].unique()
    else:
        unique_names = [st.session_state['selected_name']]
        
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
        'REAR_FORCE_Y' : ['Trail Leg' , 'blue'],
        'LEAD_FORCE_Y' : ['Stride Leg', 'red'],
    }
    vt_cols = {
        'REAR_FORCE_Z' : ['Trail Leg' , 'blue'],
        'LEAD_FORCE_Z' : ['Stride Leg', 'red']
    }
    result_cols = {
        'REAR_RESULT' : ['Trail Leg' , 'blue'],
        'LEAD_RESULT' : ['Stride Leg', 'red']
    }
    momentum_cols = {
        'REAR_MOMENTUM_Y' : ['Trail Leg' , 'blue'],
        'LEAD_MOMENTUM_Y' : ['Stride Leg', 'red']
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
        'LEAD_KNEE_ANGLE_X'               : 'LEAD KNEE FLEXION',
        'LEAD_KNEE_ANGULAR_VELOCITY_X'    : 'LEAD KNEE EXTENSION ANGULAR VELOCITY',
        'LEAD_SHOULDER_ANGLE_Y'           : 'SHOULDER ABDUCTION', 
        'TORSO_ANGLE_X'                   : 'TRUNK FORWARD TILT',
        'TORSO_ANGLE_Y'                   : 'TRUNK LATERAL TILT',
        'HAND_ELBOW_HEIGHT'               : 'HAND ELBOW HEIGHT',
        'TORSO_ANGLE_Z'                   : 'TRUNK ROTATION',
    }

    # ============================ 그래프 및 시점 수치 =======================================
    force_ap_fig, force_ap_values = grf_plotly(f_df, ap_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='ap')
    force_vt_fig, force_vt_values = grf_plotly(f_df, vt_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='vt')
    force_result_fig, force_result_values = grf_plotly(f_df, result_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='result')
    force_momentum_fig, force_momentum_values = grf_plotly(f_df, momentum_cols, f_time, f_kh_time1, f_fc_time, f_mer_time, f_br_time, axis='momentum')
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
    force_result_fig.update_layout(
        width=800,  # Set the width to your preference
        height=400  # Set the height to your preference
    )
    force_momentum_fig.update_layout(
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


    peak_pel = round(kinematic_values['peak']['PELVIS_ANGLUAR_VELOCITY_Z']); time_pel = kinematic_values['time']['PELVIS_ANGLUAR_VELOCITY_Z']
    peak_tor = round(kinematic_values['peak']['TORSO_ANGLUAR_VELOCITY_Z']);time_tor = kinematic_values['time']['TORSO_ANGLUAR_VELOCITY_Z']
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
        "Segment": ["Pelvic [°/s]", "Torso [°/s]", "Elbow [°/s]", "Shoulder [°/s]"],
        "Pro": ["649 ~ 840", "987 ~ 1174", "2211 ~ 2710", "4331 ~ 4884"],
        "Peak": [peak_pel, peak_tor, peak_elb, peak_sho],
        "Timing": [f"{pel_time} %", f"{tor_time} %", f"{elb_time} %", f"{sho_time} %"],
        "Sequence": expected_order,
        "Speed Gain": [0, tor_gain,upper_gain, fore_gain]
    }
    kinematic_sq = pd.DataFrame(data_as_dict)
    kinematic_sq = kinematic_sq.set_index('Segment')
    kinematic_sq['Speed Gain'] = kinematic_sq['Speed Gain'].astype(float).map('{:.2f}'.format)
    kinematic_sq = kinematic_sq.style.set_properties(**{'text-align': 'center'})
    # ===================================================================================
    # ============================= DashBoard ===========================================
    page_tab1, page_tab2 = st.tabs(["데이터 보기", "피드백 남기기"])

    with page_tab1:
        st.markdown('<a name="top"></a>', unsafe_allow_html=True)
        st.write('PARAMTERS')
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
        
        # 분석 구간
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
        st.image('image/analysis.png', use_column_width=True)
        
        
        st.empty()  # 상단에 빈 공간 추가
        # ENERGY FLOW
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

        st.empty()  # 상단에 빈 공간 추가
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
        
        st.markdown('<a name="kinematic-sequence"></a>', unsafe_allow_html=True)
        st.subheader('Kinematic Sequence')
        col1, col2 = st.columns([1,2.8])
        with col1:
            st.image('image/kinematic.png', use_column_width=True)
        with col2:
            st.plotly_chart(kinematic_fig, use_container_width=True)
        # Streamlit에서 테이블 형태로 DataFrame 표시.
        st.dataframe(kinematic_sq, use_container_width=True)
        st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)

        st.markdown('<a name="ball-release"></a>', unsafe_allow_html=True)
        st.subheader('Ball Release')
        tabs_keys = ['LEAD_SHOULDER_ANGLE_Y','TORSO_ANGLE_X', 'TORSO_ANGLE_Y', 'LEAD_KNEE_ANGLE_X', 'LEAD_KNEE_ANGULAR_VELOCITY_X']
        br_taps = st.tabs(['SHOULDER ABDUCTION', 'TRUNK FORWARD TILT', 'TRUNK LATERAL TILT','LEAD KNEE FLEXION','LEAD KNEE EXTENSION VELOCITY'])
        for tab, key in zip(br_taps, tabs_keys):
            if 'ANGLE' in key:
                unit = '°'
            elif 'ANGULAR' in key:
                unit = '°/s'
            with tab:
                col1, col2 = st.columns([1,2.8])
                with col1:
                    st.image(f'image/{ang_cols[key]}.png', use_column_width=True)
                with col2:
                    st.plotly_chart(kine_fig[key], use_container_width=True)
                
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

        st.markdown('<a name="arm-acceleration"></a>', unsafe_allow_html=True)
        st.subheader('ARM ACCELERATION')
        tabs_keys = ['LEAD_KNEE_ANGLE_X','HAND_ELBOW_HEIGHT', 'LEAD_ELBOW_ANGLE_X']
        aa_taps = st.tabs(['LEAD KNEE FLEXION','HIGH HAND at MAX LAYBACK','ELBOW FLEXION'])       
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
                    st.image(f'image/{ang_cols[key]}.png', use_column_width=True)
                with col2:
                    st.plotly_chart(kine_fig[key], use_container_width=True)
                
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

        st.markdown('<a name="arm-cocking"></a>', unsafe_allow_html=True)
        st.subheader('ARM COCKING')
        tabs_keys = ['TORSO_PELVIS_ANGLE_Z','HAND_ELBOW_HEIGHT','LEAD_SHOULDER_ANGLE_Z','LEAD_SHOULDER_ANGLE_X', 'LEAD_ELBOW_ANGLE_X','LEAD_KNEE_ANGLE_X']
        ac_taps = st.tabs(['X FACTOR','LATE RISE','SHOULDER EXTERNAL ROTATION','SHOULDER HORIZONTAL ABDUCTION','ELBOW FLEXION','KNEE EXTENSION'])  
        for tab, key in zip(ac_taps, tabs_keys):
            if 'ANGLE' in key:
                unit = '°'
            elif 'ANGULAR' in key:
                unit = '°/s'
            else:
                unit = 'cm'
            with tab:
                col1, col2 = st.columns([1,2.8])
                with col1:
                    st.image(f'image/{ang_cols[key]}.png', use_column_width=True)
                with col2:
                    st.plotly_chart(kine_fig[key], use_container_width=True)
                
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

        st.markdown('<a name="stride"></a>', unsafe_allow_html=True)
        st.subheader('STRIDE')
        tabs_keys = ['TORSO_ANGLE_Z','HAND_ELBOW_HEIGHT']
        st_taps = st.tabs(['TRUNK ROTATION','LATE RISE'])  
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
                    st.image(f'image/{ang_cols[key]}.png', use_column_width=True)
                with col2:
                    st.plotly_chart(kine_fig[key], use_container_width=True)
                
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

        st.empty()  # 상단에 빈 공간 추가
        # KINETICS PARAMETERS
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
        kinetics_tab = st.tabs(['GRF [AP AXIS]', 'GRF [VERTICAL AXIS]','GRF [RESULTANT]','MOMENTUM [AP AXIS]'])
        metrics = ['kh_time','fc_time','mer_time','br_time','max','max_time']
        labels = ['At KH','At FC', 'At MER', 'At BR', 'At Max', 'At Max Time']
        
        with kinetics_tab[0]:
            col1, col2 = st.columns([1,2.8])
            with col1:
                st.image('image/GRF_Y.png', use_column_width=True)
            with col2:
                st.plotly_chart(force_ap_fig, use_container_width=True)
            
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
                st.image('image/GRF_Z.png', use_column_width=True)
            with col2:
                st.plotly_chart(force_vt_fig, use_container_width=True)
            
            
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
                st.image('image/GRF_R.png', use_column_width=True)
            with col2:
                st.plotly_chart(force_result_fig, use_container_width=True)
            
            
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
                st.image('image/GRF_Y.png', use_column_width=True)
            with col2:
                st.plotly_chart(force_momentum_fig, use_container_width=True)
            
            
            
            force_key = ['REAR_MOMENTUM_Y','LEAD_MOMENTUM_Y']
            for key in force_key:
                if key == 'REAR_MOMENTUM_Y':
                    leg = 'Trail Leg'
                    metrics = ['kh_time','fc_time','mer_time','br_time','fc_time']
                elif key == 'LEAD_MOMENTUM_Y':
                    leg = 'Stride Leg'
                    metrics = ['kh_time','fc_time','mer_time','br_time','br_time']
                values = [force_momentum_values[m][key] for m in metrics]
                color = momentum_cols[key][-1]

                cols = st.columns([1,1,1,1,1,1])
                with cols[0]:
                    st.markdown(f"<h1 style='font-size: 24px;'>{leg} (N*s/BW)</h1>", unsafe_allow_html=True)  # Change 24px as needed

                    
                
                labels = ['At KH','At FC', 'At MER', 'At BR', 'Total Momentum']  
                for i, (col, label, value) in enumerate(zip(cols[1:], labels, values)):
                    with col:
                        if labels[i] == 'Total Momentum':  # Highlight the 'At BR' value in red
                            # Customize as per your actual styling needs
                            st.markdown(
                                f"<div style='text-align: left;'><span style='font-size: 15px; color:{color};'>{label}</span><br><span style='color: {color}; font-size: 36px;'>{value}</span></div>",
                                unsafe_allow_html=True
                            )
                        elif metrics[i] in ['kh_time','fc_time','mer_time','br_time']:
                            # Use Streamlit's metric for the rest of the values
                            st.metric(label=label, value=f"{value}", delta=None) 
        
        st.markdown('<a href="#top" class="scrollToTop">Go to top</a>', unsafe_allow_html=True)

    with page_tab2:  
        # 사용자 피드백 받기
        # Streamlit 세션 상태 초기화
        # 각 선택에 따른 키 생성 함수
        def create_key(player, date, trial):
            return f"{player}_{date}_{trial}_feedback"

        # 피드백 저장 함수
        def save_feedback(key, feedback):
            st.session_state[key] = feedback

        # 피드백 제출 처리
        def handle_feedback_submit(player, date, trial):
            key = create_key(player, date, trial)
            feedback = st.session_state.feedback
            save_feedback(key, feedback)
            st.success("피드백이 저장되었습니다.")

        # 피드백 입력
        feedback_input = st.text_area("피드백을 남겨주세요:", key='feedback')

        # 피드백 제출 버튼
        if st.button('제출'):
            handle_feedback_submit(selected_name, selected_date, selected_trial)

        # 저장된 피드백 표시
        feedback_key = create_key(selected_name, selected_date, selected_trial)
        if feedback_key in st.session_state:
            st.subheader('저장된 피드백')
            st.write(st.session_state[feedback_key])