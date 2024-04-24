def data_concat():
    import numpy as np
    from glob import glob
    import pandas as pd

    KINEMATIC_PATH = 'data/**/kine/*csv'
    FORCE_PATH = 'data/**/force/*csv'

    KINEMATIC_DIR = [i.replace('\\','/') for i in glob(KINEMATIC_PATH)]
    FORCE_DIR = [i.replace('\\','/') for i in glob(FORCE_PATH)]
    energy_cols = [
            "REAR_SHANK_NET_SP",    
            "LEAD_SHANK_NET_SP",
            "LEAD_THIGH_NET_SP",
            "REAR_THIGH_NET_SP",
            "PELVIS_NET_SP",
            "TORSO_NET_SP",
            "LEAD_ARM_NET_SP",
            "REAR_ARM_NET_SP",
            "LEAD_FOREARM_NET_SP",
            "REAR_FOREARM_NET_SP",
            ]
    
    kdf = pd.DataFrame()
    fdf = pd.DataFrame()

    kdf = pd.DataFrame()
    fdf = pd.DataFrame()

    for kine_dir, force_dir in zip(KINEMATIC_DIR, FORCE_DIR):
        kine = pd.read_csv(kine_dir)
        force = pd.read_csv(force_dir)
        
        _, kday, _, kfname = kine_dir.split('/')
        _, fday, _, ffname = force_dir.split('/')
        kfname = kfname.replace('.csv','')
        kplayer_name, ktrial, _, mass, _, _, kball, _,kpit_type = kfname.split('_')
        
        ffname = ffname.replace('.csv','')
        fplayer_name, ftrial, _, _, _, _, fball, _,fpit_type = ffname.split('_')
        
        lhjc = kine[['LEAD_HIP_JC_X','LEAD_HIP_JC_Y','LEAD_HIP_JC_Z']].values
        rhjc = kine[['REAR_HIP_JC_X','REAR_HIP_JC_Y','REAR_HIP_JC_Z']].values
        rsjc = kine[['REAR_SHOULDER_JC_X','REAR_SHOULDER_JC_Y','REAR_SHOULDER_JC_Z']].values
        lsjc = kine[['LEAD_SHOULDER_JC_X','LEAD_SHOULDER_JC_Y','LEAD_SHOULDER_JC_Z']].values
        pelvis_jc = pd.DataFrame((lhjc + rhjc)/2, columns = ['PELVIS_JC_X','PELVIS_JC_Y','PELVIS_JC_Z'])
        torso_jc = pd.DataFrame((rsjc + lsjc)/2, columns = ['TORSO_JC_X','TORSO_JC_Y','TORSO_JC_Z'])
        torso_pelvis_distance = torso_jc['TORSO_JC_Y'] - pelvis_jc['PELVIS_JC_Y']
        
        kine[energy_cols] = kine[energy_cols]/float(mass)
        kine['ANKLE_HAND_AP_DISTANCE'] = round(100 * (kine['LEAD_WRIST_JC_Y']-kine['LEAD_ANKLE_JC_Y']),1)
        kine['PELVIS_TORSO_AP_DISTANCE'] = round(100 * torso_pelvis_distance,1)
        
        kine['player'] = kplayer_name
        kine['day'] = kday
        kine['trial'] = ktrial
        kine['ball_speed'] = kball
        kine['pit_type'] = kpit_type

        force['player'] = fplayer_name
        force['day'] = fday
        force['trial'] = ftrial
        force['ball_speed'] = fball
        force['pit_type'] = fpit_type
        kdf = pd.concat([kdf, kine])
        fdf = pd.concat([fdf, force])
        
    return kdf, fdf