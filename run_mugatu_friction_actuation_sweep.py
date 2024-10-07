import mujoco as mjc
import mujoco_viewer as mjcv
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

old_robot_path = 'Mugatu/mugatu.xml'
new_robot_path = 'Mugatu/mugatu2.xml'
new_scene_path = 'Mugatu/scene2.xml'

new_foot_mass = '0.13'
# com_height = 0.066
joint_height = 0.15
# leg_amp_deg = 42.2
# leg_amp_rad = np.deg2rad(leg_amp_deg)
# hip_omega = np.sqrt(9.81/(joint_height - com_height))

robot_tree = ET.parse(old_robot_path)
robot_root = robot_tree.getroot()
right_foot_body = robot_root.find(".//body[@name='right_foot']")
left_foot_body = robot_root.find(".//body[@name='left_foot']")
right_foot_body.find('inertial').set('mass', new_foot_mass)
left_foot_body.find('inertial').set('mass', new_foot_mass)

robot_tree.write(new_robot_path)
max_time_range = 25

f_slide_params, amp_params, freq_params = (25,10,25)
f_slide_range = np.linspace(0.1, 2, f_slide_params)
amp_range = np.linspace(24.2 * 0.9, 42.2 * 1.1, amp_params)
amp_range_rad = np.deg2rad(amp_range)
freq_range = np.linspace(1.3, 1.8, freq_params)
freq_range_rad = np.deg2rad(freq_range)

tot_params = f_slide_params*amp_params*freq_params

param_data = np.zeros((tot_params,4))
count = 0

for cnt_slide, f_slide in enumerate(f_slide_range):
    for cnt_amp, amp in enumerate(amp_range_rad):
        for cnt_freq, freq in enumerate(freq_range):
            count += 1
            failed = False
            model = mjc.MjModel.from_xml_path(new_scene_path)
            data = mjc.MjData(model)
            model.opt.timestep = 0.001

            for item in model.geom_friction:
                item[0] = f_slide

            mjc.mj_step(model, data)
            trial_init_pos = data.qpos.copy()

            while data.time < max_time_range:
                mjc.mj_step(model, data)
                if data.time > 3:
                    data.actuator("hip_joint_act").ctrl = amp * \
                        np.sin(freq*data.time)
                if data.qpos[2] < joint_height / 3:
                    print(f"fell! ({count}/{tot_params})")
                    failed = True
                    param_data[count-1,:] = [0,f_slide, amp_range[cnt_amp], freq_range[cnt_freq]]
                    break
                
            if not failed:
                print(f"done! ({count}/{tot_params})")
                dist_traveled = np.linalg.norm(data.qpos[0:2] - trial_init_pos[0:2])
                param_data[count-1,:] = [dist_traveled,f_slide, amp_range[cnt_amp], freq_range[cnt_freq]]

np.savetxt('friction_act_sweep.csv', param_data, delimiter=',')



            