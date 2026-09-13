import json,numpy as np
from check_mappings import run,OUT
out=OUT/'modified_offset_results.json';rows=json.loads(out.read_text())
cases=[('original_mapping','[-FOOT_X, FOOT_Y, FOOT_Z]','[FOOT_X, FOOT_Y, FOOT_Z]',[-.103,-.113,-.123,-.133,-.143,-.153]),('closest_alternative','[FOOT_Z, -FOOT_Y, FOOT_X]','[FOOT_X, -FOOT_Y, FOOT_Z]',[-.103,-.113,-.123,-.133]),('original_com_held_at_baseline','[-.004, -.023, 0]','[.004, -.023, 0]',[-.033,-.043,-.053,-.063,-.073,-.083,-.093]),('aligned_com_held_at_baseline','[0, .023, .004]','[.004, -.023, 0]',[-.033,-.043,-.053,-.063,-.073,-.083,-.093])]
for label,lb,rb,ys in cases:
 for y in ys:
  trace,row=run(lb,rb,model_path='modified_model.xml',foot_y=y)
  row.update(case=label,pitch_at_gait_start_deg=float(np.rad2deg(trace[150,4])),hip_at_gait_start_rad=float(trace[150,3]),net_gait_displacement_m=float(np.linalg.norm(trace[-1,:2]-trace[150,:2])),final_yaw_change_deg=float(np.rad2deg(np.unwrap(trace[:,6])[-1]-trace[150,6])))
  rows.append(row);out.write_text(json.dumps(rows,indent=2))
  print(label,y,'upright',row['upright'],'distance',round(row['distance_m'],3),'lean',round(row['pitch_at_gait_start_deg'],2),'hip',round(row['hip_at_gait_start_rad'],3),flush=True)
