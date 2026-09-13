import json,numpy as np
from check_mappings import run,OUT
p=OUT/'modified_offset_results.json';rows=json.loads(p.read_text());new=[]
for y in np.round(np.arange(-.099,-.1175,-.001),6):
 if any(r['case']=='aligned_com_held_at_baseline' and abs(r['foot_y']-y)<1e-8 for r in rows):continue
 trace,row=run('[0, .023, .004]','[.004, -.023, 0]',model_path='modified_model.xml',foot_y=float(y))
 row.update(case='aligned_com_held_at_baseline',pitch_at_gait_start_deg=float(np.rad2deg(trace[150,4])),hip_at_gait_start_rad=float(trace[150,3]),net_gait_displacement_m=float(np.linalg.norm(trace[-1,:2]-trace[150,:2])),final_yaw_change_deg=float(np.rad2deg(np.unwrap(trace[:,6])[-1]-trace[150,6])))
 rows.append(row);new.append(row);p.write_text(json.dumps(rows,indent=2))
 print(y,'survive',row['upright'],'drop',row['first_drop_s'],'distance',round(row['distance_m'],3),flush=True)
for r in new:
 if r['upright']:
  _,long=run(r['left_body'],r['right_body'],model_path='modified_model.xml',foot_y=r['foot_y'],duration=60)
  long['case']='60_second_validation'
  rows.append(long);p.write_text(json.dumps(rows,indent=2));print('60 SECOND',json.dumps(long),flush=True)
