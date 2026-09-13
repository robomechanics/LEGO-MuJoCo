import json,numpy as np
from check_mappings import run,OUT
mapping_rows=json.loads((OUT/'results.json').read_text())['trials']
rows=json.loads((OUT/'modified_offset_results.json').read_text())
seen={(r['left_body'],r['right_body'],r['foot_y']) for r in rows}
candidates=[r for r in mapping_rows if r['settle']=='original' and r['upright']]
for i,c in enumerate(candidates,1):
 for y in [-.033,-.053,-.073,-.093]:
  if (c['left_body'],c['right_body'],y) in seen:continue
  trace,row=run(c['left_body'],c['right_body'],model_path='modified_model.xml',foot_y=y)
  row.update(case='original_survivor_mapping',pitch_at_gait_start_deg=float(np.rad2deg(trace[150,4])),hip_at_gait_start_rad=float(trace[150,3]),net_gait_displacement_m=float(np.linalg.norm(trace[-1,:2]-trace[150,:2])),final_yaw_change_deg=float(np.rad2deg(np.unwrap(trace[:,6])[-1]-trace[150,6])))
  rows.append(row)
  if row['upright']:print('SURVIVOR',json.dumps(row),flush=True)
 (OUT/'modified_offset_results.json').write_text(json.dumps(rows,indent=2));print('mapping',i,'of',len(candidates),'total trials',len(rows),'survivors',sum(r['upright'] for r in rows),flush=True)
# Fine check matching the original lean with fixed CoM coordinates.
for y in [-.098,-.103,-.108,-.113]:
 trace,row=run('[0, .023, .004]','[.004, -.023, 0]',model_path='modified_model.xml',foot_y=y)
 row.update(case='aligned_com_held_at_baseline',pitch_at_gait_start_deg=float(np.rad2deg(trace[150,4])),hip_at_gait_start_rad=float(trace[150,3]),net_gait_displacement_m=float(np.linalg.norm(trace[-1,:2]-trace[150,:2])),final_yaw_change_deg=float(np.rad2deg(np.unwrap(trace[:,6])[-1]-trace[150,6])))
 rows.append(row);print('fine',y,'upright',row['upright'],'distance',row['distance_m'],'lean',row['pitch_at_gait_start_deg'],flush=True)
(OUT/'modified_offset_results.json').write_text(json.dumps(rows,indent=2))
