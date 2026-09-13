"""Compare backward foot offsets with the original successful closed-loop gait."""
import json
from pathlib import Path
import numpy as np
from check_mappings import run,OUT
cases=[('original_mapping','[-FOOT_X, FOOT_Y, FOOT_Z]','[FOOT_X, FOOT_Y, FOOT_Z]'),('closest_alternative','[FOOT_Z, -FOOT_Y, FOOT_X]','[FOOT_X, -FOOT_Y, FOOT_Z]'),('matched_foot_com','[FOOT_Z, -FOOT_Y, FOOT_X]','[FOOT_X, FOOT_Y, FOOT_Z]')]
rows=[]
for label,lb,rb in cases:
 for y in [-.023,-.033,-.043,-.053,-.063,-.073,-.083,-.093]:
  trace,row=run(lb,rb,model_path='modified_model.xml',foot_y=y)
  start=150 # 3 seconds at a 0.02 s recording interval
  row.update(case=label,pitch_at_gait_start_deg=float(np.rad2deg(trace[start,4])),hip_at_gait_start_rad=float(trace[start,3]),net_gait_displacement_m=float(np.linalg.norm(trace[-1,:2]-trace[start,:2])),final_yaw_change_deg=float(np.rad2deg(np.unwrap(trace[:,6])[-1]-trace[start,6])))
  rows.append(row)
  print(json.dumps(row),flush=True)
  (OUT/'modified_offset_results.json').write_text(json.dumps(rows,indent=2))
