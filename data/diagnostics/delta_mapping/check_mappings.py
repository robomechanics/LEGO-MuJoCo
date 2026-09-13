"""Headless delta ablations using the actual original closed-loop AST.
Run from the repository root: python3 data/diagnostics/delta_mapping/check_mappings.py
No production code or model edits; artifacts are written alongside this script.
"""
import ast,contextlib,csv,io,itertools,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import settle_utils
OUT=Path(__file__).resolve().parent
SOURCE=(OUT/'original_reference.py').read_text()
TREE=ast.parse(SOURCE)
SPLIT=next(i for i,n in enumerate(TREE.body) if isinstance(n,ast.With) and 'launch_passive' in ast.unparse(n.items[0].context_expr))
LOOP=next(n for n in TREE.body[SPLIT].body if isinstance(n,ast.While))
BODY=[n for n in LOOP.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) in ('viewer.sync','time.sleep'))]
STEP=compile(ast.fix_missing_locations(ast.Module(body=BODY,type_ignores=[])),'closedloop_step','exec')

def run(left_body,right_body='[FOOT_X, FOOT_Y, FOOT_Z]',left_geom='[FOOT_Z, -FOOT_Y, FOOT_X]',settle='original',duration=20,model_path='Bigfoot/scene.xml',foot_y=-0.023):
    tree=ast.parse(SOURCE)
    replacements={'delta_left_body':left_body,'delta_right_body':right_body,'delta_left_geom':left_geom}
    for i,node in enumerate(tree.body[:SPLIT]):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
            name=node.targets[0].id
            tree.body[i]=ast.parse(f'{name} = np.array({replacements[name]}, dtype=float)').body[0]
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FOOT_Y' for t in node.targets):
            tree.body[i]=ast.parse(f'FOOT_Y = {foot_y!r}').body[0]
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model' for t in node.targets):
            tree.body[i]=ast.parse(f'model = mujoco.MjModel.from_xml_path({model_path!r})').body[0]
        if settle=='current' and isinstance(node,ast.FunctionDef) and node.name=='startup_settle_orientation':
            tree.body[i]=ast.parse('''def startup_settle_orientation():
    return new_settle(model,data,hip_qpos_adr,hip_qvel_adr,TORQUE_LIMIT,STARTUP_SETTLE_S,STARTUP_SETTLE_AVG_S,KP,KD,print_fn=None)
''').body[0]
    ns={'__name__':'delta_diagnostic','new_settle':settle_utils.startup_settle_orientation}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.fix_missing_locations(ast.Module(body=tree.body[:SPLIT],type_ignores=[])),'closedloop_setup','exec'),ns)
    # The current closed-loop loop preserves the holding command in its delay buffer.
    if settle=='current':ns['cmd_buffer'][:]=[float(ns['data'].ctrl[0])]*ns['CMD_DELAY_STEPS']
    data,model=ns['data'],ns['model'];samples=[];first_drop=None;minheight=10
    for i in range(round(duration/model.opt.timestep)):
        if i%10==0:
            samples.append(np.r_[data.qpos[:3],data.qpos[ns['hip_qpos_adr']],settle_utils.quat_to_rpy(data.qpos[3:7])[0]])
        exec(STEP,ns)
        minheight=min(minheight,float(data.qpos[2]))
        if first_drop is None and data.qpos[2]<.5:first_drop=float(data.time)
    return np.array(samples),{'model':model_path,'foot_y':foot_y,'left_body':left_body,'right_body':right_body,'left_geom':left_geom,'settle':settle,'upright':first_drop is None,'first_drop_s':first_drop,'minimum_base_height_m':minheight,'distance_m':float(np.linalg.norm(data.qpos[:2])),'final_height_m':float(data.qpos[2])}

if __name__=='__main__':
    reference,baseline=run('[-FOOT_X, FOOT_Y, FOOT_Z]')
    np.save(OUT/'reference_trajectory.npy',reference)
    def evaluate(lb,rb,lg,settle='original'):
        try:
            trace,row=run(lb,rb,lg,settle)
            row['position_rmse_m']=float(np.sqrt(np.mean(np.sum((trace[:,:3]-reference[:,:3])**2,axis=1))))
            row['hip_rmse_rad']=float(np.sqrt(np.mean((trace[:,3]-reference[:,3])**2)))
            yaw=(trace[:,6]-reference[:,6]+np.pi)%(2*np.pi)-np.pi
            row['yaw_rmse_deg']=float(np.rad2deg(np.sqrt(np.mean(yaw**2))))
            row['match_error']=row['position_rmse_m']+row['hip_rmse_rad']+np.deg2rad(row['yaw_rmse_deg'])
        except RuntimeError as error:row=dict(left_body=lb,right_body=rb,left_geom=lg,settle=settle,error=str(error),upright=False,match_error=float('inf'))
        return row
    rows=[]
    for p in itertools.permutations(['FOOT_X','FOOT_Y','FOOT_Z']):
        for sx,sy in itertools.product([1,-1],repeat=2):
            lb='['+', '.join(('-' if {'FOOT_X':sx,'FOOT_Y':sy,'FOOT_Z':1}[v]<0 else '')+v for v in p)+']'
            for rb in ['[FOOT_X, FOOT_Y, FOOT_Z]','[FOOT_X, -FOOT_Y, FOOT_Z]']:
                for lg in ['[FOOT_Z, -FOOT_Y, FOOT_X]','[FOOT_Z, FOOT_Y, FOOT_X]']:
                    rows.append(evaluate(lb,rb,lg))
        print(f'{len(rows)}/96 checked; upright={sum(r["upright"] for r in rows)}',flush=True)
    rows.sort(key=lambda r:(not r['upright'],r['match_error']))
    print('BEST ORIGINAL STARTUP',json.dumps(rows[:8]),flush=True)
    # Retest all surviving arrangements under the current initializer as a separate factor.
    chosen=[row for row in rows if row["upright"]]
    for row in chosen:rows.append(evaluate(row['left_body'],row['right_body'],row['left_geom'],'current'))
    (OUT/'results.json').write_text(json.dumps({'baseline':baseline,'trials':rows},indent=2))
    print('CURRENT STARTUP',json.dumps(rows[-len(chosen):]),flush=True)
