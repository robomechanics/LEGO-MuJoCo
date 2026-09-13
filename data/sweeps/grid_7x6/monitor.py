"""Monitor this run incrementally; preserve independent survival/stepping counts."""
import csv
import io
import json
import time
from pathlib import Path

root = Path(__file__).resolve().parent
position = 0
pending = ''
header = None
counts = {'trials': 0, 'fell': 0, 'gait_passes': 0, 'survived_with_steps_and_clearance': 0}
examples = []
started = time.time()
while True:
    path = root / 'sweep_results.csv'
    if path.exists():
        with path.open() as handle:
            handle.seek(position)
            chunk = handle.read()
            position = handle.tell()
        pending += chunk
        complete, _, pending = pending.rpartition('\n') if '\n' in pending else ('', '', pending)
        for values in csv.reader(io.StringIO(complete)):
            if header is None:
                header = values
                continue
            row = dict(zip(header, values))
            counts['trials'] += 1
            counts['fell'] += row['Fell'] == 'True'
            counts['gait_passes'] += row['Gait_Quality_Pass'] == 'True'
            if row['Fell'] == 'False' and int(row['Alternating_Steps']) >= 1 and float(row['Min_Swing_Clearance']) >= 0.02:
                counts['survived_with_steps_and_clearance'] += 1
                if len(examples) < 10:
                    examples.append({k: row[k] for k in ('Curve_X', 'Curve_Y', 'Foot_X', 'Foot_Y', 'Amplitude_Deg', 'Start_Freq_Mult', 'Distance_Traversed', 'Alternating_Steps')})
    summary = {**counts, 'total_grid_points': 117649, 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
               'no_surviving_stepping_trials_so_far': counts['survived_with_steps_and_clearance'] == 0,
               'note': 'Existing gait pass/forward score uses X as forward; surviving stepping is a separate diagnostic, not verified forward walking.',
               'examples': examples}
    with (root / 'monitor.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k: summary[k] for k in ('updated_at', 'trials', 'fell', 'gait_passes', 'survived_with_steps_and_clearance')}), flush=True)
    with (root / 'run.log').open('rb') as log:
        log.seek(max(0, log.seek(0, 2) - 6000))
        tail = log.read().decode(errors='replace')
    if 'Done. Combined results' in tail or 'Traceback (most recent call last)' in tail:
        break
    time.sleep(30)
