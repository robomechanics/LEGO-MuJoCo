import csv
import json
from pathlib import Path
import tempfile
import unittest

from record_target_walk_xy import select_trials, saved_trial


class SavedWalkTest(unittest.TestCase):
    def test_select_only_passes_rank_and_ignore_partial_tail(self):
        fields = ['Point_Index', 'Fell', 'Gait_Quality_Pass', 'Distance_Traversed', 'CoT', 'Walk_Score']
        rows = [(1, False, True, 2, 4, 1), (2, False, True, 3, 5, 2),
                (3, False, False, 100, 1, 99), (4, True, True, 200, 0.5, 99)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.csv'
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            with path.open('a') as handle:
                handle.write('5,False,True,999')
            self.assertEqual([r['Point_Index'] for r in select_trials(path, 5, 'distance')], ['2', '1'])
            self.assertEqual(select_trials(path, 1, 'cot')[0]['Point_Index'], '1')

    def test_saved_parameters_are_required_and_not_rounded(self):
        params = dict(foot_x=-0.026666667, foot_y=0.008333333, torque_limit=23.7,
                      Kp=29.1, Kd=8.2, start_amp_mult=1.31, start_freq_mult=1.966666667,
                      ramp_time=0, amp_deg=33.333333333, freq_hz=0.524)
        row = dict(Curve_X='0.7488', Curve_Y='0.728', Box_X='0.667', Box_Y='0.24',
                   Replay_Params_JSON=json.dumps(params))
        geometry, replay = saved_trial(row)
        self.assertEqual(replay, params)
        self.assertEqual(geometry['curve_x'], 0.7488)
        del params['Kp']
        row['Replay_Params_JSON'] = json.dumps(params)
        with self.assertRaises(ValueError):
            saved_trial(row)
