"""Sweep routing and physical displacement checks; no mesh generation required."""
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

import run_sweep
import sweep_config as config
import test_sim_sweep as sim

ROOT = Path(__file__).resolve().parents[1]


class FootOffsetSweepTest(unittest.TestCase):
    def test_offset_grid_reaches_trial_parameters_and_metadata(self):
        with patch.object(config, 'SWEEP_AXES', ('foot_x', 'foot_y')):
            jobs = run_sweep.build_jobs()
        expected_count = len(config.SWEEP_VALUES["foot_x"]) * len(config.SWEEP_VALUES["foot_y"])
        self.assertEqual(len(jobs), expected_count)
        self.assertEqual(len({tuple(j['axis_values'].values()) for j in jobs}), expected_count)
        for job in jobs:
            for axis in ('foot_x', 'foot_y'):
                self.assertEqual(job['trial_overrides'][axis], job['axis_values'][axis])
                self.assertEqual(job['geometry'][axis], job['axis_values'][axis])
            rows = sim.generate_parameter_samples(1, fixed_params=job['trial_overrides'], normal_distributions={})
            self.assertEqual(rows[0]['foot_x'], job['axis_values']['foot_x'])
            self.assertEqual(rows[0]['foot_y'], job['axis_values']['foot_y'])
            metadata = run_sweep.generic_axis_metadata(job)
            self.assertEqual(float(metadata['Sweep_foot_x_Value']), rows[0]['foot_x'])
        with patch.object(config, 'SWEEP_AXES', ('curve_x', 'foot_y', 'amp_deg')):
            mixed = run_sweep.build_jobs()[0]
        self.assertEqual(mixed['trial_overrides']['foot_x'], config.GEOMETRY_BASE['foot_x'])
        self.assertIn('amp_deg', mixed['trial_overrides'])

    def test_offsets_are_symmetric_and_com_follows_each_foot(self):
        for path in ('Bigfoot/scene.xml', 'modified_model.xml'):
            with self.subTest(model=path):
                ctx = sim.load_simulation(ROOT / path)
                m, d = ctx.model, ctx.data
                mujoco.mj_forward(m, d)
                torso_rotation = d.xmat[m.body('motor').id].reshape(3, 3).copy()
                geom_before = d.geom_xpos.copy()
                com_before = d.xipos.copy()
                sim.apply_foot_offsets(ctx, 0.009, -0.018)
                for side, expected in (('right', [0.009, -0.018, 0]), ('left', [-0.009, -0.018, 0])):
                    gid = m.geom(side + '_foot_1_col').id
                    bid = int(m.geom_bodyid[gid])
                    foot_delta = d.geom_xpos[gid] - geom_before[gid]
                    com_delta = d.xipos[bid] - com_before[bid]
                    np.testing.assert_allclose(torso_rotation.T @ foot_delta, expected, atol=1e-12)
                    np.testing.assert_allclose(com_delta, foot_delta, atol=1e-12)
                # Reusing a worker must not accumulate offsets from previous trials.
                sim.apply_foot_offsets(ctx, 0, 0)
                np.testing.assert_allclose(d.geom_xpos, geom_before, atol=1e-12)
                np.testing.assert_allclose(d.xipos, com_before, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
