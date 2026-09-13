"""Headless regression checks: python3 -m unittest discover -s tests -p test_startup_settle.py."""
from pathlib import Path
import unittest

import mujoco
import numpy as np

from control_waveform import DEFAULT_KP, DEFAULT_KD, DEFAULT_TORQUE_LIMIT
from settle_utils import quat_to_rpy, startup_settle_orientation

ROOT = Path(__file__).resolve().parents[1]


def load_trial(path):
    model = mujoco.MjModel.from_xml_path(str(ROOT / path))
    data = mujoco.MjData(model)
    # Match the closed-loop trial's geometry and inertial offsets.
    for side, offset in (("right", [0.004, -0.023, 0]), ("left", [0, 0.023, 0.004])):
        for suffix in ("", "_col"):
            model.geom_pos[model.geom(f"{side}_foot_1{suffix}").id] += offset
    model.body_ipos[model.body("motor").id] += [0.004, -0.023, 0]
    model.body_ipos[model.body("simplified_motor___arm_rod").id] += [0, 0.023, 0.004]
    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    return model, data


class StartupSettleTest(unittest.TestCase):
    def test_supported_stance_remains_still_after_release(self):
        for path in ("Bigfoot/scene.xml", "modified_model.xml"):
            with self.subTest(model=path):
                model, data = load_trial(path)
                qi = int(model.joint("hip").qposadr[0])
                vi = int(model.joint("hip").dofadr[0])
                startup_settle_orientation(model, data, qi, vi, DEFAULT_TORQUE_LIMIT, 8, 2, DEFAULT_KP, DEFAULT_KD, print_fn=None)
                self.assertEqual(data.time, 0)
                np.testing.assert_array_equal(data.qfrc_applied, 0)
                self.assertLess(data.qpos[2], 1.1)  # No second drop from 1.2 m.
                delayed = float(data.ctrl[0])
                angles = [quat_to_rpy(data.qpos[3:7])[0]]
                for _ in range(round(5 / model.opt.timestep)):
                    command = float(np.clip(-DEFAULT_KP * data.qpos[qi] - DEFAULT_KD * data.qvel[vi], -DEFAULT_TORQUE_LIMIT, DEFAULT_TORQUE_LIMIT))
                    data.ctrl[0], delayed = delayed, command
                    mujoco.mj_step(model, data)
                    angles.append(quat_to_rpy(data.qpos[3:7])[0])
                span = np.rad2deg(np.ptp(np.unwrap(angles, axis=0), axis=0))
                self.assertLess(span[2], 0.02)
                self.assertLess(max(span[:2]), 0.05)

    def test_original_spawn_contacts_are_simultaneous(self):
        for path in ("Bigfoot/scene.xml", "modified_model.xml"):
            with self.subTest(model=path):
                model, data = load_trial(path)
                qi = int(model.joint("hip").qposadr[0])
                vi = int(model.joint("hip").dofadr[0])
                first = {}
                for step in range(round(0.5 / model.opt.timestep)):
                    data.ctrl[0] = np.clip(-45 * data.qpos[qi] - 7 * data.qvel[vi], -25, 25)
                    mujoco.mj_step(model, data)
                    for contact in data.contact:
                        names = [model.geom(int(g)).name for g in (contact.geom1, contact.geom2)]
                        if "floor" in names:
                            for name in names:
                                if "foot" in name:
                                    first.setdefault(name, step)
                    if len(first) == 2:
                        break
                self.assertEqual(len(first), 2)
                self.assertEqual(first["left_foot_1_col"], first["right_foot_1_col"])


if __name__ == "__main__":
    unittest.main()
