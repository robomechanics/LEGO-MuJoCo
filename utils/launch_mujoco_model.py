import os
import mujoco
import mujoco.viewer

# curr_dir = os.path.dirname(os.path.abspath(__file__))
# file_path = "old/old_robot_files/duplo_hip_offset3/robot.urdf"
# file_path = "robots/duplo_ballfeet_mjcf/scene_motor_temp.xml"
# file_path = os.path.join(curr_dir, file_rel_path)
# file_path = "robots/zippy_mjcf/scene_motor_scaled_temp.xml"
# file_path = "old/old_robot_files/mugatu_nice_feet_fixed_urdf/robot.urdf"
# file_path = "old/old_robot_files/robotis_op3/scene.xml"

file_path = "zippy_hardstop/scene_tweaked.xml"

# Load your model
model = mujoco.MjModel.from_xml_path(file_path)
# Create a simulation data structure
data = mujoco.MjData(model)
# mujoco.mj_saveLastXML("old/old_robot_files/mugatu_nice_feet_fixed_urdf/robot.xml", model)

# Launch the viewer (GUI)
mujoco.viewer.launch(model, data)
