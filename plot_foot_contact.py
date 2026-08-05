import numpy as np
import pyvista as pv
import pickle
import argparse
import ctypes

# Fixes X11 threading issues for interactive windows on Linux/macOS
try:
    ctypes.CDLL("libX11.so").XInitThreads()
except OSError:
    pass

parser = argparse.ArgumentParser(description="Plot simulated contact trajectory data")
parser.add_argument("-fn", "--filename", type=str, default="results/contact_dict.pkl", help="Name of your simulation pickle file")
args = vars(parser.parse_args())

# ---------------------------------------------------------------------------
# Y-OFFSET MAP
# Shifts specific geoms along the Y-axis in the visualization so that all
# three foot parts sit side-by-side in the combined plot.
# This is purely for visual clarity and does not affect the underlying body-frame coordinates.
# ---------------------------------------------------------------------------
Y_OFFSETS: dict[str, float] = {
    "right_foot_1_col": 0.0,   # shift 5 cm in +Y  ← adjust as needed
    "right_foot_2_col": 0.0,   # shift 5 cm in -Y  ← adjust as needed
}

# 1. Load your simulated data dictionary
file_path = args['filename']
with open(file_path, 'rb') as f:
    con_pts_dict = pickle.load(f)

plotter = pv.Plotter()

# Unique colormaps to distinguish different geoms (e.g., heel vs toe)
colormaps = ['viridis', 'plasma', 'cividis', 'inferno']

# Keep a list of meshes we plotted so the callback can check against them
plotted_meshes = []

# --- INTERACTIVE CALLBACK ---
def my_callback(point_coords):
    """
    Callback triggered on Control + Left-Click.
    Receives the clicked XYZ coordinates as a numpy array.
    Coordinates printed to the terminal are always the original body-frame
    values, regardless of any visual Y-offset applied for display.
    """
    if point_coords is None:
        return

    best_mesh = None
    best_point_id = -1
    min_dist = float('inf')

    # Iterate through all plotted meshes to find which one is closest to the click.
    # mesh.points holds the (potentially shifted) display positions, which is
    # correct here because the click coordinate lives in display space too.
    for mesh in plotted_meshes:
        point_id = mesh.find_closest_point(point_coords)
        closest_pt = mesh.points[point_id]
        dist = np.linalg.norm(closest_pt - point_coords)

        if dist < min_dist:
            min_dist = dist
            best_mesh = mesh
            best_point_id = point_id

    if best_mesh is not None and min_dist < 0.05:  # 5 cm tolerance threshold
        geom_name   = best_mesh.field_data.get('geom_name', ['Unknown'])[0]
        time_val    = best_mesh["Time"][best_point_id]
        display_coords = best_mesh.points[best_point_id]

        # Recover the original body-frame coordinates by reversing the Y-offset
        y_shift = Y_OFFSETS.get(geom_name, 0.0)
        body_coords = display_coords.copy()
        body_coords[1] -= y_shift   # undo the visual shift

        # 1. Print to Terminal — always in body frame
        print("=" * 50)
        print(f"[CONTACT DETECTED] Geometry: {geom_name}")
        print(f"Point Index      : {best_point_id}")
        print(f"Simulation Time  : {time_val:.4f} s")
        print(f"Body-Frame XYZ   : [{body_coords[0]:.4f}, {body_coords[1]:.4f}, {body_coords[2]:.4f}]")
        if y_shift != 0.0:
            print(f"Display-Frame XYZ: [{display_coords[0]:.4f}, {display_coords[1]:.4f}, {display_coords[2]:.4f}]")
            print(f"(Y-offset of {y_shift:+.4f} m applied for visualization)")
        print("=" * 50 + "\n")

        # 2. Update on-screen label at the display position
        # Using name="current_contact_label" forces PyVista to replace the old
        # label instead of stacking duplicates.
        plotter.add_point_labels(
            [display_coords],
            [f"{geom_name}\nT: {time_val:.3f}s"],
            font_size=14,
            point_color="red",
            point_size=8,
            always_visible=True,
            name="current_contact_label"
        )

# 2. Iterate through tracked simulation geoms and extract coordinates
for i, (geom_name, data) in enumerate(con_pts_dict.items()):
    if geom_name == 'params':
        continue

    timed_coords = np.array(data['t_coords'])
    if timed_coords.shape[0] == 0:
        print(f"Skipping {geom_name}: No contact points recorded.")
        continue

    times          = timed_coords[:, 0]
    contact_points = timed_coords[:, 1:4]

    valid_mask = (
        ~np.isnan(contact_points).any(axis=1)
        & (np.linalg.norm(contact_points, axis=1) > 0.0001)
    )
    if not np.any(valid_mask):
        continue

    clean_points = contact_points[valid_mask]
    clean_times  = times[valid_mask]

    # Apply visual Y-offset if this geom is in the offset map.
    # We work on a copy so the original body-frame data is never mutated.
    y_shift = Y_OFFSETS.get(geom_name, 0.0)
    display_points = clean_points.copy()
    if y_shift != 0.0:
        display_points[:, 1] += y_shift
        print(f"[Y-offset] {geom_name}: {y_shift:+.4f} m applied for display")

    # Build the point cloud from the (possibly shifted) display positions
    point_cloud = pv.PolyData(display_points)
    point_cloud["Time"] = clean_times
    point_cloud.field_data['geom_name'] = [geom_name]

    # Track this mesh so the callback can search through it
    plotted_meshes.append(point_cloud)

    plotter.add_points(
        point_cloud,
        scalars="Time",
        cmap=colormaps[i % len(colormaps)],
        point_size=16,
        render_points_as_spheres=True,
        scalar_bar_args={'title': f'Time (s) - {geom_name}'}
    )

# 3. Configure layout and snap camera to the flat (X, Y) sole view
plotter.add_title("Simulated Foot Sole Contact Trajectories")
plotter.show_bounds(
    grid='back',
    minor_ticks=1,
    xtitle='Local X (m) [Length]',
    ytitle='Local Y (m) [Width]',
    ztitle='Local Z (m) [Height]'
)

plotter.camera_position = 'xy'

plotter.enable_point_picking(callback=my_callback, show_message=True)
plotter.show()