import numpy as np
import ctypes
ctypes.CDLL("libX11.so").XInitThreads()
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from stl import mesh
import pickle

# Load your contact points dictionary from pickle
with open('contact_dict_A17.pkl', 'rb') as f:
    con_pts_dict = pickle.load(f)

mesh_dir = con_pts_dict['params']['mesh_dir']
stretch_factors = con_pts_dict['params']['stretch_factors']

# Create a PyVista plotter
plotter = pv.Plotter()

for key, v in con_pts_dict.items():
    if key != 'params':
        # Load the mesh from file (assumes filename stored without extension)
        mesh_name = v['mesh']
        stl_mesh = mesh.Mesh.from_file(f"{mesh_dir}/{mesh_name}.stl")

        # Get contact points and transformation parameters
        timed_coords = np.array(v['t_coords'])
        contact_points = timed_coords[:, 1:4]
        times = timed_coords[:, 0]

        rot = R.from_quat(v['quat'], scalar_first=True)
        trans = np.array(v['pos'])
        offset = np.array(v['mesh_offset'])

        # Apply rotation+translation to the contact points
        contact_points_transformed = rot.apply(contact_points) + trans + offset
        # Flatten the mesh vertices and apply the rotation+translation
        mesh_points = stl_mesh.vectors.reshape(-1, 3)
        
        # Apply element-wise multiplication (scaling) to the mesh points:
        mesh_points_stretched = mesh_points * stretch_factors
        mesh_points_transformed = rot.apply(mesh_points_stretched) + trans + offset

        # If you need to update the PolyData, rebuild the connectivity as before:
        num_triangles = mesh_points_stretched.shape[0] // 3
        triangle_indices = np.arange(mesh_points_stretched.shape[0]).reshape(num_triangles, 3)
        faces = np.hstack([np.full((num_triangles, 1), 3), triangle_indices]).flatten()

        # Create new PolyData with stretched points:
        pv_mesh = pv.PolyData(mesh_points_stretched, faces)

        # Build the connectivity (faces) array for triangles.
        # Each triangle face is stored as [3, idx0, idx1, idx2]
        num_triangles = mesh_points_transformed.shape[0] // 3
        triangle_indices = np.arange(mesh_points_transformed.shape[0]).reshape(num_triangles, 3)
        faces = np.hstack([np.full((num_triangles, 1), 3), triangle_indices]).flatten()

        # Create a PyVista PolyData mesh from the points and connectivity
        pv_mesh = pv.PolyData(mesh_points_transformed, faces)

        # Add the mesh to the plotter (using a cyan color and some transparency)
        plotter.add_mesh(pv_mesh, 
                        color='pink', 
                        opacity=0.5, 
                        show_edges=True, 
                        edge_color='black', 
                        line_width=2,
                        lighting=False)

        # ----- Add contact points -----
        plotter.add_points(contact_points_transformed,
                        scalars=times,
                        cmap='viridis',
                        point_size=5,
                        render_points_as_spheres=True)

# Display the interactive 3D plot
plotter.show_bounds( 
    grid='back', 
    minor_ticks=1,
    n_xlabels=20,
    xtitle='X', 
    ytitle='Y', 
    ztitle='Z'
    )
plotter.show_axes()
plotter.show()
