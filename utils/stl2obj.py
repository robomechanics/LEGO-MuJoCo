import numpy as np
import trimesh
import os

trimesh.util.attach_to_log()

# Load the mesh from file
stl_dir = "robots/zippy_mjcf/"

# loop through all the files in the directory
for file in os.listdir(stl_dir):
    if file.endswith(".stl"):
        mesh = trimesh.load_mesh(f"{stl_dir}/{file}")
        print(f"Mesh: {file}")
        print(f"Vertices: {mesh.vertices.shape}")
        print(f"Faces: {mesh.faces.shape}")

        # Save the mesh to an obj file
        mesh.export(f"{stl_dir}/{file.replace('.stl', '.obj')}")
        print(f"Saved {file.replace('.stl', '.obj')}")
        print("\n")