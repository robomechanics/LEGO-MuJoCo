import pymeshlab

ms = pymeshlab.MeshSet()
ms.load_new_mesh('Bigfoot/assets/footstl_scaled_v4_mirrored.stl')
ms.generate_convex_hull()          # older versions: ms.convex_hull()
ms.save_current_mesh('Bigfoot/assets/right_foot_1.stl')

ms.load_new_mesh('Bigfoot/assets/footstl_scaled_v4.stl')
ms.generate_convex_hull()          # older versions: ms.convex_hull()
ms.save_current_mesh('Bigfoot/assets/left_foot_1.stl')