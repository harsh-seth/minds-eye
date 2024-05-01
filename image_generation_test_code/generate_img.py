import open3d as o3d
import os
import matplotlib.pyplot as plt
import numpy as np

mesh = o3d.io.read_triangle_mesh('../pix3d_full/model/bed/IKEA_BEDDINGE/model.obj') 
mesh.compute_vertex_normals()

# o3d.visualization.draw_plotly([mesh])
# o3d.visualization.draw_geometries([mesh])

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
view_control = vis.get_view_control()
view_control.change_field_of_view(step=90)
vis.run()
vis.update_renderer()
vis.capture_screen_image('test-12.png')
############
# view_control.rotate(x=90,y=0)
# vis.run()
# vis.capture_screen_image('test-2.png')
# ############
# view_control.rotate(x=90,y=90)
# vis.run()
# vis.capture_screen_image('test-3.png')
# ############
# view_control.rotate(x=180,y=180)
# vis.run()
# vis.capture_screen_image('test-4.png')
############
view_control.rotate(x=-180,y=-180)
vis.update_geometry(mesh)
vis.poll_events()
vis.update_renderer()
vis.run()
vis.capture_screen_image('test-52.png')
vis.destroy_window()
