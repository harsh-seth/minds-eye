import os
import json

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


class Viewer:
    def __init__(self, on, of, fd):        #objectname, objectFile and folderdirectory
        self.index = 0
        self.objectName = on
        self.objectFile = of
        self.folderDirectory = fd
        self.vis = o3d.visualization.Visualizer()
        self.view = o3d.visualization.ViewControl()

        model_path = os.path.join(self.folderDirectory, self.objectFile)
        if os.path.exists(model_path):
            self.pcd = o3d.io.read_triangle_mesh(model_path)
            self.pcd.compute_vertex_normals()
            self.do = True
        else:
            self.do = False

        self.full_rotation = 2094.3951

        self.N = 2

        self.views_path = os.path.join(self.folderDirectory, 'images')
        if not os.path.exists(self.views_path):
            os.mkdir(self.views_path)

    def depthFullCapture(self):

        # self.numberOfTimes = times

        def captureDepth(vis):
            # print('Capturing')

            image_path = os.path.join(self.views_path, f"{self.objectName}_{self.index}.png")
            vis.capture_screen_image(image_path)
            vis.register_animation_callback(rotate)

        def rotate(vis):
            # print('Rotating')
            ctr = vis.get_view_control()
            if(self.index % self.N == 0):
                self.vis.reset_view_point(True)
                ctr.rotate(((self.full_rotation/self.N)*(self.index/self.N)), 0)
            else:
                ctr.rotate(0, self.full_rotation/self.N)
            # ctr.set_zoom(0.75)
            self.index += 1
            if not (self.index == self.N**2 + 1):
                vis.register_animation_callback(captureDepth)
            else:
                vis.register_animation_callback(None)
                vis.destroy_window()

                print(self.folderDirectory)


        self.vis.create_window(width = 200, height = 200)
        self.vis.add_geometry(self.pcd)
        self.vis.register_animation_callback(captureDepth)
        self.vis.run()

# def gridify(image_dir, obj_name):
#     images = []
#     for view in range(1,10):
#         image_path = os.path.join(image_dir, f"{obj_name}_{view}.png")
#         images.append(cv2.imread(image_path))

#     im_v1 = cv2.vconcat([images[0], images[1], images[2]])
#     im_v2 = cv2.vconcat([images[3], images[4], images[5]])
#     im_v3 = cv2.vconcat([images[6], images[7], images[8]])

#     im = cv2.hconcat([im_v2, im_v1, im_v3])

#     return im

def gridify(image_dir, obj_name):
    images = []
    for view in range(1,5):
        image_path = os.path.join(image_dir, f"{obj_name}_{view}.png")
        images.append(cv2.imread(image_path))

    im_v1 = cv2.vconcat([images[0], images[1]])
    im_v2 = cv2.vconcat([images[2], images[3]])

    im = cv2.hconcat([im_v2, im_v1])

    return im

data_dir = "data/"
data = json.load(open(os.path.join(data_dir, 'pix3d.json')))

for item in data:
    print(f"JSON: {item['model']}")
    folder_dir = os.path.join(data_dir, os.path.dirname(item['model']))
    object_name = os.path.basename(folder_dir)
    filename = "model.obj"

    viewer = Viewer(object_name, filename, folder_dir)
    
    if viewer.do:
        image_dir = os.path.join(folder_dir, 'images')
        # grid_path = os.path.join(image_dir, f"{object_name}_grid.png")
        grid_path = os.path.join('data/grids4/', f"{object_name}_4x4grid.png")

        if not os.path.exists(grid_path):
            viewer.depthFullCapture()
            grid = gridify(image_dir, object_name)
            
            cv2.imwrite(grid_path, grid)

    # print(grid_path)
    # break


