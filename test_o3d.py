import open3d as o3d
import cv2
import numpy as np

img = cv2.imread('right_rgb.png', cv2.IMREAD_UNCHANGED)
width, height = img.shape[1], img.shape[0]
print(width, height)
img2 = o3d.geometry.Image(img)
depth = cv2.imread('right_depth.png', cv2.IMREAD_UNCHANGED)
depth_img = o3d.geometry.Image(depth)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        img2, depth_img)

fx, fy, cx, cy = -1082.5700408788937, 1082.5700408788937, 309.5, 239.5
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx, fy=fy,
        cx=cx, cy=cy
    )
)
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

o3d.visualization.draw_geometries([pcd])    # visualize the point cloud
