# copy from https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/utils/vis_utils.py
# copy from nerfstudio and 2DGS
import torch
from matplotlib import cm
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import cv2

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    # apply color map to each pixel
    colored_image = colormap[image_long].squeeze(0).permute(2, 0, 1)
    return colored_image


def apply_depth_colormap(
    depth,
    accumulation = None,
    near_plane = 2.0,
    far_plane = 4.0,
    cmap="turbo",
):
    near_plane = float(torch.min(depth))
    far_plane = float(torch.max(depth))
    #near_plane = near_plane
    #far_plane = far_plane

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this
    colored_image = apply_colormap(depth, cmap=cmap)
    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image

def save_images(path_save, idx, rgb, depth):
    # save to disk
    torchvision.utils.save_image(rgb, path_save + f"{idx:05d}_rgb.png")
    #torchvision.utils.save_image(depth, path_save + f"{idx:05d}_depth.png")
    # save depth as uint16, val  = round(depth * 1000), save with cv2
    depth_mm = depth * 1000
    depth_np = depth_mm.permute(1, 2, 0).to("cpu").numpy().astype(np.uint16)
    cv2.imwrite(path_save + f"{idx:05d}_depth.png", depth_np)
    # also save colored depth
    depth_colored = apply_depth_colormap(depth)
    
    torchvision.utils.save_image(depth_colored, path_save + f"{idx:05d}_depth_colored.png")
  


def save_points(path_save, pts, colors=None, normals=None, BRG2RGB=False):
    """save points to point cloud using open3d"""
    assert len(pts) > 0
    if colors is not None:
        assert colors.shape[1] == 3
    assert pts.shape[1] == 3

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        # Open3D assumes the color values are of float type and in range [0, 1]
        if np.max(colors) > 1:
            colors = colors / np.max(colors)
        if BRG2RGB:
            colors = np.stack([colors[:, 2], colors[:, 1], colors[:, 0]], axis=-1)
        cloud.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(path_save, cloud)
    

def colormap(img, cmap='jet'):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    if img.shape[1:] != (H, W):
        img = torch.nn.functional.interpolate(img[None], (W, H), mode='bilinear', align_corners=False)[0]
    return img