import os
offscreen = True

import torch, numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from pyquaternion import Quaternion
from mpl_toolkits.axes_grid1 import ImageGrid
import os
os.environ['ETS_TOOLKIT'] = 'null'  # GUI toolkit kullanma
os.environ['QT_API'] = 'pyqt5'  
os.environ["VTK_DEFAULT_OPENGL_WINDOW"] = "vtkOSOpenGLRenderWindow"

from mayavi import mlab
mlab.options.offscreen = True
from xvfbwrapper import Xvfb

import torch.distributed as dist

def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

SIGMOID_MAX = 9.21024
LOGIT_MAX = 0.9999

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -9.21, 9.21)
    return torch.sigmoid(tensor)

def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def save_occ(save_dir, gaussian, name, sem=False, cap=2, dataset='nusc'):
    if not is_main_process():
        return
    
    # 1. PREPARE DATA (Your original logic)
    if dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
        vmin, vmax = 0, 16
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 19
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 18

    voxels = gaussian[0].cpu().to(torch.int)
    voxels[0, 0, 0] = 1
    voxels[-1, -1, -1] = 1
    if not sem:
        voxels[..., (-cap):] = 0
        for z in range(voxels.shape[-1] - cap):
            mask = (voxels > 0)[..., z]
            voxels[..., z][mask] = z + 1 
    
    # Compute coordinates
    grid_coords = get_grid_coords(
        voxels.shape, voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    fov_grid_coords = grid_coords

    if not sem:
        fov_voxels = fov_grid_coords[
            (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 100)
        ]
    else:
        if dataset == 'nusc':
            fov_voxels = fov_grid_coords[(fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 17)]
        elif dataset == 'kitti360':
            fov_voxels = fov_grid_coords[(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 19)]
        else:
            fov_voxels = fov_grid_coords[(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)]


    # 2. CREATE FIGURE (Offscreen)
    # We create the figure explicitly to ensure settings apply
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))

    # 3. DRAW VOXELS
    voxel_size_val = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0], -fov_voxels[:, 1], fov_voxels[:, 2], fov_voxels[:, 3],
            colormap="jet", scale_factor=1.0 * voxel_size_val, mode="cube", opacity=1.0
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0], -fov_voxels[:, 1], fov_voxels[:, 2], fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size_val, mode="cube", opacity=1.0, vmin=vmin, vmax=vmax
        )

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    
    # Apply Colors (Your original logic)
    if sem:
        if dataset == 'nusc':
            colors = np.array([[0,0,0,255], [255,120,50,255], [255,192,203,255], [255,255,0,255], 
                               [0,150,245,255], [0,255,255,255], [255,127,0,255], [255,0,0,255], 
                               [255,240,150,255], [135,60,0,255], [160,32,240,255], [255,0,255,255], 
                               [139,137,137,255], [75,0,75,255], [150,240,80,255], [230,230,250,255], 
                               [0,175,0,255]]).astype(np.uint8)
        elif dataset == 'kitti360':
            colors = (get_kitti360_colormap()[1:, :] * 255).astype(np.uint8)
        else:
            colors = (get_kitti_colormap()[1:, :] * 255).astype(np.uint8)
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    # 4. SET CAMERA & RENDER ONCE
    # I replaced your loop with a single camera setup (using the final position from your loop)
    scene = figure.scene
    scene.camera.position = [-107.155, -0.008, 92.167]
    scene.camera.focal_point = [0.008, -0.008, 1.4]
    scene.camera.view_angle = 30.0
    scene.camera.view_up = [0.646, 0.0, 0.763]
    scene.camera.clipping_range = [78.84, 218.29]
    scene.camera.compute_view_plane_normal()
    
    scene.render() # Render once to memory

    # 5. SAVE AND CLOSE
    filepath = os.path.join(save_dir, f'{name}.png')
    mlab.savefig(filepath)
    mlab.close(all=True) # Clean up memory
    



def get_nuscenes_colormap():
    colors = np.array(
        [
            [  0,   0,   0, 255],       # others
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            # [175,   0,  75, 255],       # other_flat           dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
            # [  0, 255, 127, 255],       # ego car              dark cyan
            # [255,  99,  71, 255],       # ego car
            # [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.float32) / 255.
    return colors

def save_gaussian(save_dir, gaussian, name, scalar=1.5, ignore_opa=False, filter_zsize=False):
    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    torch.save(gaussian, os.path.join(save_dir, f'{name}_attr.pth'))

    means = gaussian.means[0].detach().cpu().numpy() # g, 3
    scales = gaussian.scales[0].detach().cpu().numpy() # g, 3
    rotations = gaussian.rotations[0].detach().cpu().numpy() # g, 4
    opas = gaussian.opacities[0]
    if opas.numel() == 0:
        opas = torch.ones_like(gaussian.means[0][..., :1])
    opas = opas.squeeze().detach().cpu().numpy() # g
    sems = gaussian.semantics[0].detach().cpu().numpy() # g, 18
    pred = np.argmax(sems, axis=-1)

    if ignore_opa:
        opas[:] = 1.
        mask = (pred != empty_label)
    else:
        mask = (pred != empty_label) & (opas > 0.75)

    if filter_zsize:
        zdist, zbins = np.histogram(means[:, 2], bins=100)
        zidx = np.argsort(zdist)[::-1]
        for idx in zidx[:10]:
            binl = zbins[idx]
            binr = zbins[idx + 1]
            zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
            mask = mask & zmsk
        
        z_small_mask = scales[:, 2] > 0.1
        mask = z_small_mask & mask


    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opas = opas[mask]
    pred = pred[mask]

    # number of ellipsoids 
    ellipNumber = means.shape[0]

    #set colour map so each ellipsoid as a unique colour
    norm = colors.Normalize(vmin=-1.0, vmax=5.4)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(9, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=46, azim=-180)

    # compute each and plot each ellipsoid iteratively
    border = np.array([
        [-50.0, -50.0, 0.0],
        [-50.0, 50.0, 0.0],
        [50.0, -50.0, 0.0],
        [50.0, 50.0, 0.0],
    ])
    ax.plot_surface(border[:, 0:1], border[:, 1:2], border[:, 2:], 
        rstride=1, cstride=1, color=[0, 0, 0, 1], linewidth=0, alpha=0., shade=True)

    for indx in range(ellipNumber):
        
        center = means[indx]
        radii = scales[indx] * scalar
        rot_matrix = rotations[indx]
        rot_matrix = Quaternion(rot_matrix).rotation_matrix.T

        u = np.linspace(0.0, 2.0 * np.pi, 10)
        v = np.linspace(0.0, np.pi, 10)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

        xyz = np.stack([x, y, z], axis=-1) # phi, theta, 3
        xyz = rot_matrix[None, None, ...] @ xyz[..., None]
        xyz = np.squeeze(xyz, axis=-1)

        xyz = xyz + center[None, None, ...]

        ax.plot_surface(
            xyz[..., 1], -xyz[..., 0], xyz[..., 2], 
            rstride=1, cstride=1, color=sem_cmap[pred[indx]], linewidth=0, alpha=opas[indx], shade=True)

    plt.axis("equal")
    ax.grid(False)
    ax.set_axis_off()    

    filepath = os.path.join(save_dir, f'{name}.png')
    plt.savefig(filepath)

def save_gaussian_topdown(save_dir, anchor_init, gaussian, name):
    init_means = safe_sigmoid(anchor_init[:, :2]) * 100 - 50
    means = [init_means] + [g.means[0, :, :2] for g in gaussian]

    fig = plt.figure(figsize=(24., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, 5),  # creates 2x2 grid of Axes
                    axes_pad=0.,  # pad between Axes in inch.
                    share_all=True
                    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    for ax, im in zip(grid, means):
        im = im.cpu().detach().numpy()
        ax.scatter(im[:, 0], im[:, 1], s=0.1, marker='o')
    plt.savefig(os.path.join(save_dir, f"{name}.jpg"))
