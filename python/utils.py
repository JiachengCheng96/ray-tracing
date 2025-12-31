import open3d as o3d
import numpy as np
import requests
import io
import trimesh
import matplotlib.pyplot as plt

def load_bunny_mesh(target_triangles=2000):
    """
    Downloads Stanford Bunny, converts to Open3D, and simplifies it for performance.
    """
    print("1. Loading Stanford Bunny...")
    url = "https://raw.githubusercontent.com/mikedh/trimesh/master/models/bunny.ply"
    try:
        data = requests.get(url).content
        mesh_tri = trimesh.load(io.BytesIO(data), file_type='ply')
    except Exception as e:
        print(f"   Download failed ({e}), creating sphere instead.")
        mesh_tri = trimesh.creation.icosphere(subdivisions=3)

    # Convert to Open3D
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mesh_tri.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(mesh_tri.faces)
    mesh.compute_vertex_normals()
    
    # Simplify mesh
    print(f"   Original triangles: {len(mesh.triangles)}")
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print(f"   Simplified triangles: {len(mesh.triangles)}")
    
    # Center and scale
    center = mesh.get_center()
    mesh.translate(-center)
    scale = 1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound())
    mesh.scale(scale, center=[0,0,0])
    
    return mesh

def create_pixel_grid(center, normal, up, width, height, resolution_x, resolution_y, verbose=True):
    """
    Creates a grid of 3D points representing pixels on a virtual screen defined by position and orientation.
    """
    if verbose:
        print(f"2. Creating Pixel Grid ({resolution_x}x{resolution_y})...")
    
    # Normalize vectors
    normal = normal / np.linalg.norm(normal)
    up = up / np.linalg.norm(up)
    
    # Calculate Right vector
    right = np.cross(normal, up)
    right /= np.linalg.norm(right)
    
    # Recompute Up to ensure orthogonality
    true_up = np.cross(right, normal)
    
    half_width = width / 2.0
    half_height = height / 2.0
    
    # Generate grid
    xs = np.linspace(-half_width, half_width, resolution_x)
    ys = np.linspace(half_height, -half_height, resolution_y) # Top to bottom
    xx, yy = np.meshgrid(xs, ys)
    
    # Grid points in World Space
    grid_points = center + (xx[..., np.newaxis] * right) + (yy[..., np.newaxis] * true_up)
    
    return grid_points.reshape(-1, 3)

def plot_heatmap(heatmap, save_path=None, title_str='Ray Hits per Pixel', block=True):
    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap='jet', interpolation='nearest')
    plt.colorbar(label='Hit Count')
    plt.title(title_str)
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    # plt.pause(0.1)
    # plt.show(block=False)
    plt.show(block=block)

def create_ground_grid(size=10, step=1.0, y_level=-0.5):
    """Creates a grid of lines to visualize the ground plane."""
    lines = []
    points = []
    
    # X-lines
    for z in np.arange(-size, size + step, step):
        p1 = [-size, y_level, z]
        p2 = [size, y_level, z]
        points.append(p1)
        points.append(p2)
        lines.append([len(points)-2, len(points)-1])
        
    # Z-lines
    for x in np.arange(-size, size + step, step):
        p1 = [x, y_level, -size]
        p2 = [x, y_level, size]
        points.append(p1)
        points.append(p2)
        lines.append([len(points)-2, len(points)-1])
        
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0.7, 0.7, 0.7]) # Light grey
    return line_set

def create_sensor_wireframe(center, normal, up, width, height):
    """Creates a wireframe rectangle for the sensor."""
    normal = normal / np.linalg.norm(normal)
    up = up / np.linalg.norm(up)
    right = np.cross(normal, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, normal)
    
    w2 = width / 2.0
    h2 = height / 2.0
    
    c1 = center - w2 * right + h2 * true_up
    c2 = center + w2 * right + h2 * true_up
    c3 = center + w2 * right - h2 * true_up
    c4 = center - w2 * right - h2 * true_up
    
    points = [c1, c2, c3, c4]
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 0, 0]) # Black outline
    return line_set
