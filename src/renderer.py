import torch
from PIL import Image
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardFlatShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    look_at_view_transform,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.structures import Meshes

def render_normalmap(vertices, faces, image_size=128, dist=1.0, elev=30, azim=150):
    """Render world-space normal maps of meshes with a given color + resolution.
    Source: https://github.com/facebookresearch/pytorch3d/issues/865

    Parameters
    ----------
    vertices : torch.Tensor, shape=(B, N, 3)
        Array of vertex coordinates.
    faces : torch.Tensor, shape=(B, M, 3)
        Array of vertex indices for each face.
    image_size : int, optional
        Image resulution, by default 128
    dist : float, optional
        Camera distance from the origin, by default 1.0
    elev : int, optional
        Eelevation of the camera viewpoint in degrees, by default 30
    azim : int, optional
        Azimuth of the camera viewpoint in degrees (180 is from the front),
        by default 150
    Returns
    -------
    normal_maps : torch.Tensor, shape=(B, image_size, image_size)
        The normal maps from the given viewpoint.
    """
    mesh = Meshes(verts=vertices.float(), faces=faces.float())
    # create texture
    # Initialize a camera.
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size, bin_size=[0, None][0], cull_backfaces=True,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)
    
    
    def phong_normal_shading(meshes, fragments) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, ones, faces_normals
        )
        return pixel_normals

    normal_maps = phong_normal_shading(mesh, fragments)
    # Do we really need to permute?? We permute again while comparing hist and it seems not beneficial
    normal_maps = normal_maps.min(dim=-2)[0].permute((0, 3, 1, 2))
    return normal_maps

def render_view(mesh, device, image_size=128, dist=1.0, elev=30, azim=150):
    """Render a textured mesh to the given view
    Parameters
    ----------
    mesh : Mesh obj loaded as pytorch3d.structures.meshes.Meshes
    image_size : int, optional
        Image resulution, by default 128
    dist : float, optional
        Camera distance from the origin, by default 1.0
    elev : int, optional
        Eelevation of the camera viewpoint in degrees, by default 30
    azim : int, optional
        Azimuth of the camera viewpoint in degrees (180 is from the front),
        by default 150
    Returns
    -------
    rendered view : torch.Tensor, shape=(B, image_size, image_size, 3)
        Visualize the image as .
    """
    # If texture doesnot exist, create a plain texture to render
    if mesh.textures == None:
        temp_mesh = mesh
        verts_rgb = torch.ones_like(mesh.verts_list()[0])[None]
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        mesh = Meshes(
                verts=mesh.verts_list(),   
                faces=mesh.faces_list(), 
                textures=textures
                )
    # Set up camera    
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    
    # Set up light
    lights = PointLights(device=device, location=[[3.0, 3.0, 3.0]])
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights,
            blend_params=BlendParams(background_color=(0,0,0))
        )
    )
    
    return renderer(mesh)[:, :, :, :3]    