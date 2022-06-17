# -*- coding: utf-8 -*-
# @Time         : 2022/3/13 17:36
# @Author       : juju
# @File         : render
# @Description  : ***

import numpy as np
import math
import trimesh
import pyrender
from pyrender.constants import RenderFlags
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"


class WeakPerspectiveCamera(pyrender.Camera):

    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:

    def __init__(self,
                 smpl_faces,
                 resolution=(224, 224),
                 orig_img=False,
                 wireframe=False):
        self.resolution = resolution

        self.faces = smpl_faces
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0)

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                    ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self,
               img,
               verts,
               cam,
               angle=None,
               axis=None,
               mesh_filename=None,
               color=[1.0, 1.0, 0.9]):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180),
                                                     [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(
                math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(scale=[sx, sy],
                                       translation=[tx, ty],
                                       zfar=1000.)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0))

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:, 0] * (1. / (img_width / h))
    sy = cam[:, 0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:, 1]
    ty = ((cy - hh) / hh / sy) + cam[:, 2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def revert_to_bbox(center, scale, height=200, scale_factor=1):
    h = scale * height / scale_factor
    cx = center[0]
    cy = center[1]
    bbox = [cx, cy, h]
    return bbox


def parse_cam(cam):
    x = (2 * 5000 / cam[:, 2] - 1e-9) / 224
    y = cam[:, 0]
    z = cam[:, 1]
    return np.stack([x, y, z], axis=1)
