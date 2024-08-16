import torch

import trimesh
import glob
import os
import numpy as np
import copy
import roma


def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces + 1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))


class SvhHandLayer(torch.nn.Module):
    def __init__(self, to_mano_frame=False, show_mesh=False):
        # The forward kinematics equations implemented here are from
        super().__init__()
        self.show_mesh = show_mesh
        zero_tensor = torch.tensor(0.0, dtype=torch.float32)
        one_tensor = torch.tensor(1.0, dtype=torch.float32)
        self.J1_a = zero_tensor
        self.J1_alpha = zero_tensor
        self.J1_d = zero_tensor
        self.J1_theta = zero_tensor

        self.J2_a = one_tensor * -0.04596
        self.J2_alpha = zero_tensor
        self.J2_d = zero_tensor
        self.J2_theta = np.pi * one_tensor

        self.J3_a = one_tensor * 0.0485
        self.J3_alpha = zero_tensor
        self.J3_d = zero_tensor
        self.J3_theta = zero_tensor

        self.J4_a = one_tensor * 0.03
        self.J4_alpha = zero_tensor
        self.J4_d = zero_tensor
        self.J4_theta = zero_tensor

        self.J5_a = zero_tensor
        self.J5_alpha = zero_tensor
        self.J5_d = zero_tensor
        self.J5_theta = zero_tensor

        self.J6a_a = zero_tensor
        self.J6a_alpha = zero_tensor
        self.J6a_d = zero_tensor
        self.J6a_theta = zero_tensor

        self.J6b_a = zero_tensor
        self.J6b_alpha = one_tensor * -1.571
        self.J6b_d = zero_tensor
        self.J6b_theta = zero_tensor

        self.J7a_a = zero_tensor
        self.J7a_alpha = zero_tensor
        self.J7a_d = zero_tensor
        self.J7a_theta = zero_tensor

        self.J7b_a = zero_tensor
        self.J7b_alpha = one_tensor * -1.571
        self.J7b_d = zero_tensor
        self.J7b_theta = zero_tensor

        self.J8a_a = zero_tensor
        self.J8a_alpha = zero_tensor
        self.J8a_d = zero_tensor
        self.J8a_theta = zero_tensor

        self.J8b_a = zero_tensor
        self.J8b_alpha = one_tensor * 1.571
        self.J8b_d = zero_tensor
        self.J8b_theta = zero_tensor

        self.J9a_a = zero_tensor
        self.J9a_alpha = zero_tensor
        self.J9a_d = zero_tensor
        self.J9a_theta = zero_tensor

        self.J9b_a = zero_tensor
        self.J9b_alpha = one_tensor * 1.571
        self.J9b_d = zero_tensor
        self.J9b_theta = zero_tensor

        self.J10_a = 0.04804 * one_tensor
        self.J10_alpha = zero_tensor
        self.J10_d = zero_tensor
        self.J10_theta = zero_tensor

        self.J11_a = 0.05004 * one_tensor
        self.J11_alpha = zero_tensor
        self.J11_d = zero_tensor
        self.J11_theta = zero_tensor

        self.J12_a = 0.05004 * one_tensor
        self.J12_alpha = zero_tensor
        self.J12_d = zero_tensor
        self.J12_theta = zero_tensor

        self.J13_a = 0.04454 * one_tensor
        self.J13_alpha = zero_tensor
        self.J13_d = zero_tensor
        self.J13_theta = zero_tensor

        self.J14_a = 0.026 * one_tensor
        self.J14_alpha = zero_tensor
        self.J14_d = zero_tensor
        self.J14_theta = zero_tensor

        self.J15_a = 0.032 * one_tensor
        self.J15_alpha = zero_tensor
        self.J15_d = zero_tensor
        self.J15_theta = zero_tensor

        self.J16_a = 0.032 * one_tensor
        self.J16_alpha = zero_tensor
        self.J16_d = zero_tensor
        self.J16_theta = zero_tensor

        self.J17_a = 0.022 * one_tensor
        self.J17_alpha = zero_tensor
        self.J17_d = zero_tensor
        self.J17_theta = zero_tensor

        if to_mano_frame:
            base_2_world = torch.tensor([[0.0, 0.0, -1.0, 0.16],
                                              [0.0, -1.0, 0.0, -0.0075],
                                              [-1.0, 0.0, 0.0, 0.0],
                                              [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

            base_2_world_normal = torch.tensor([[0.0, 0.0, -1.0, 0.0],
                                                     [0.0, -1.0, 0.0, 0.0],
                                                     [-1.0, 0.0, 0.0, 0.0],
                                                     [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
            pose = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
            pose[:3, :3] = roma.rotvec_to_rotmat(torch.tensor([0., -0.175, 0.0], dtype=torch.float32))
            base_2_world = torch.matmul(pose, base_2_world)
            base_2_world_normal = torch.matmul(pose, base_2_world_normal)
        else:
            base_2_world = torch.from_numpy(np.identity(4).astype(np.float32))
            base_2_world_normal = torch.from_numpy(np.identity(4).astype(np.float32))
        self.register_buffer('base_2_world', base_2_world)
        self.register_buffer('base_2_world_normal', base_2_world_normal)
        # transformation of palm_1 to base link
        p1_2_base = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, -0.01313],
                                  [0.0, 0.0, 1.0, 0.032],
                                  [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

        p1_2_base_normal = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0, 0.0],
                                         [0.0, 0.0, 1.0, 0.0],
                                         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        self.register_buffer('p1_2_base', p1_2_base)
        self.register_buffer('p1_2_base_normal', p1_2_base_normal)

        # transformation of palm_2 to palm_1
        p2_2_p1 = torch.tensor([[1.0, 0.0, 0, 0.0184],
                                [0.0, 1, 0, 0.006],
                                [0.0, 0.0, 1.0, 0.0375],
                                [0, 0, 0, 1]], dtype=torch.float32)

        p2_2_p1_normal = torch.tensor([[1.0, 0.0, 0, 0],
                                       [0.0, 1, 0, 0.0],
                                       [0.0, 0.0, 1.0, 0],
                                       [0, 0, 0, 1]], dtype=torch.float32)
        self.register_buffer('p2_2_p1', p2_2_p1)
        self.register_buffer('p2_2_p1_normal', p2_2_p1_normal)

        # transformation of thumb to palm_1
        thumb_2_p1 = torch.tensor([[0, -1, 0, -0.0169],
                                   [0.966, 0, 0.2588, 0.02626],
                                   [-0.2588, 0.0, 0.9659, 0],
                                   [0, 0, 0, 1]], dtype=torch.float32)

        thumb_2_p1_normal = torch.tensor([[0, -1, 0, 0],
                                          [0.966, 0, 0.2588, 0],
                                          [-0.2588, 0.0, 0.9659, 0],
                                          [0, 0, 0, 1]], dtype=torch.float32)
        self.register_buffer('thumb_2_p1', thumb_2_p1)
        self.register_buffer('thumb_2_p1_normal', thumb_2_p1_normal)

        # transformation of fore to palm_1
        fore_2_p1 = torch.tensor([[0, -1, 0, -0.025],
                                  [0.0, 0, -1, 0.0],
                                  [1, 0.0, 0.0, 0.110],
                                  [0, 0, 0, 1]], dtype=torch.float32)

        fore_2_p1_normal = torch.tensor([[0, -1, 0, 0],
                                         [0.0, 0, -1, 0.0],
                                         [1, 0.0, 0.0, 0.0],
                                         [0, 0, 0, 1]], dtype=torch.float32)
        self.register_buffer('fore_2_p1', fore_2_p1)
        self.register_buffer('fore_2_p1_normal', fore_2_p1_normal)

        # middle
        middle_2_p1 = torch.tensor([[0, -1, 0, 0.0],
                                    [0.0, 0, -1, 0.0],
                                    [1, 0.0, 0.0, 0.110],
                                    [0, 0, 0, 1]], dtype=torch.float32)

        middle_2_p1_normal = torch.tensor([[0, -1, 0, 0],
                                           [0.0, 0, -1, 0.0],
                                           [1, 0.0, 0.0, 0.0],
                                           [0, 0, 0, 1]], dtype=torch.float32)
        self.register_buffer('middle_2_p1', middle_2_p1)
        self.register_buffer('middle_2_p1_normal', middle_2_p1_normal)

        # ring
        ring_2_p2 = torch.tensor([[0, 1, 0, 0.003855],
                                  [0.0, 0, 1, -0.006],
                                  [1, 0.0, 0.0, 0.0655],
                                  [0, 0, 0, 1]], dtype=torch.float32)

        ring_2_p2_normal = torch.tensor([[0, 1, 0, 0],
                                         [0.0, 0, 1, 0.0],
                                         [1, 0.0, 0.0, 0.0],
                                         [0, 0, 0, 1]], dtype=torch.float32)
        self.register_buffer('ring_2_p2', ring_2_p2)
        self.register_buffer('ring_2_p2_normal', ring_2_p2_normal)

        # little
        little_2_p2 = torch.tensor([[0, 1, 0, 0.025355],
                                    [0.0, 0, 1, -0.006],
                                    [1, 0.0, 0.0, 0.056],
                                    [0, 0, 0, 1]], dtype=torch.float32)

        little_2_p2_normal = torch.tensor([[0, 1, 0, 0],
                                           [0.0, 0, 1, 0.0],
                                           [1, 0.0, 0.0, 0.0],
                                           [0, 0, 0, 1]], dtype=torch.float32)
        self.register_buffer('little_2_p2', little_2_p2)
        self.register_buffer('little_2_p2_normal', little_2_p2_normal)



        meshes = self.load_meshes()

        # righthand_base
        self.register_buffer('righthand', meshes["righthand_base"][0])
        self.register_buffer('righthand_normal', meshes["righthand_base"][2])

        # palm_1 (fore and middle)
        self.register_buffer('h10', meshes['h10'][0])
        self.register_buffer('h10_normal', meshes['h10'][2])

        # palm_2 (ring and little)
        self.register_buffer('h11', meshes['h11'][0])
        self.register_buffer('h11_normal', meshes['h11'][2])

        # thumb
        self.register_buffer('d10', meshes['d10'][0])
        self.register_buffer('d10_normal', meshes['d10'][2])
        self.register_buffer('d11', meshes['d11'][0])
        self.register_buffer('d11_normal', meshes['d11'][2])
        self.register_buffer('d12', meshes['d12'][0])
        self.register_buffer('d12_normal', meshes['d12'][2])
        self.register_buffer('d13', meshes['d13'][0])
        self.register_buffer('d13_normal', meshes['d13'][2])

        # fore
        self.register_buffer('f10', meshes['f10'][0])
        self.register_buffer('f10_normal', meshes['f10'][2])
        self.register_buffer('f11', meshes['f11'][0])
        self.register_buffer('f11_normal', meshes['f11'][2])
        self.register_buffer('f12', meshes['f12'][0])
        self.register_buffer('f12_normal', meshes['f12'][2])
        self.register_buffer('f13', meshes['f13'][0])
        self.register_buffer('f13_normal', meshes['f13'][2])

        # middle
        self.register_buffer('f20', meshes['f20'][0])
        self.register_buffer('f20_normal', meshes['f20'][2])
        self.register_buffer('f21', meshes['f21'][0])
        self.register_buffer('f21_normal', meshes['f21'][2])
        self.register_buffer('f22', meshes['f22'][0])
        self.register_buffer('f22_normal', meshes['f22'][2])
        self.register_buffer('f23', meshes['f23'][0])
        self.register_buffer('f23_normal', meshes['f23'][2])

        # ring
        self.register_buffer('f30', meshes['f30'][0])
        self.register_buffer('f30_normal', meshes['f30'][2])
        self.register_buffer('f31', meshes['f31'][0])
        self.register_buffer('f31_normal', meshes['f31'][2])
        self.register_buffer('f32', meshes['f32'][0])
        self.register_buffer('f32_normal', meshes['f32'][2])
        self.register_buffer('f33', meshes['f33'][0])
        self.register_buffer('f33_normal', meshes['f33'][2])

        # little
        self.register_buffer('f40', meshes['f40'][0])
        self.register_buffer('f40_normal', meshes['f40'][2])
        self.register_buffer('f41', meshes['f41'][0])
        self.register_buffer('f41_normal', meshes['f41'][2])
        self.register_buffer('f42', meshes['f42'][0])
        self.register_buffer('f42_normal', meshes['f42'][2])
        self.register_buffer('f43', meshes['f43'][0])
        self.register_buffer('f43_normal', meshes['f43'][2])

        self.gripper_faces = [
            meshes["righthand_base"][1],
            meshes['h10'][1], meshes['h11'][1],
            meshes['d10'][1], meshes['d11'][1], meshes['d12'][1], meshes['d13'][1],
            meshes['f10'][1], meshes['f11'][1], meshes['f12'][1], meshes['f13'][1],
            meshes['f20'][1], meshes['f21'][1], meshes['f22'][1], meshes['f23'][1],
            meshes['f30'][1], meshes['f31'][1], meshes['f32'][1], meshes['f33'][1],
            meshes['f40'][1], meshes['f41'][1], meshes['f42'][1], meshes['f43'][1],
        ]

        self.register_buffer("roty_90",
                             torch.from_numpy(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])).float())

    def load_meshes(self):
        mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/../meshes/svh_hand_new/*"
        mesh_files = glob.glob(mesh_path)
        mesh_files = [f for f in mesh_files if os.path.isfile(f)]
        meshes = {}

        for mesh_file in mesh_files:
            name = os.path.basename(mesh_file)[:-4]

            if self.show_mesh:
                mesh = trimesh.load(mesh_file)
                temp = torch.ones(mesh.vertices.shape[0], 1).float()
                vertex_normals = copy.deepcopy(mesh.vertex_normals)
                meshes[name] = [
                    torch.cat((torch.FloatTensor(np.array(mesh.vertices)), temp), dim=-1),
                    mesh.faces,
                    torch.cat((torch.FloatTensor(vertex_normals), temp), dim=-1).to(torch.float)
                ]
            # write by v-wewei 2023.01.14
            else:
                vertex_path = mesh_file.replace('svh_hand_new', 'svh_hand_points').replace('.stl', '.npy')

                points_info = np.load(vertex_path)
                # idxs = np.arange(len(points_info))
                idxs = np.load(os.path.dirname(os.path.realpath(__file__)) + '/../meshes/visible_point_indices/{}.npy'.format(name))
                temp = torch.ones(idxs.shape[0], 1)

                meshes[name] = [
                    torch.cat((torch.FloatTensor(points_info[idxs, :3]), temp), dim=-1),
                    torch.zeros([0]),  # no real meanning, just for placeholder
                    torch.cat((torch.FloatTensor(points_info[idxs, 3:6]), temp), dim=-1).to(torch.float)
                ]

        return meshes

    # @timeCalc
    def forward(self, pose, theta):
        """[summary]
        Args:
            pose (Tensor (batch_size x 4 x 4)): The pose of the base link of the hand as a translation matrix.
            theta (Tensor (batch_size x 15)): The seven degrees of freedome of the Barrett hand. The first column specifies the angle between
            fingers F1 and F2,  the second to fourth column specifies the joint angle around the proximal link of each finger while the fifth
            to the last column specifies the joint angle around the distal link for each finger

       """
        device=pose.device
        batch_size = pose.shape[0]
        pose_normal = pose.clone()

        pose_normal[:, :3, 3] = torch.zeros(3, device=pose.device)

        # right_hand_base
        right_hand_vertices = self.righthand.repeat(batch_size, 1, 1)
        T_base_2_w = torch.matmul(pose, self.base_2_world)
        right_hand_vertices = torch.matmul(T_base_2_w,
                                           right_hand_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        righthand_vertices_normal = self.righthand_normal.repeat(batch_size, 1, 1)
        T_base_2_w_normal = torch.matmul(pose_normal, self.base_2_world_normal)
        righthand_vertices_normal = torch.matmul(T_base_2_w_normal,
                                                 righthand_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # palm_1
        h10_vertices = self.h10.repeat(batch_size, 1, 1)
        T_p1_2_w = torch.matmul(T_base_2_w, self.p1_2_base)
        h10_vertices = torch.matmul(T_p1_2_w,
                                    h10_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]
        h10_vertices_normal = self.h10_normal.repeat(batch_size, 1, 1)
        T_p1_2_w_normal = torch.matmul(T_base_2_w_normal, self.p1_2_base_normal)
        h10_vertices_normal = torch.matmul(T_p1_2_w_normal,
                                           h10_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # palm_2
        h11_vertices = self.h11.repeat(batch_size, 1, 1)

        T_J5 = self.forward_kinematics(self.J5_a, self.J5_alpha,
                                       self.J5_d, theta[:, 0]/3 + self.J5_theta, batch_size,  device)
        T_p2_2_w = torch.matmul(torch.matmul(T_p1_2_w, self.p2_2_p1), T_J5)
        h11_vertices = torch.matmul(T_p2_2_w,
                                    h11_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        h11_vertices_normal = self.h11_normal.repeat(batch_size, 1, 1)
        T_p2_2_w_normal = torch.matmul(torch.matmul(T_p1_2_w_normal, self.p2_2_p1_normal), T_J5)
        h11_vertices_normal = torch.matmul(T_p2_2_w_normal,
                                           h11_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # thumb
        d10_vertices = self.d10.repeat(batch_size, 1, 1)

        T_J1 = self.forward_kinematics(self.J1_a, self.J1_alpha,
                                       self.J1_d, -(theta[:, 0] - 0.26) + self.J1_theta, batch_size,  device)
        T_thumb0_2_w = torch.matmul(torch.matmul(T_p1_2_w, self.thumb_2_p1), T_J1)
        d10_vertices = torch.matmul(T_thumb0_2_w,
                                    d10_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d10_vertices_normal = self.d10_normal.repeat(batch_size, 1, 1)
        T_thumb0_2_w_normal = torch.matmul(torch.matmul(T_p1_2_w_normal, self.thumb_2_p1_normal), T_J1)
        d10_vertices_normal = torch.matmul(T_thumb0_2_w_normal,
                                           d10_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d11_vertices = self.d11.repeat(batch_size, 1, 1)
        T_J2 = self.forward_kinematics(self.J2_a, self.J2_alpha,
                                       self.J2_d, (theta[:, 1] - 0.17) + self.J2_theta - 0.9704, batch_size,  device)

        T_thumb1_2_w = torch.matmul(torch.matmul(T_thumb0_2_w, self.roty_90), T_J2)
        # T_thumb1_2_w = torch.matmul(T_thumb0_2_w, T_J2)
        d11_vertices = torch.matmul(T_thumb1_2_w,  # T_thumb1_2_w
                                    d11_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d11_vertices_normal = self.d11_normal.repeat(batch_size, 1, 1)
        T_thumb1_2_w_normal = torch.matmul(torch.matmul(T_thumb0_2_w_normal, self.roty_90), T_J2)
        d11_vertices_normal = torch.matmul(T_thumb1_2_w_normal,
                                           d11_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d12_vertices = self.d12.repeat(batch_size, 1, 1)
        T_J3 = self.forward_kinematics(self.J3_a, self.J3_alpha,
                                       self.J3_d, (1.01511 * theta[:, 1] + 0.35) + self.J3_theta, batch_size,  device)
        T_thumb2_2_w = torch.matmul(T_thumb1_2_w, T_J3)
        d12_vertices = torch.matmul(T_thumb2_2_w,
                                    d12_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d12_vertices_normal = self.d12_normal.repeat(batch_size, 1, 1)
        T_thumb2_2_w_normal = torch.matmul(T_thumb1_2_w_normal, T_J3)
        d12_vertices_normal = torch.matmul(T_thumb2_2_w_normal,
                                           d12_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d13_vertices = self.d13.repeat(batch_size, 1, 1)
        T_J4 = self.forward_kinematics(self.J4_a, self.J4_alpha,
                                       self.J4_d, (1.44889 * theta[:, 1] + 0.087) + self.J4_theta, batch_size,  device)
        T_thumb3_2_w = torch.matmul(T_thumb2_2_w, T_J4)
        d13_vertices = torch.matmul(T_thumb3_2_w,
                                    d13_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        d13_vertices_normal = self.d13_normal.repeat(batch_size, 1, 1)
        T_thumb3_2_w_normal = torch.matmul(T_thumb2_2_w_normal, T_J4)
        d13_vertices_normal = torch.matmul(T_thumb3_2_w_normal,
                                           d13_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # fore
        f10_vertices = self.f10.repeat(batch_size, 1, 1)
        T_J6a = self.forward_kinematics(self.J6a_a, self.J6a_alpha,
                                        self.J6a_d, (0.8 - theta[:, 2]) / 0.8 * (theta[:, 8] - 0.18) + self.J6a_theta,
                                        batch_size,  device)  # 1x Finger_spread
        T_fore_2_w = torch.matmul(torch.matmul(T_p1_2_w, self.fore_2_p1), T_J6a)
        f10_vertices = torch.matmul(T_fore_2_w,
                                    f10_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f10_vertices_normal = self.f10_normal.repeat(batch_size, 1, 1)
        T_fore_2_w_normal = torch.matmul(torch.matmul(T_p1_2_w_normal, self.fore_2_p1_normal), T_J6a)
        f10_vertices_normal = torch.matmul(T_fore_2_w_normal,
                                           f10_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f11_vertices = self.f11.repeat(batch_size, 1, 1)
        T_J6b = self.forward_kinematics(self.J6b_a, self.J6b_alpha,
                                        self.J6b_d, (theta[:, 2] + 0.10) + self.J6b_theta, batch_size,  device)
        T_fore1_2_w = torch.matmul(T_fore_2_w, T_J6b)
        f11_vertices = torch.matmul(T_fore1_2_w,
                                    f11_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f11_vertices_normal = self.f11_normal.repeat(batch_size, 1, 1)
        T_fore1_2_w_normal = torch.matmul(T_fore_2_w_normal, T_J6b)
        f11_vertices_normal = torch.matmul(T_fore1_2_w_normal,
                                           f11_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f12_vertices = self.f12.repeat(batch_size, 1, 1)
        T_J10 = self.forward_kinematics(self.J10_a, self.J10_alpha,
                                        self.J10_d, (theta[:, 3] + 0.26) + self.J10_theta, batch_size,  device)
        T_fore2_2_w = torch.matmul(T_fore1_2_w, T_J10)
        f12_vertices = torch.matmul(T_fore2_2_w,
                                    f12_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f12_vertices_normal = self.f12_normal.repeat(batch_size, 1, 1)
        T_fore2_2_w_normal = torch.matmul(T_fore1_2_w_normal, T_J10)
        f12_vertices_normal = torch.matmul(T_fore2_2_w_normal,
                                           f12_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f13_vertices = self.f13.repeat(batch_size, 1, 1)
        T_J14 = self.forward_kinematics(self.J14_a, self.J14_alpha,
                                        self.J14_d, 1.045 * theta[:, 3] + self.J14_theta, batch_size,  device)
        T_fore3_2_w = torch.matmul(T_fore2_2_w, T_J14)
        f13_vertices = torch.matmul(T_fore3_2_w,
                                    f13_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f13_vertices_normal = self.f13_normal.repeat(batch_size, 1, 1)
        T_fore3_2_w_normal = torch.matmul(T_fore2_2_w_normal, T_J14)
        f13_vertices_normal = torch.matmul(T_fore3_2_w_normal,
                                           f13_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # middle
        f20_vertices = self.f20.repeat(batch_size, 1, 1)
        T_J7a = self.forward_kinematics(self.J7a_a, self.J7a_alpha,
                                        self.J7a_d, self.J7a_theta, batch_size,  device)  # No Finger_spread
        T_middle_2_w = torch.matmul(torch.matmul(T_p1_2_w, self.middle_2_p1), T_J7a)
        f20_vertices = torch.matmul(T_middle_2_w,
                                    f20_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f20_vertices_normal = self.f20_normal.repeat(batch_size, 1, 1)
        T_middle_2_w_normal = torch.matmul(torch.matmul(T_p1_2_w_normal, self.middle_2_p1_normal), T_J7a)
        f20_vertices_normal = torch.matmul(T_middle_2_w_normal,
                                           f20_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f21_vertices = self.f21.repeat(batch_size, 1, 1)
        T_J7b = self.forward_kinematics(self.J7b_a, self.J7b_alpha,
                                        self.J7b_d, (theta[:, 4] + 0.10) + self.J7b_theta, batch_size,  device)
        T_middle1_2_w = torch.matmul(T_middle_2_w, T_J7b)
        f21_vertices = torch.matmul(T_middle1_2_w,
                                    f21_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f21_vertices_normal = self.f21_normal.repeat(batch_size, 1, 1)
        T_middle1_2_w_normal = torch.matmul(T_middle_2_w_normal, T_J7b)
        f21_vertices_normal = torch.matmul(T_middle1_2_w_normal,
                                           f21_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f22_vertices = self.f22.repeat(batch_size, 1, 1)
        T_J11 = self.forward_kinematics(self.J11_a, self.J11_alpha,
                                        self.J11_d, (theta[:, 5] + 0.26) + self.J11_theta, batch_size,  device)
        T_middle2_2_w = torch.matmul(T_middle1_2_w, T_J11)
        f22_vertices = torch.matmul(T_middle2_2_w,
                                    f22_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f22_vertices_normal = self.f22_normal.repeat(batch_size, 1, 1)
        T_middle2_2_w_normal = torch.matmul(T_middle1_2_w_normal, T_J11)
        f22_vertices_normal = torch.matmul(T_middle2_2_w_normal,
                                           f22_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f23_vertices = self.f23.repeat(batch_size, 1, 1)
        T_J15 = self.forward_kinematics(self.J15_a, self.J15_alpha,
                                        self.J15_d, 1.0434 * theta[:, 5] + self.J15_theta, batch_size,  device)
        T_middle3_2_w = torch.matmul(T_middle2_2_w, T_J15)
        f23_vertices = torch.matmul(T_middle3_2_w,
                                    f23_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f23_vertices_normal = self.f23_normal.repeat(batch_size, 1, 1)
        T_middle3_2_w_normal = torch.matmul(T_middle2_2_w_normal, T_J15)
        f23_vertices_normal = torch.matmul(T_middle3_2_w_normal,
                                           f23_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # ring
        f30_vertices = self.f30.repeat(batch_size, 1, 1)
        T_J8a = self.forward_kinematics(self.J8a_a, self.J8a_alpha,
                                        self.J8a_d, (theta[:, 8] - 0.18) + self.J8a_theta,
                                        batch_size,  device)  # 1x Finger_spread
        T_ring_2_w = torch.matmul(torch.matmul(T_p2_2_w, self.ring_2_p2), T_J8a)
        f30_vertices = torch.matmul(T_ring_2_w,
                                    f30_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f30_vertices_normal = self.f30_normal.repeat(batch_size, 1, 1)
        T_ring_2_w_normal = torch.matmul(torch.matmul(T_p2_2_w_normal, self.ring_2_p2_normal), T_J8a)
        f30_vertices_normal = torch.matmul(T_ring_2_w_normal,
                                           f30_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f31_vertices = self.f31.repeat(batch_size, 1, 1)
        T_J8b = self.forward_kinematics(self.J8b_a, self.J8b_alpha,
                                        self.J8b_d, (theta[:, 6] + 0.23) + self.J8b_theta, batch_size,  device)
        T_ring1_2_w = torch.matmul(T_ring_2_w, T_J8b)
        f31_vertices = torch.matmul(T_ring1_2_w,
                                    f31_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f31_vertices_normal = self.f31_normal.repeat(batch_size, 1, 1)
        T_ring1_2_w_normal = torch.matmul(T_ring_2_w_normal, T_J8b)
        f31_vertices_normal = torch.matmul(T_ring1_2_w_normal,
                                           f31_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f32_vertices = self.f32.repeat(batch_size, 1, 1)
        T_J12 = self.forward_kinematics(self.J12_a, self.J12_alpha,
                                        self.J12_d, (1.3588 * theta[:, 6] + 0.42) + self.J12_theta, batch_size,  device)
        T_ring2_2_w = torch.matmul(T_ring1_2_w, T_J12)
        f32_vertices = torch.matmul(T_ring2_2_w,
                                    f32_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f32_vertices_normal = self.f32_normal.repeat(batch_size, 1, 1)
        T_ring2_2_w_normal = torch.matmul(T_ring1_2_w_normal, T_J12)
        f32_vertices_normal = torch.matmul(T_ring2_2_w_normal,
                                           f32_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f33_vertices = self.f33.repeat(batch_size, 1, 1)
        T_J16 = self.forward_kinematics(self.J16_a, self.J16_alpha,
                                        self.J16_d, (1.42093 * theta[:, 6] + 0.17) + self.J16_theta, batch_size,  device)
        T_ring3_2_w = torch.matmul(T_ring2_2_w, T_J16)
        f33_vertices = torch.matmul(T_ring3_2_w,
                                    f33_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f33_vertices_normal = self.f33_normal.repeat(batch_size, 1, 1)
        T_ring3_2_w_normal = torch.matmul(T_ring2_2_w_normal, T_J16)
        f33_vertices_normal = torch.matmul(T_ring3_2_w_normal,
                                           f33_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        # little
        f40_vertices = self.f40.repeat(batch_size, 1, 1)
        T_J9a = self.forward_kinematics(self.J9a_a, self.J9a_alpha,
                                        self.J9a_d, 2 * (theta[:, 8] - 0.18) + self.J9a_theta,
                                        batch_size,  device)  # 2x Finger_spread
        T_little_2_w = torch.matmul(torch.matmul(T_p2_2_w, self.little_2_p2), T_J9a)
        f40_vertices = torch.matmul(T_little_2_w,
                                    f40_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f40_vertices_normal = self.f40_normal.repeat(batch_size, 1, 1)
        T_little_2_w_normal = torch.matmul(torch.matmul(T_p2_2_w_normal, self.little_2_p2_normal), T_J9a)
        f40_vertices_normal = torch.matmul(T_little_2_w_normal,
                                           f40_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f41_vertices = self.f41.repeat(batch_size, 1, 1)
        T_J9b = self.forward_kinematics(self.J9b_a, self.J9b_alpha,
                                        self.J9b_d, (theta[:, 7] + 0.23) + self.J9b_theta, batch_size,  device)
        T_little1_2_w = torch.matmul(T_little_2_w, T_J9b)
        f41_vertices = torch.matmul(T_little1_2_w,
                                    f41_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f41_vertices_normal = self.f41_normal.repeat(batch_size, 1, 1)
        T_little1_2_w_normal = torch.matmul(T_little_2_w_normal, T_J9b)
        f41_vertices_normal = torch.matmul(T_little1_2_w_normal,
                                           f41_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f42_vertices = self.f42.repeat(batch_size, 1, 1)
        T_J13 = self.forward_kinematics(self.J13_a, self.J13_alpha,
                                        self.J13_d, (1.3588 * theta[:, 7] + 0.24) + self.J13_theta, batch_size,  device)
        T_little2_2_w = torch.matmul(T_little1_2_w, T_J13)
        f42_vertices = torch.matmul(T_little2_2_w,
                                    f42_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f42_vertices_normal = self.f42_normal.repeat(batch_size, 1, 1)
        T_little2_2_w_normal = torch.matmul(T_little1_2_w_normal, T_J13)
        f42_vertices_normal = torch.matmul(T_little2_2_w_normal,
                                           f42_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f43_vertices = self.f43.repeat(batch_size, 1, 1)
        T_J17 = self.forward_kinematics(self.J17_a, self.J17_alpha,
                                        self.J17_d, 1.42307 * theta[:, 7] + self.J17_theta, batch_size,  device)
        T_little3_2_w = torch.matmul(T_little2_2_w, T_J17)
        f43_vertices = torch.matmul(T_little3_2_w,
                                    f43_vertices.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        f43_vertices_normal = self.f43_normal.repeat(batch_size, 1, 1)
        T_little3_2_w_normal = torch.matmul(T_little2_2_w_normal, T_J17)
        f43_vertices_normal = torch.matmul(T_little3_2_w_normal,
                                           f43_vertices_normal.transpose(2, 1)).transpose(1, 2)[:, :, :3]

        return right_hand_vertices, h10_vertices, h11_vertices, \
               d10_vertices, d11_vertices, d12_vertices, d13_vertices, \
               f10_vertices, f11_vertices, f12_vertices, f13_vertices, \
               f20_vertices, f21_vertices, f22_vertices, f23_vertices, \
               f30_vertices, f31_vertices, f32_vertices, f33_vertices, \
               f40_vertices, f41_vertices, f42_vertices, f43_vertices, \
               righthand_vertices_normal, h10_vertices_normal, h11_vertices_normal, \
               d10_vertices_normal, d11_vertices_normal, d12_vertices_normal, d13_vertices_normal, \
               f10_vertices_normal, f11_vertices_normal, f12_vertices_normal, f13_vertices_normal, \
               f20_vertices_normal, f21_vertices_normal, f22_vertices_normal, f23_vertices_normal, \
               f30_vertices_normal, f31_vertices_normal, f32_vertices_normal, f33_vertices_normal, \
               f40_vertices_normal, f41_vertices_normal, f42_vertices_normal, f43_vertices_normal

    # @timeCalc
    def forward_kinematics(self, A, alpha, D, theta, batch_size=1, device='cpu'):
        c_theta = torch.cos(theta)
        s_theta = torch.sin(theta)
        c_alpha = torch.cos(alpha)
        s_alpha = torch.sin(alpha)
        l_1_to_l = torch.zeros((batch_size, 4, 4), device=device)
        l_1_to_l[:, 0, 0] = c_theta
        l_1_to_l[:, 0, 1] = -s_theta
        l_1_to_l[:, 0, 3] = A
        l_1_to_l[:, 1, 0] = s_theta * c_alpha
        l_1_to_l[:, 1, 1] = c_theta * c_alpha
        l_1_to_l[:, 1, 2] = -s_alpha
        l_1_to_l[:, 1, 3] = -s_alpha * D
        l_1_to_l[:, 2, 0] = s_theta * s_alpha
        l_1_to_l[:, 2, 1] = c_theta * s_alpha
        l_1_to_l[:, 2, 2] = c_alpha
        l_1_to_l[:, 2, 3] = c_alpha * D
        l_1_to_l[:, 3, 3] = 1
        return l_1_to_l

    def get_hand_mesh(self, vertices_list, faces, save_mesh=False, path='./output_mesh'):
        if save_mesh:
            assert os.path.exists(path)

        right_hand_verts = vertices_list[0]
        right_hand_faces = faces[0]
        palm1_verts = vertices_list[1]
        palm1_faces = faces[1]
        palm2_verts = vertices_list[2]
        palm2_faces = faces[2]

        thumb_base_verts = vertices_list[3]
        thumb_base_faces = faces[3]
        thumb_proximal_verts = vertices_list[4]
        thumb_proximal_faces = faces[4]
        thumb_medial_verts = vertices_list[5]
        thumb_medial_faces = faces[5]
        thumb_distal_verts = vertices_list[6]
        thumb_distal_faces = faces[6]

        fore_base_verts = vertices_list[7]
        fore_base_faces = faces[7]
        fore_proximal_verts = vertices_list[8]
        fore_proximal_faces = faces[8]
        fore_medial_verts = vertices_list[9]
        fore_medial_faces = faces[9]
        fore_distal_verts = vertices_list[10]
        fore_distal_faces = faces[10]

        middle_base_verts = vertices_list[11]
        middle_base_faces = faces[11]
        middle_proximal_verts = vertices_list[12]
        middle_proximal_faces = faces[12]
        middle_medial_verts = vertices_list[13]
        middle_medial_faces = faces[13]
        middle_distal_verts = vertices_list[14]
        middle_distal_faces = faces[14]

        ring_base_verts = vertices_list[15]
        ring_base_faces = faces[15]
        ring_proximal_verts = vertices_list[16]
        ring_proximal_faces = faces[16]
        ring_medial_verts = vertices_list[17]
        ring_medial_faces = faces[17]
        ring_distal_verts = vertices_list[18]
        ring_distal_faces = faces[18]

        little_base_verts = vertices_list[19]
        little_base_faces = faces[19]
        little_proximal_verts = vertices_list[20]
        little_proximal_faces = faces[20]
        little_medial_verts = vertices_list[21]
        little_medial_faces = faces[21]
        little_distal_verts = vertices_list[22]
        little_distal_faces = faces[22]

        if save_mesh:
            save_to_mesh(right_hand_verts, right_hand_faces, '{}/svh_base.obj'.format(path))
            save_to_mesh(palm1_verts, palm1_faces, '{}/svh_palm1.obj'.format(path))
            save_to_mesh(palm2_verts, palm2_faces, '{}/svh_palm2.obj'.format(path))
            save_to_mesh(thumb_base_verts, thumb_base_faces, '{}/svh_thumb_base.obj'.format(path))
            save_to_mesh(thumb_proximal_verts, thumb_proximal_faces, '{}/svh_thumb_proximal.obj'.format(path))
            save_to_mesh(thumb_medial_verts, thumb_medial_faces, '{}/svh_thumb_medial.obj'.format(path))
            save_to_mesh(thumb_distal_verts, thumb_distal_faces, '{}/svh_thumb_distal.obj'.format(path))
            save_to_mesh(fore_base_verts, fore_base_faces, '{}/svh_fore_base.obj'.format(path))
            save_to_mesh(fore_proximal_verts, fore_proximal_faces, '{}/svh_fore_proximal.obj'.format(path))
            save_to_mesh(fore_medial_verts, fore_medial_faces, '{}/svh_fore_medial.obj'.format(path))
            save_to_mesh(fore_distal_verts, fore_distal_faces, '{}/svh_fore_distal.obj'.format(path))
            save_to_mesh(middle_base_verts, middle_base_faces, '{}/svh_middle_base.obj'.format(path))
            save_to_mesh(middle_proximal_verts, middle_proximal_faces, '{}/svh_middle_proximal.obj'.format(path))
            save_to_mesh(middle_medial_verts, middle_medial_faces, '{}/svh_middle_medial.obj'.format(path))
            save_to_mesh(middle_distal_verts, middle_distal_faces, '{}/svh_middle_distal.obj'.format(path))
            save_to_mesh(ring_base_verts, ring_base_faces, '{}/svh_ring_base.obj'.format(path))
            save_to_mesh(ring_proximal_verts, ring_proximal_faces, '{}/svh_ring_proximal.obj'.format(path))
            save_to_mesh(ring_medial_verts, ring_medial_faces, '{}/svh_ring_medial.obj'.format(path))
            save_to_mesh(ring_distal_verts, ring_distal_faces, '{}/svh_ring_distal.obj'.format(path))
            save_to_mesh(little_base_verts, little_base_faces, '{}/svh_little_base.obj'.format(path))
            save_to_mesh(little_proximal_verts, little_proximal_faces, '{}/svh_little_proximal.obj'.format(path))
            save_to_mesh(little_medial_verts, little_medial_faces, '{}/svh_little_medial.obj'.format(path))
            save_to_mesh(little_distal_verts, little_distal_faces, '{}/svh_little_distal.obj'.format(path))

            hand_mesh = []
            for root, dirs, files in os.walk('{}'.format(path)):
                for filename in files:
                    if filename.endswith('.obj'):
                        filepath = os.path.join(root, filename)
                        mesh = trimesh.load_mesh(filepath)
                        hand_mesh.append(mesh)
            hand_mesh = np.sum(hand_mesh)
        else:
            right_hand_mesh = trimesh.Trimesh(right_hand_verts, right_hand_faces)
            palm1_mesh = trimesh.Trimesh(palm1_verts, palm1_faces)
            palm2_mesh = trimesh.Trimesh(palm2_verts, palm2_faces)
            thumb_base_mesh = trimesh.Trimesh(thumb_base_verts, thumb_base_faces)
            thumb_proximal_mesh = trimesh.Trimesh(thumb_proximal_verts, thumb_proximal_faces)
            thumb_medial_mesh = trimesh.Trimesh(thumb_medial_verts, thumb_medial_faces)
            thumb_distal_mesh = trimesh.Trimesh(thumb_distal_verts, thumb_distal_faces)
            fore_base_mesh = trimesh.Trimesh(fore_base_verts, fore_base_faces)
            fore_proximal_mesh = trimesh.Trimesh(fore_proximal_verts, fore_proximal_faces)
            fore_medial_mesh = trimesh.Trimesh(fore_medial_verts, fore_medial_faces)
            fore_distal_mesh = trimesh.Trimesh(fore_distal_verts, fore_distal_faces)
            middle_base_mesh = trimesh.Trimesh(middle_base_verts, middle_base_faces)
            middle_proximal_mesh = trimesh.Trimesh(middle_proximal_verts, middle_proximal_faces)
            middle_medial_mesh = trimesh.Trimesh(middle_medial_verts, middle_medial_faces)
            middle_distal_mesh = trimesh.Trimesh(middle_distal_verts, middle_distal_faces)
            ring_base_mesh = trimesh.Trimesh(ring_base_verts, ring_base_faces)
            ring_proximal_mesh = trimesh.Trimesh(ring_proximal_verts, ring_proximal_faces)
            ring_medial_mesh = trimesh.Trimesh(ring_medial_verts, ring_medial_faces)
            ring_distal_mesh = trimesh.Trimesh(ring_distal_verts, ring_distal_faces)
            little_base_mesh = trimesh.Trimesh(little_base_verts, little_base_faces)
            little_proximal_mesh = trimesh.Trimesh(little_proximal_verts, little_proximal_faces)
            little_medial_mesh = trimesh.Trimesh(little_medial_verts, little_medial_faces)
            little_distal_mesh = trimesh.Trimesh(little_distal_verts, little_distal_faces)

            hand_mesh = [right_hand_mesh, palm1_mesh, palm2_mesh,
                         thumb_base_mesh, thumb_proximal_mesh, thumb_medial_mesh, thumb_distal_mesh,
                         fore_base_mesh, fore_proximal_mesh, fore_medial_mesh, fore_distal_mesh,
                         middle_base_mesh, middle_proximal_mesh, middle_medial_mesh, middle_distal_mesh,
                         ring_base_mesh, ring_proximal_mesh, ring_medial_mesh, ring_distal_mesh,
                         little_base_mesh, little_proximal_mesh, little_medial_mesh, little_distal_mesh
                         ]

        return hand_mesh

    def get_forward_hand_mesh(self, pose, theta, save_mesh=False, path='./output_mesh'):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)

        hand_vertices_list = [[outputs[j][i].detach().cpu().numpy() for j in range(23)]
                              for i in range(batch_size)]

        hand_meshes = [self.get_hand_mesh(hand_vertices, self.gripper_faces, save_mesh=save_mesh, path=path) for
                       hand_vertices in hand_vertices_list]
        hand_meshes = [np.sum(hand_mesh) for hand_mesh in hand_meshes]

        return hand_meshes

    # @timeCalc
    def get_forward_vertices(self, pose, theta):
        batch_size = pose.size()[0]
        outputs = self.forward(pose, theta)

        hand_vertices = torch.cat((
                                   outputs[0].view(batch_size, -1, 3),
                                   outputs[1].view(batch_size, -1, 3),
                                   outputs[2].view(batch_size, -1, 3),
                                   outputs[3].view(batch_size, -1, 3),
                                   outputs[4].view(batch_size, -1, 3),
                                   outputs[5].view(batch_size, -1, 3),
                                   outputs[6].view(batch_size, -1, 3),
                                   outputs[7].view(batch_size, -1, 3),
                                   outputs[8].view(batch_size, -1, 3),
                                   outputs[9].view(batch_size, -1, 3),
                                   outputs[10].view(batch_size, -1, 3),
                                   outputs[11].view(batch_size, -1, 3),
                                   outputs[12].view(batch_size, -1, 3),
                                   outputs[13].view(batch_size, -1, 3),
                                   outputs[14].view(batch_size, -1, 3),
                                   outputs[15].view(batch_size, -1, 3),
                                   outputs[16].view(batch_size, -1, 3),
                                   outputs[17].view(batch_size, -1, 3),
                                   outputs[18].view(batch_size, -1, 3),
                                   outputs[19].view(batch_size, -1, 3),
                                   outputs[20].view(batch_size, -1, 3),
                                   outputs[21].view(batch_size, -1, 3),
                                   outputs[22].view(batch_size, -1, 3)), 1)

        hand_vertices_normal = torch.cat((
                                          outputs[23].view(batch_size, -1, 3),
                                          outputs[24].view(batch_size, -1, 3),
                                          outputs[25].view(batch_size, -1, 3),
                                          outputs[26].view(batch_size, -1, 3),
                                          outputs[27].view(batch_size, -1, 3),
                                          outputs[28].view(batch_size, -1, 3),
                                          outputs[29].view(batch_size, -1, 3),
                                          outputs[30].view(batch_size, -1, 3),
                                          outputs[31].view(batch_size, -1, 3),
                                          outputs[32].view(batch_size, -1, 3),
                                          outputs[33].view(batch_size, -1, 3),
                                          outputs[34].view(batch_size, -1, 3),
                                          outputs[35].view(batch_size, -1, 3),
                                          outputs[36].view(batch_size, -1, 3),
                                          outputs[37].view(batch_size, -1, 3),
                                          outputs[38].view(batch_size, -1, 3),
                                          outputs[39].view(batch_size, -1, 3),
                                          outputs[40].view(batch_size, -1, 3),
                                          outputs[41].view(batch_size, -1, 3),
                                          outputs[42].view(batch_size, -1, 3),
                                          outputs[43].view(batch_size, -1, 3),
                                          outputs[44].view(batch_size, -1, 3),
                                          outputs[45].view(batch_size, -1, 3)), 1)

        return hand_vertices, hand_vertices_normal


class SvhAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # vert_idx
        vert_idx = np.array([
            4646, 4779, 5143, 5109, 5112,
            3207, 2391,

            5398, 5786, 5925, 5831, 5895,
            2158,

            6208, 6428, 6585, 6615, 6620,
            2039, 2828,

            6783, 7158, 7407, 7308, 7368,
            3820, 3536,

            7707, 7856, 8051, 8056, 8063,

            # plus
            5669, 5891, 5780, 5740,
            6468, 6554, 6412, 6297,
            7214, 7389, 7122, 7144,
            7975, 8059

        ])
        self.register_buffer("vert_idx", torch.from_numpy(vert_idx).long())

    def forward(self, vertices):
        """
        vertices: TENSOR[N_BATCH, 4040, 3]
        """
        anchor_pos = vertices[:, self.vert_idx, :]
        return anchor_pos

    def pick_points(self, vertices: np.ndarray):
        import open3d as o3d
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print(vis.get_picked_points())
        return vis.get_picked_points()


if __name__ == "__main__":
    device = 'cuda'
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    # 0:Thumb Opposition, 1:Thumb Flexion, 2:Index Finger Proximal, 3:Index Finger Distal, 4:Middle Finger Proximal,
    # 5:Middle Finger Distal, 6:Ring Finger, 7:Pinky, 8:Spread
    # theta = torch.ones((1, 9), dtype=torch.float32).to(device) * 0.1
    theta = torch.zeros((1, 9), dtype=torch.float32).to(device)
    theta[0][0] = 0.45
    theta[0][2] = -0.15
    theta[0][3] = -0.15
    theta[0][4] = -0.15
    theta[0][5] = -0.15
    theta[0][6] = -0.15
    theta[0][7] = -0.15
    theta[0][8] = 0.3

    svh = SvhHandLayer(to_mano_frame=True, show_mesh=True).to(device)
    mesh = svh.get_forward_hand_mesh(pose, theta, save_mesh=False)
    mesh[0].show()


