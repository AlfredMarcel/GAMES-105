'''
Description: 说明
Author: Marcel
Date: 2023-04-13 19:32:25
'''
from bvh_utils import *
from scipy.spatial.transform import Rotation as R
#---------------你的代码------------------#
# translation 和 orientation 都是全局的
def skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = T_pose_vertex_translation.copy()  
    #---------------你的代码------------------#

    # [N,4,3]
    toj = T_pose_joint_translation[skinning_idx.reshape(-1),:].reshape(-1,4,3)
    rij = T_pose_vertex_translation[:,np.newaxis,:]-toj

    # [M,3,3]
    joint_orientation = R(joint_orientation).as_matrix()
    # [N,4,3,3]
    Qj = joint_orientation[skinning_idx.reshape(-1),:,:].reshape(-1,4,3,3)
    # [N,4,3]
    oj = joint_translation[skinning_idx.reshape(-1),:].reshape(-1,4,3)
    
    # [N,4,3]
    # tmp = np.einsum('abij,abi -> abj',Qj,rij) 就错了 相当于是拿坐标的转置左乘旋转矩阵
    tmp = np.einsum('abij,abj -> abi',Qj,rij)
    tmp = tmp+oj

    tmp = np.einsum('ij,ijk -> ik',skinning_weight,tmp)

    return tmp
