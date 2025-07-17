import numpy as np
import rerun as rr
from typing import List, Optional, Tuple, Generator
import time
from dataclasses import dataclass
import math
import cv2
from scipy.spatial.transform import Rotation as R

@dataclass
class BoundingBox3D:
    """3D边界框数据结构"""
    center: Tuple[float, float, float]  # (x, y, z) 中心坐标
    size: Tuple[float, float, float]    # (width, height, depth) 尺寸
    rotation: Optional[Tuple[float, float, float, float]] = None  # 四元数 (x, y, z, w)
    color: Optional[Tuple[int, int, int]] = None  # RGB颜色 (0-255)
    label: Optional[str] = None         # 标签文本

class CombinedVisualizer:
    def __init__(self):
        rr.init("Combined Visualizer", spawn=False)
        rr.serve_web(open_browser=True)
        self._frame_count = 0
        
        # 2D视图相关路径
        self._image_entity = "camera/image"  # 原始图像路径
        self._bbox_2d_entity = "camera/image_with_2d_bboxes"  # 带2D边界框图像路径
        self._text_entity = "bbox_info/parameters"  # 边界框参数文本路径
        
        # 3D视图相关路径
        self._bbox_3d_entity = "world/3d_bboxes"   # 3D边界框存储路径
        self._bbox_3d22d="camera/image_with_3d_bboxes"

        # 默认相机参数（可根据实际相机调整）
        self.camera_intrinsics = np.array([
            [500, 0, 320],  # fx, 0, cx
            [0, 500, 240],  # 0, fy, cy
            [0, 0, 1]       # 0, 0, 1
        ])
        self.dist_coeffs = np.zeros(5)  # 无畸变
        
        # 相机外参（世界坐标系到相机坐标系的变换）
        self.camera_pose = np.eye(4)  # 初始为单位矩阵（相机位于原点）
        self.camera_pose[:3, 3] = [0, 0, 0]  # 相机位置

    def set_camera_parameters(
        self,
        intrinsics: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        pose: Optional[np.ndarray] = None
    ):
        """设置相机参数"""
        self.camera_intrinsics = intrinsics
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs
        if pose is not None:
            self.camera_pose = pose

    def _project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """将3D点投影到图像平面"""
        # 转换为齐次坐标
        points_3d_hom = np.column_stack([points_3d, np.ones(len(points_3d))])
        
        # 世界坐标系到相机坐标系
        points_cam = (np.linalg.inv(self.camera_pose) @ points_3d_hom.T).T[:, :3]
        
        # 投影到图像平面
        rvec = np.zeros(3)  # 零旋转
        tvec = np.zeros(3)  # 零平移
        points_2d, _ = cv2.projectPoints(
            points_cam.reshape(-1, 1, 3),
            rvec,
            tvec,
            self.camera_intrinsics,
            self.dist_coeffs
        )
        
        return points_2d.reshape(-1, 2)

    def _get_bbox_corners(self, bbox: BoundingBox3D) -> np.ndarray:
        """获取3D边界框的8个角点"""
        # 计算未旋转时的角点（局部坐标系）
        half_w, half_h, half_d = bbox.size[0]/2, bbox.size[1]/2, bbox.size[2]/2
        corners_local = np.array([
            [-half_w, -half_h, -half_d],
            [ half_w, -half_h, -half_d],
            [ half_w,  half_h, -half_d],
            [-half_w,  half_h, -half_d],
            [-half_w, -half_h,  half_d],
            [ half_w, -half_h,  half_d],
            [ half_w,  half_h,  half_d],
            [-half_w,  half_h,  half_d]
        ])
        
        # 应用旋转（如果有）
        if bbox.rotation is not None:
            rotation = R.from_quat(bbox.rotation).as_matrix()
            corners_local = corners_local @ rotation.T
        
        # 转换到世界坐标系
        corners_world = corners_local + np.array(bbox.center)
        return corners_world

    def stream_image(self, 
                    image: np.ndarray, 
                    frame_idx: Optional[int] = None) -> None:
        """显示原始图像流"""
        assert image.dtype == np.uint8, "图像需要是uint8格式"
        assert image.ndim == 3, "图像需要是HWC格式"
        
        # 设置时间轴
        current_frame = frame_idx if frame_idx is not None else self._frame_count
        rr.set_time_sequence("frame", current_frame)
        
        # 记录原始图像数据
        rr.log(self._image_entity, rr.Image(image))

    def log_bbox_info(
        self,
        bboxes_2d: List[Tuple[float, float, float, float]],
        frame_idx: Optional[int] = None,
        class_labels: Optional[List[str]] = None,
        bbox_colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> None:
        """记录2D边界框参数信息"""
        # 设置时间轴
        current_frame = frame_idx if frame_idx is not None else self._frame_count
        rr.set_time_sequence("frame", current_frame)
        
        # 准备边界框参数文本
        bbox_info_text = "2D bbox parameters:\n\n"
        
        for i, bbox in enumerate(bboxes_2d):
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            
            bbox_info_text += (
                f"bbox {i}:\n"
                f"position: ({x_min:.1f}, {y_min:.1f}) -> ({x_max:.1f}, {y_max:.1f})\n"
                f"size:({width:.1f}x{height:.1f})\n"
                f"color: {bbox_colors[i] if bbox_colors and i < len(bbox_colors) else 'N/A'}\n"
                f"label: {class_labels[i] if class_labels and i < len(class_labels) else 'N/A'}\n\n"
            )
        
        rr.log(
            self._text_entity,
            rr.TextDocument(
                bbox_info_text,
                media_type=rr.MediaType.MARKDOWN,
            )
        )

    def visualize_with_2d_bboxes(
        self,
        image: np.ndarray,
        bboxes_2d: List[Tuple[float, float, float, float]],
        frame_idx: Optional[int] = None,
        class_labels: Optional[List[str]] = None,
        bbox_colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> None:
        """显示带2D边界框的图像"""
        assert image.dtype == np.uint8, "图像需要是uint8格式"
        assert image.ndim == 3, "图像需要是HWC格式"
        
        # 设置时间轴
        current_frame = frame_idx if frame_idx is not None else self._frame_count
        rr.set_time_sequence("frame", current_frame)
        
        # 记录带边界框的图像数据
        rr.log(self._bbox_2d_entity, rr.Image(image))
        
        # 记录边界框数据
        for i, bbox in enumerate(bboxes_2d):
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            
            # 创建bbox实体路径
            bbox_entity = f"{self._bbox_2d_entity}/bbox_{i}"
            
            # 设置bbox颜色（默认随机）
            color = bbox_colors[i] if bbox_colors and i < len(bbox_colors) else None
            
            # 记录矩形框
            rr.log(
                bbox_entity,
                rr.Boxes2D(
                    array=[x_min, y_min, width, height],
                    array_format=rr.Box2DFormat.XYWH,
                    colors=color,
                    labels=class_labels[i] if class_labels and i < len(class_labels) else None
                )
            )

    def visualize_with_3d_bboxes(
        self,
        image: np.ndarray,
        bboxes_3d: List[BoundingBox3D],
        frame_idx: Optional[int] = None,
        draw_projection: bool = False
    ) -> None:
        """
        可视化带3D边界框投影的图像
        :param image: 输入图像 (HWC格式, uint8)
        :param bboxes_3d: 3D边界框列表
        :param frame_idx: 自定义帧索引
        :param draw_projection: 是否在图像上绘制投影
        """
        assert image.dtype == np.uint8, "图像需要是uint8格式"
        assert image.ndim == 3, "图像需要是HWC格式"
        
        # 设置时间轴
        current_frame = frame_idx if frame_idx is not None else self._frame_count
        rr.set_time_sequence("frame", current_frame)
        if frame_idx is None:
            self._frame_count += 1

        # 如果需要绘制投影，先复制一份图像
        if draw_projection:
            image_with_proj = image.copy()
        else:
            image_with_proj = image

        # 记录3D边界框数据
        for i, bbox in enumerate(bboxes_3d):
            bbox_entity = f"{self._bbox_3d_entity}/box_{i}"
            
            # 记录3D边界框
            rr.log(
                bbox_entity,
                rr.Boxes3D(
                    centers=[bbox.center],
                    half_sizes=[s/2 for s in bbox.size],  # 转换为半尺寸
                    rotations=[bbox.rotation] if bbox.rotation else None,
                    colors=[bbox.color] if bbox.color else None,
                    labels=[bbox.label] if bbox.label else None
                )
            )
            
            # 在图像上绘制投影
            if draw_projection:
                corners_3d = self._get_bbox_corners(bbox)
                corners_2d = self._project_points(corners_3d)
                
                # 定义边界框的12条边（连接8个角点）
                lines = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                    [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                    [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
                ]
                
                # 绘制所有边
                for line in lines:
                    start = tuple(corners_2d[line[0]].astype(int))
                    end = tuple(corners_2d[line[1]].astype(int))
                    color = bbox.color if bbox.color else (255, 255, 255)
                    cv2.line(image_with_proj, start, end, color, 2)
                
                # 添加标签
                if bbox.label:
                    front_face_center = np.mean(corners_2d[[0,1,5,4]], axis=0).astype(int)
                    cv2.putText(
                        image_with_proj,
                        bbox.label,
                        tuple(front_face_center),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA
                    )
        
        # 更新带有投影的图像
        rr.log(self._bbox_3d22d, rr.Image(image_with_proj))
               
    def close(self) -> None:
        """显式关闭连接"""
        rr.disconnect()

def generate_image_stream(
    width: int = 640,
    height: int = 480,
    frames: int = 100
) -> Generator[np.ndarray, None, None]:
    """生成图像流"""
    for i in range(frames):
        # 生成渐变背景图像
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        rgb = np.stack([(gradient * (i % 3 + 1) / 3) for _ in range(3)], axis=-1)
        image = np.tile(rgb, (height, 1, 1))
        
        # 添加随机噪声
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        yield np.clip(image + noise, 0, 255).astype(np.uint8)

def generate_2d_bboxes(
    width: int = 640,
    height: int = 480,
    frames: int = 100,
    max_objects: int = 3
) -> Generator[List[Tuple[float, float, float, float]], None, None]:
    """生成2D边界框序列"""
    for _ in range(frames):
        num_objects = np.random.randint(1, max_objects + 1)
        bboxes = []
        
        for _ in range(num_objects):
            # 随机生成bbox坐标（确保在图像范围内）
            obj_width = np.random.randint(50, 200)
            obj_height = np.random.randint(50, 200)
            x_min = np.random.randint(0, width - obj_width)
            y_min = np.random.randint(0, height - obj_height)
            x_max = x_min + obj_width
            y_max = y_min + obj_height
            
            bboxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))
        
        yield bboxes

def generate_3d_bboxes(
    camera_pose: np.ndarray,
    frames: int = 100,
    max_objects: int = 3
) -> Generator[List[BoundingBox3D], None, None]:
    """生成3D边界框序列"""
    # 获取相机位置和朝向
    cam_pos = camera_pose[:3, 3]
    cam_forward = camera_pose[:3, 2]  # 相机前向向量
    
    for _ in range(frames):
        num_objects = np.random.randint(1, max_objects + 1)
        bboxes = []
        
        for j in range(num_objects):
            while True:
                # 在相机前方锥形区域内生成点
                forward_dist = np.random.uniform(4, 10)
                lateral_angle = np.random.uniform(-0.3, 0.3)
                vertical_angle = np.random.uniform(-0.2, 0.2)
                
                # 计算相对于相机的位置
                right = camera_pose[:3, 0]
                up = camera_pose[:3, 1]
                
                # 计算偏移量
                lateral_offset = np.tan(lateral_angle) * forward_dist
                vertical_offset = np.tan(vertical_angle) * forward_dist
                
                # 计算世界坐标
                center = (
                    cam_pos + 
                    cam_forward * forward_dist +
                    right * lateral_offset +
                    up * vertical_offset
                )
                
                # 检查是否在相机前方（点积为正）
                to_obj = center - cam_pos
                if np.dot(cam_forward, to_obj) > 0:
                    break
            
            # 根据距离调整大小，使近处的物体更大
            size_scale = 1.0 / (forward_dist / 5.0)
            size = (
                np.random.uniform(0.5, 1.5) * size_scale,  # width
                np.random.uniform(0.5, 1.5) * size_scale,  # height
                np.random.uniform(0.5, 1.5) * size_scale   # depth
            )
            
            color = (
                np.random.randint(150, 256),
                np.random.randint(150, 256),
                np.random.randint(150, 256)
            )
            label = f"Obj_{j}"
            
            # 使对象朝向相机
            look_dir = cam_pos - np.array(center)
            look_dir[1] = 0  # 保持水平
            look_dir = look_dir / np.linalg.norm(look_dir)
            
            # 计算四元数旋转
            angle = np.arctan2(look_dir[2], look_dir[0]) - np.pi/2
            axis = np.array([0, 1, 0])  # 绕Y轴旋转
            rotation = tuple(np.append(axis * math.sin(angle/2), math.cos(angle/2)))
            
            bboxes.append(BoundingBox3D(
                center=tuple(center),
                size=size,
                rotation=rotation,
                color=color,
                label=label
            ))
        
        yield bboxes

if __name__ == "__main__":
    visualizer = CombinedVisualizer()
    
    camera_pose = np.array([
        [1, 0,  0, 0],  # 相机朝Z轴正方向
        [0, 0, -1, 0],  # Y轴朝下（计算机视觉惯例）
        [0, 1,  0, 2],  # 相机高度2米
        [0, 0,  0, 1]
    ])
    try:
        # 生成2D和3D测试数据
        img_gen = generate_image_stream()
        bbox2d_gen = generate_2d_bboxes()
        bbox3d_gen = generate_3d_bboxes(camera_pose=camera_pose)
        
        # 模拟实时图像流处理
        for idx, (img, bboxes_2d, bboxes_3d) in enumerate(zip(img_gen, bbox2d_gen, bbox3d_gen)):
            # 为2D bbox生成随机颜色和标签
            colors = [tuple(np.random.randint(0, 256, 3).tolist()) for _ in bboxes_2d]
            labels = [f"obj_{i}" for i in range(len(bboxes_2d))]
            
            # 显示原始图像
            visualizer.stream_image(img, frame_idx=idx)
            
            # 显示带2D边界框的图像
            visualizer.visualize_with_2d_bboxes(
                img,
                bboxes_2d,
                frame_idx=idx,
                class_labels=labels,
                bbox_colors=colors
            )
            
            # 独立记录2D边界框参数信息
            visualizer.log_bbox_info(
                bboxes_2d,
                frame_idx=idx,
                class_labels=labels,
                bbox_colors=colors
            )
            
            # 显示带3D边界框的图像
            visualizer.visualize_with_3d_bboxes(
                img,
                bboxes_3d,
                frame_idx=idx,
                draw_projection=True
            )
            
            # 模拟实时处理间隔
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("用户中断")
    finally:
        visualizer.close()
        print("可视化服务已关闭")
