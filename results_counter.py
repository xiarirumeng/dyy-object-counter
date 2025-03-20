'''
自定义了check_imshow、colors、Annotator函数或类
使用 OpenCV 自带的几何函数或简单的数学公式来完成「判断点是否在多边形内」和「计算点到线段的距离」等功能
'''
from collections import defaultdict
import cv2
import sys
import matplotlib
import numpy as np
from PIL import Image
import math
def distance_point_to_line(point, line_pts):
    """
    计算 point 到由 line_pts[0], line_pts[1] 这两个点所定义线段的最短距离
    point: (x, y)
    line_pts: [(x1, y1), (x2, y2)]
    """
    (x, y) = point
    (x1, y1), (x2, y2) = line_pts

    # 若线段退化成一个点，直接返回两点距离
    if (x1 == x2) and (y1 == y2):
        return math.hypot(x - x1, y - y1)

    # 线段向量
    line_vec = (x2 - x1, y2 - y1)
    # 点相对线段起点的向量
    pt_vec = (x - x1, y - y1)

    # 线段长度的平方
    line_len_sq = line_vec[0]**2 + line_vec[1]**2
    # 投影系数 t
    t = (pt_vec[0]*line_vec[0] + pt_vec[1]*line_vec[1]) / line_len_sq

    # 如果投影超出线段范围，则截断到 [0,1]
    t = max(0, min(1, t))

    # 计算线段上最靠近该点的投影坐标
    proj_x = x1 + t * line_vec[0]
    proj_y = y1 + t * line_vec[1]

    # 返回 point 到投影点的距离
    return math.hypot(x - proj_x, y - proj_y)
def compute_centroid(pts):
    """
    根据给定点列表(可以是线段2点，也可以是多边形>=3点)计算区域质心
    pts: [(x1, y1), (x2, y2), ...]
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    return (cx, cy)
def is_point_in_polygon(point, polygon_pts):
    """
    利用 OpenCV 的 pointPolygonTest 判断点是否在多边形内部
    point: (x, y)
    polygon_pts: numpy.ndarray, shape=(N, 2), int32
    返回 True 表示在多边形内或边界上
    """
    # 当 measureDist=False 时：
    # 返回值 > 0 表示在多边形内
    # = 0 表示在多边形边界上
    # < 0 表示在多边形外
    result = cv2.pointPolygonTest(polygon_pts, point, False)
    return result >= 0
def colors(i, bgr=False):
    """
    为目标跟踪生成独特的颜色
    """
    hex_colors = [
        '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A', '#92CC17',
        '#3DDB86', '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF', '#344593', '#6473FF',
        '#0018EC', '#8438FF', '#520085', '#CB38FF', '#FF95C8', '#FF37C7'
    ]
    def hex2rgb(hex_str):
        h = hex_str.lstrip('#')
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    color_idx = i % len(hex_colors)
    rgb_color = hex2rgb(hex_colors[color_idx])
    if bgr:
        return (rgb_color[2], rgb_color[1], rgb_color[0])
    return rgb_color
def check_imshow(warn=False):
    """
    自定义检查图像显示功能的函数。
    """
    try:
        if sys.platform == 'win32' or sys.platform == 'darwin' or (
                sys.platform == 'linux' and matplotlib.get_backend() != 'Agg'):
            return True
        else:
            raise EnvironmentError("当前平台不支持图像显示。")
    except Exception as e:
        if warn:
            print(f"警告: 图像显示功能无法正常工作，错误信息：{e}")
        return False
class Annotator:
    tf = 1
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, names=None):
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
        self.font_size = font_size or max(round(sum(im.shape) / 2 * 0.035), 12)
        self.pil = pil
        self.names = names

        if self.pil:
            self.im = Image.fromarray(self.im)

        self.default_color = (255, 255, 255)
        self.default_txt_color = (255, 255, 255)

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=max(int(self.lw), 1), lineType=cv2.LINE_AA)

        if label:
            tf = max(self.lw - 1, 1)
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(self.im,
                        label,
                        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def draw_region(self, reg_pts, color=(255, 0, 255), thickness=2):
        reg_pts = np.array(reg_pts, dtype=np.int32)
        if len(reg_pts) == 2:
            cv2.line(self.im, tuple(reg_pts[0]), tuple(reg_pts[1]), color, thickness)
        elif len(reg_pts) >= 3:
            cv2.polylines(self.im, [reg_pts], True, color, thickness)

    def draw_centroid_and_tracks(self, track_line, color=(255, 0, 0), track_thickness=2):
        points = np.array(track_line, dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(self.im, [points], False, color, track_thickness)

    def display_counts(self, counts, count_txt_color=(0, 0, 0), count_bg_color=(255, 255, 255), txt_gap=50):
        for i, count_str in enumerate(counts):
            tf = max(self.lw - 1, 1)
            w, h = cv2.getTextSize(count_str, 0, fontScale=self.lw / 3, thickness=tf)[0]
            txt_pos = (10, 30 + i * txt_gap)
            bg_pos = (txt_pos[0] + w + 5, txt_pos[1] + 5)
            cv2.rectangle(self.im,
                          (txt_pos[0] - 5, txt_pos[1] - h - 5),
                          (int(bg_pos[0]), int(bg_pos[1])),
                          count_bg_color,
                          -1)
            cv2.putText(self.im,
                        count_str,
                        txt_pos,
                        0,
                        self.lw / 3,
                        count_txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def result(self):
        return self.im

class ObjectCounter:
    def __init__(self):
        self.is_drawing = False
        self.selected_point = None
        self.reg_pts = [(20, 400), (1260, 400)]
        self.is_polygon = False         # True 表示多边形, False 表示线段
        self.polygon_pts = None         # np.array 格式
        self.line_pts = None            # [(x1, y1), (x2, y2)]
        self.region_centroid = None     # 区域质心

        self.line_dist_thresh = 15
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = None
        self.annotator = None
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = (255, 255, 255)
        self.count_bg_color = (255, 255, 255)
        self.cls_txtdisplay_gap = 50
        self.fontsize = 0.6

        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = None

        self.env_check = check_imshow(warn=True)

    def set_args(
            self,
            classes_names,
            reg_pts,
            count_reg_color=(255, 0, 255),
            count_txt_color=(0, 0, 0),
            count_bg_color=(255, 255, 255),
            line_thickness=2,
            track_thickness=2,
            view_img=False,
            view_in_counts=True,
            view_out_counts=True,
            draw_tracks=False,
            track_color=None,
            region_thickness=5,
            line_dist_thresh=15,
            cls_txtdisplay_gap=50,
    ):
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # ======== 根据传入的点数判断是线段还是多边形 ========
        if len(reg_pts) == 2:
            print("初始化线条计数器。")
            self.reg_pts = reg_pts
            self.is_polygon = False
            self.line_pts = reg_pts
        elif len(reg_pts) >= 3:
            print("初始化多边形计数器。")
            self.reg_pts = reg_pts
            self.is_polygon = True
            # OpenCV 多边形需要 np.array 格式
            self.polygon_pts = np.array(self.reg_pts, dtype=np.int32)
        else:
            print("提供的区域点无效，区域点必须是2个（线条）或>=3个（多边形）。")
            print("现在使用线条计数器")
            self.reg_pts = [(20, 400), (1260, 400)]
            self.is_polygon = False
            self.line_pts = self.reg_pts

        # 计算区域质心，后续用于判定物体移动方向
        self.region_centroid = compute_centroid(self.reg_pts)
        self.names = classes_names
        self.track_color = track_color
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        self.cls_txtdisplay_gap = cls_txtdisplay_gap

    def mouse_event_for_region(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                        isinstance(point, (tuple, list))
                        and len(point) >= 2
                        and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                # 重新计算区域质心
                self.region_centroid = compute_centroid(self.reg_pts)
                if self.is_polygon:
                    self.polygon_pts = np.array(self.reg_pts, dtype=np.int32)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(
            reg_pts=self.reg_pts,
            color=self.region_color,
            thickness=self.region_thickness
        )

        if len(tracks) > 0 and tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu().tolist()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                self.annotator.box_label(box, label=f"{self.names[cls]}#{track_id}", color=colors(int(track_id), True))

                if self.names[cls] not in self.class_wise_count:
                    # 截断一下类别名称，避免过长
                    if len(self.names[cls]) > 5:
                        self.names[cls] = self.names[cls][:5]
                    self.class_wise_count[self.names[cls]] = {"in": 0, "out": 0}

                # 更新轨迹历史 (追踪中心点)
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                track_line = self.track_history[track_id]
                track_line.append((cx, cy))
                if len(track_line) > 30:
                    track_line.pop(0)

                # 绘制轨迹
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=self.track_color if self.track_color else colors(int(track_id), True),
                        track_thickness=self.track_thickness,
                    )

                prev_position = None
                if len(self.track_history[track_id]) > 1:
                    prev_position = self.track_history[track_id][-2]

                # ========== 处理多边形区域计数 ==========
                if self.is_polygon:
                    inside_polygon = is_point_in_polygon((cx, cy), self.polygon_pts)
                    if prev_position is not None and inside_polygon and (track_id not in self.count_ids):
                        self.count_ids.append(track_id)

                        # 判断方向（基于当前中心与上一个中心，以及区域质心的简单乘积判断）
                        # 原代码: (box[0] - prev_position[0]) * (self.counting_region.centroid.x - prev_position[0]) > 0
                        # 这里用(cx - prev_x) * (region_centroid_x - prev_x) > 0
                        prev_x = prev_position[0]
                        region_cx = self.region_centroid[0]

                        if (cx - prev_x) * (region_cx - prev_x) > 0:
                            self.in_counts += 1
                            self.class_wise_count[self.names[cls]]["in"] += 1
                        else:
                            self.out_counts += 1
                            self.class_wise_count[self.names[cls]]["out"] += 1

                # ========== 处理线段区域计数 ==========
                else:
                    if prev_position is not None and (track_id not in self.count_ids):
                        dist_to_line = distance_point_to_line((cx, cy), self.line_pts)
                        if dist_to_line < self.line_dist_thresh:
                            self.count_ids.append(track_id)

                            prev_x = prev_position[0]
                            region_cx = self.region_centroid[0]

                            if (cx - prev_x) * (region_cx - prev_x) > 0:
                                self.in_counts += 1
                                self.class_wise_count[self.names[cls]]["in"] += 1
                            else:
                                self.out_counts += 1
                                self.class_wise_count[self.names[cls]]["out"] += 1

        label = "Ultralytics Analytics \t"
        for key, value in self.class_wise_count.items():
            if value["in"] != 0 or value["out"] != 0:
                if not self.view_in_counts and not self.view_out_counts:
                    label = None
                elif not self.view_in_counts:
                    label += f"{str.capitalize(key)}: IN {value['in']} \t"
                elif not self.view_out_counts:
                    label += f"{str.capitalize(key)}: OUT {value['out']} \t"
                else:
                    label += f"{str.capitalize(key)}: IN {value['in']} OUT {value['out']} \t"

        if label is not None:
            label = label.rstrip()
            label = label.split("\t")
            self.annotator.display_counts(
                counts=label,
                count_txt_color=self.count_txt_color,
                count_bg_color=self.count_bg_color,
            )

    def display_frames(self):
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if self.is_polygon:
                # 只有在多边形模式时，可能想用鼠标拖拽多边形顶点
                cv2.setMouseCallback(self.window_name, self.mouse_event_for_region, {"region_points": self.reg_pts})
            cv2.imshow(self.window_name, self.im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        self.im0 = im0
        self.extract_and_process_tracks(tracks)

        if self.view_img:
            self.display_frames()

        return self.im0

if __name__ == "__main__":
    ObjectCounter()