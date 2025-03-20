
def iou(box1, box2):
    """
    计算两个边界框的 IoU（交并比）
    box1, box2 格式均为 (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

class SimpleTracker:
    """
    一个简单的追踪器：
      - 输入每帧的检测结果（格式：((x1, y1, w, h), score, class_id)）。
      - 内部将检测框转换为 (x1, y1, x2, y2) 格式。
      - 通过 IoU 进行简单匹配，若检测框与现有轨迹的 IoU 大于阈值（且类别相同），则认为是同一目标并更新轨迹；
        否则创建新的轨迹。
      - 对于连续多帧未匹配的轨迹（missed 超过 max_age），则将其删除。
    """
    def __init__(self, iou_threshold=0.3, max_age=5):
        """
        初始化 SimpleTracker。
        参数:
            iou_threshold (float): 检测框匹配的 IoU 阈值。
            max_age (int): 连续 max_age 帧未匹配则删除该轨迹。
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = []  # 存储活动轨迹，每个轨迹为字典，包含：
                          # 'box': (x1, y1, x2, y2)
                          # 'score': 当前帧检测得分
                          # 'class_id': 目标类别
                          # 'track_id': 分配的持续追踪 ID
                          # 'age': 连续匹配帧数（可用于评估稳定性）
                          # 'missed': 连续未匹配帧数
        self.next_track_id = 0

    def update(self, detections):
        """
        更新追踪器，关联当前帧的检测与历史轨迹。
        参数:
            detections (list): 每个元素为 ((x1, y1, w, h), score, class_id)
                              其中 (x1, y1, w, h) 为检测框（左上角坐标及宽高）。

        返回:
            results (list): 每个元素为字典，包含键：
                - 'box': (x1, y1, w, h)
                - 'score': 检测得分
                - 'class_id': 目标类别
                - 'track_id': 持续追踪的 ID
        """
        # 将 detections 转换为统一格式 (x1, y1, x2, y2)
        dets = []
        for det in detections:
            box, score, class_id = det
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h
            dets.append({'box': (x1, y1, x2, y2), 'score': score, 'class_id': class_id})

        # 标记哪些检测已经被匹配
        assigned_dets = set()
        updated_tracks = []

        # 尝试将已有轨迹与当前检测进行匹配（仅匹配同一类别的目标）
        for track in self.tracks:
            best_iou = 0
            best_det_idx = -1
            for idx, det in enumerate(dets):
                if idx in assigned_dets:
                    continue
                if det['class_id'] != track['class_id']:
                    continue
                current_iou = iou(track['box'], det['box'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_det_idx = idx
            if best_iou >= self.iou_threshold and best_det_idx != -1:
                # 认为匹配成功，更新轨迹
                det = dets[best_det_idx]
                track['box'] = det['box']
                track['score'] = det['score']
                track['age'] += 1
                track['missed'] = 0
                updated_tracks.append(track)
                assigned_dets.add(best_det_idx)
            else:
                # 未匹配上，则增加未匹配计数
                track['missed'] += 1
                if track['missed'] <= self.max_age:
                    updated_tracks.append(track)
                # 超过 max_age 的轨迹将被舍弃

        # 对于没有匹配到已有轨迹的检测，创建新轨迹
        for idx, det in enumerate(dets):
            if idx not in assigned_dets:
                new_track = {
                    'box': det['box'],
                    'score': det['score'],
                    'class_id': det['class_id'],
                    'track_id': self.next_track_id,
                    'age': 1,
                    'missed': 0
                }
                self.next_track_id += 1
                updated_tracks.append(new_track)

        self.tracks = updated_tracks

        # 将内部 box 格式转换回 (x1, y1, w, h) 格式返回
        results = []
        for track in self.tracks:
            x1, y1, x2, y2 = track['box']
            w = x2 - x1
            h = y2 - y1
            results.append({
                'box': (x1, y1, w, h),
                'score': track['score'],
                'class_id': track['class_id'],
                'track_id': track['track_id']
            })
        return results

if __name__ == "__main__":
    # 可以在这里添加测试代码来验证 SimpleTracker 的功能
    tracker = SimpleTracker(iou_threshold=0.3, max_age=3)
    # 模拟第一帧的检测结果
    dets_frame1 = [
        ((50, 50, 100, 150), 0.9, 0),
        ((200, 80, 120, 160), 0.85, 1)
    ]
    tracks1 = tracker.update(dets_frame1)
    print("Frame 1 tracks:", tracks1)

    # 模拟第二帧，目标略有移动
    dets_frame2 = [
        ((55, 55, 100, 150), 0.88, 0),
        ((205, 85, 120, 160), 0.80, 1)
    ]
    tracks2 = tracker.update(dets_frame2)
    print("Frame 2 tracks:", tracks2)

