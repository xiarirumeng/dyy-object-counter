'''
使用pipline加速物体识别，整合识别和跟踪信息接入物体计数部分
'''
import cv2
import numpy as np
import yaml
from pathlib import Path
import re
from tflite_runtime import interpreter as tflite
import argparse
from results_counter import ObjectCounter
from results_track import SimpleTracker
import time

# 定义全局变量：模型输入图片的目标宽度和高度（可根据训练模型进行调整）
img_width = 416
img_height = 416

def yaml_load(file="data.yaml", append_filename=False):
    """
    从 YAML 文件中加载数据
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
        data = yaml.safe_load(s) or {}
        if append_filename:
            data["yaml_file"] = str(file)
        return data

class LetterBox:
    def __init__(self, new_shape=(img_width, img_height), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))
        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

class TFLiteTracker:
    def __init__(self, tflite_model, source, confidence_thres, iou_thres, ext_delegate):
        """
        初始化 TFLiteTracker 类，并创建 TFLite 解释器（仅创建一次）
        """
        self.tflite_model = tflite_model
        self.source=source
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        if ext_delegate is not None:
            print('Loading external delegate from {} with args: {}'.format(
                ext_delegate, self.ext_delegate_options))
            self.ext_delegate = [
                tflite.load_delegate(ext_delegate, self.ext_delegate_options)
            ]

        # 从 COCO 数据集的配置文件中加载类别名称
        self.classes = yaml_load("coco8.yaml")["names"]

        # 创建并初始化 TFLite 解释器，仅执行一次
        self.interpreter = tflite.Interpreter(model_path=self.tflite_model)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        input_shape = self.input_details[0]["shape"]
        self.input_width = input_shape[1]
        self.input_height = input_shape[2]
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        在图像上绘制检测框和标签
        """
        x1, y1, w, h = box
        # 根据类别索引获取颜色
        color = self.color_palette[class_id]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        label = f"{self.classes[class_id]}: {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        cv2.rectangle(img, (int(label_x), int(label_y - label_height)),
                      (int(label_x + label_width), int(label_y + label_height)), color, cv2.FILLED)
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img):
        """
        对输入图片进行预处理，转换为模型需要的输入格式
        """
        self.img_height, self.img_width = img.shape[:2]
        letterbox = LetterBox(new_shape=[img_width, img_height], auto=False, stride=32)
        image = letterbox(image=img)
        image = [image]
        image = np.stack(image)
        image = image[..., ::-1].transpose((0, 3, 1, 2))
        img_contiguous = np.ascontiguousarray(image)
        image = img_contiguous.astype(np.float32)
        return image / 255

    def run_inference(self, img):
        """
        使用已创建的 TFLite 解释器执行推理，并返回输出结果
        """
        img_data = self.preprocess(img)
        img_data = img_data.transpose((0, 2, 3, 1))
        scale, zero_point = self.input_details[0]["quantization"]
        img_data_int8 = (img_data / scale + zero_point).astype(np.int8)
        self.interpreter.set_tensor(self.input_details[0]["index"], img_data_int8)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        scale, zero_point = self.output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale
        output[:, [0, 2]] *= img_width
        output[:, [1, 3]] *= img_height
        return output

    def extract_detections(self, output):
        """
        从模型输出中解析检测结果，返回检测列表，每个检测为 ((x1, y1, w, h), score, class_id)
        """
        out = output[0]
        out = out.T
        boxes = out[..., :4]
        scores = np.max(out[..., 4:], axis=1)
        class_ids = np.argmax(out[..., 4:], axis=1)
        boxes_list = boxes.tolist()
        scores_list = scores.tolist()
        indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, self.confidence_thres, self.iou_thres)
        detections = []
        if len(indices) > 0:
            if isinstance(indices, (list, np.ndarray)):
                indices = np.array(indices).flatten()
            for i in indices:
                score = scores[i]
                if score > 0.5:
                    box = boxes[i]
                    gain = min(img_width / self.img_width, img_height / self.img_height)
                    pad = (
                        round((img_width - self.img_width * gain) / 2 - 0.1),
                        round((img_height - self.img_height * gain) / 2 - 0.1),
                    )
                    x1 = (box[0] - box[2] / 2 - pad[0]) / gain
                    y1 = (box[1] - box[3] / 2 - pad[1]) / gain
                    w = box[2] / gain
                    h = box[3] / gain
                    detections.append(((x1, y1, w, h), score, class_ids[i]))
        return detections

    def update_tracker(self, detections):
        """
        使用 SimpleTracker 对当前帧的检测结果进行跟踪，并返回跟踪结果。
        参数:
            detections: list，每个元素格式为 ((x1, y1, w, h), score, class_id)
        返回:
            track_results: list，每个元素为字典，包含 'box', 'score', 'class_id' 和 'track_id'
        """
        # 如果尚未创建 SimpleTracker 实例，则创建它
        if not hasattr(self, 'simple_tracker'):
            # 这里的 iou_threshold 和 max_age 可以根据实际情况调整
            self.simple_tracker = SimpleTracker(iou_threshold=self.iou_thres, max_age=30)

        # 调用 simple_tracker.update 进行跟踪
        track_results = self.simple_tracker.update(detections)
        return track_results

    def main(self):
        """
        主函数：读取视频流中每一帧，执行推理、提取检测信息、绘制检测框，并实时显示视频
        """
        pipeline = 'filesrc location=' + self.source + ' ! qtdemux ! h264parse ! vpudec ! videoconvert ! video/x-raw format=RGB ! appsink'
        cap = cv2.VideoCapture(pipeline)
        # cap = cv2.VideoCapture(self.source)
        # start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # # 检查是否超过30秒
            # if time.time() - start_time > 30:
            #     break

            # 为 output_image 初始化（可以选择复制原始帧）
            output_image = frame.copy()

            # 执行推理
            output = self.run_inference(frame)
            print("已推理当前帧")
            detections = self.extract_detections(output)

            # 如果检测到物体，更新追踪器并绘制检测框
            if detections:
                track_results = self.update_tracker(detections)
                print("当前帧检测到的物体及得分:")
                for det in detections:
                    box, score, class_id = det
                    print("物体类别：{}, 得分：{}".format(class_id, score))
                    self.draw_detections(output_image, box, score, class_id)
            else:
                track_results = []

            cv2.imshow("Object Recongnition", output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print("当前帧执行结束")

            # yield FakeResult(frame, track_results)
            yield FakeResult(frame.copy(), track_results)

        cap.release()
        cv2.destroyAllWindows()

class FakeTensor:
    def __init__(self, array):
        self.array = np.array(array)
    def cpu(self):
        return self
    def tolist(self):
        return self.array.tolist()
    def int(self):
        return self
class FakeBoxes:
    def __init__(self, track_results):
        xyxy_list = []
        cls_list = []
        id_list = []
        # track_results 是一个列表，每个元素是字典，格式：{"box": (x, y, w, h), "score": score, "class_id": class_id, "track_id": track_id}
        for track in track_results:
            x, y, w, h = track["box"]
            xyxy_list.append((x, y, x + w, y + h))
            cls_list.append(track["class_id"])
            id_list.append(track["track_id"])
        self.xyxy = FakeTensor(xyxy_list)
        self.cls = FakeTensor(cls_list)
        self.id = FakeTensor(id_list)
class FakeResult:
    def __init__(self, orig_img, track_results):
        self.orig_img = orig_img
        self.boxes = FakeBoxes(track_results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="yolov8n_full_integer_quant.tflite", help="Input your TFLite model."
    )
    parser.add_argument("--video", type=str, default="./test.mp4", help="Path to input video.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("-e", "--ext_delegate", help='external_delegate_library path')
    args = parser.parse_args()

    tracker = TFLiteTracker(args.model, args.video, args.conf_thres, args.iou_thres, args.ext_delegate)

    # 初始化计数器
    counter = ObjectCounter()
    # 配置计数器
    counter.set_args(
        classes_names=yaml_load("coco8.yaml")["names"],
        reg_pts=[(20, 400), (1260, 400)],
        # reg_pts=[(150, 400), (150, 200), (600, 200), (600, 400)],
        view_img=True,
        draw_tracks=True,  # 显示轨迹
        count_reg_color=(255, 0, 255),  # 计数线颜色
        line_thickness=2,
    )

    # 逐帧计数并显示
    for result in tracker.main():
        # counter.start_counting(result.orig_img, [result])
        counter.start_counting(result.orig_img.copy(), [result])

    # 循环结束后打印最终计数结果
    print("最终计数结果：")
    print("总计进入计数 (in_counts):", counter.in_counts)
    print("总计离开计数 (out_counts):", counter.out_counts)
    print("各类别计数详情：")
    for class_name, counts in counter.class_wise_count.items():
        print(f"{class_name} : 进入 {counts['in']}，离开 {counts['out']}")

if __name__ == "__main__":
    main()