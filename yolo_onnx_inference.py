import os
import cv2
import numpy as np
import onnxruntime

class_names = ['dog', 'person']

input_h = 640
input_w = 640

def letterbox_resize(img, new_shape=(input_h, input_w), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 防止四舍五入边界偏差
    # dh = 4.5
    # round(dh - 0.1) = round(4.4) = 4
    # round(dh + 0.1) = round(4.6) = 5
    # 这里计算上下（left/right）边框的整数宽度
    # 直接用 int(dw) 会向下取整，导致两边padding宽度相等且总和小于实际需要的2*dw
    # 这样填充后图像尺寸可能会少1个像素，不符合模型输入的固定尺寸（如640x640）
    # 导致ONNX模型运行时出现尺寸不匹配错误（Resize节点报错）
    #
    # 因此用 round(dw - 0.1) 和 round(dw + 0.1) 分别向下和向上取整，
    # 保证上下边框的整数宽度和精确等于原始的2*dw，
    # 从而保证图像尺寸符合模型要求，避免运行时错误
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh

def bgr_to_rgb_chw(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    return img_chw

class YOLO():
    def __init__(self, onnx_path):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

    def get_input_name(self):
        return [node.name for node in self.onnx_session.get_inputs()]

    def get_output_name(self):
        return [node.name for node in self.onnx_session.get_outputs()]

    def get_input_feed(self, img_tensor):
        return {self.input_name[0]: img_tensor}

    def inference(self, img_list):
        processed_imgs = []
        original_imgs = []
        ratios = []
        pads = []

        for path in img_list:
            img = cv2.imread(path)
            or_img, r, dw, dh = letterbox_resize(img, (input_h, input_w))
            original_imgs.append(or_img.copy())
            ratios.append(r)
            pads.append((dw, dh))
            img = bgr_to_rgb_chw(or_img)
            img = img.astype(np.float32) / 255.0
            processed_imgs.append(img)

        batch_input = np.stack(processed_imgs, axis=0)
        input_feed = self.get_input_feed(batch_input)
        preds = self.onnx_session.run(None, input_feed)[0]

        print(f"ONNX model output shape: {preds.shape}")

        return preds, original_imgs, ratios, pads

# 假设当前待处理框的索引列表为：
# index = [3, 7, 5, 2]
# 其中 index[0] = 3 是置信度最高的框，先保留它。

# 我们计算 index[0]（框3）与后面所有框 index[1:] = [7, 5, 2] 的IoU：
# 比如计算得到 IoU = [0.1, 0.85, 0.2]

# 设定阈值 thresh = 0.5，保留 IoU <= 0.5 的框，
# 用 np.where(ovr <= thresh)[0] 得到的索引是相对于 index[1:] 的相对索引，
# 这里假设得到 idx = [0, 2]，表示保留相对位置为0和2的框，
# 也就是原来的框7和框2。

# 由于 idx 是基于 index[1:] 的索引，
# 为了映射回原来的 index 数组，需要将 idx + 1，
# 即 idx + 1 = [1, 3]。

# 最后用 index[idx + 1] 就得到下一轮还需要处理的框索引，
# 也就是 index[[1, 3]] = [7, 2]。

# 这样，当前框3和重叠太大的框5都被剔除，
# 只留下框7和框2继续下一轮NMS。
def nms(dets, thresh):
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1] # 所有框按置信度从高到低排序后的索引

    while index.size > 0:
        i = index[0]
        keep.append(i) # 选择当前分数最高的框的索引
        # 计算剩余框与该框的重叠区域
        # 计算交集左上角坐标
        xx1 = np.maximum(x1[i], x1[index[1:]])
        yy1 = np.maximum(y1[i], y1[index[1:]])
        # 计算交集右下角坐标
        xx2 = np.minimum(x2[i], x2[index[1:]])
        yy2 = np.minimum(y2[i], y2[index[1:]])
        # 计算交集宽度和高度
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # 计算交集面积
        inter = w * h
        # 计算交并比（IOU）
        # IOU = 交集面积 / (面积A + 面积B - 交集面积)
        ovr = inter / (areas[i] + areas[index[1:]] - inter)
        idx = np.where(ovr <= thresh)[0]
        index = index[idx + 1]
    return keep

def filter_box(output, conf_thres=0.5, iou_thres=0.5, output_format='CN'):
    """
        np.ndarray，过滤后的框，格式 [x1, y1, x2, y2, conf, cls_id]
    """
    if output_format == 'CN':
        dets = output.T  # 转为 (N, C+4)
    elif output_format == 'NC':
        dets = output
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    boxes_with_scores = []
    for det in dets:
        x_c, y_c, w, h = det[:4]
        class_scores = det[4:]
        cls_id = np.argmax(class_scores)
        conf = class_scores[cls_id]

        if conf < conf_thres:
            continue
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        
        boxes_with_scores.append([x1, y1, x2, y2, conf, int(cls_id)])

    if len(boxes_with_scores) == 0:
        return np.array([])

    boxes_with_scores = np.array(boxes_with_scores)
    detections = []
    unique_classes = np.unique(boxes_with_scores[:, 5].astype(int))

    for cls in unique_classes:
        cls_mask = boxes_with_scores[:, 5] == cls
        cls_boxes = boxes_with_scores[cls_mask]
        keep = nms(cls_boxes, iou_thres)
        detections.extend(cls_boxes[keep])

    return np.array(detections)

def scale_coords(coords, r, dw, dh, original_shape):
    coords[:, [0, 2]] -= dw
    coords[:, [1, 3]] -= dh
    coords[:, :4] /= r
    coords[:, 0] = coords[:, 0].clip(0, original_shape[1] - 1)  # x1 限制在 [0, width-1]
    coords[:, 2] = coords[:, 2].clip(0, original_shape[1] - 1)  # x2 限制在 [0, width-1]
    coords[:, 1] = coords[:, 1].clip(0, original_shape[0] - 1)  # y1 限制在 [0, height-1]
    coords[:, 3] = coords[:, 3].clip(0, original_shape[0] - 1)  # y2 限制在 [0, height-1]
    return coords
def draw(image, detections):
    boxes = detections[..., :4].astype(np.int32)
    scores = detections[..., 4]
    classes = detections[..., 5].astype(np.int32)

    for box, score, cl in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        print(f'class: {class_names[cl]}, score: {score:.2f}')
        print(f'box coordinate x1,y1,x2,y2: [{x1}, {y1}, {x2}, {y2}]')
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'{class_names[cl]} {score:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

if __name__ == "__main__":
    model = YOLO('runs/custom-yolov11s/weights/best.onnx')
    img_paths = ['test.jpg', 'test.jpg']
    outputs, letterbox_imgs, ratios, pads = model.inference(img_paths)
    # ultralytics输出格式和旧版本的不一样，ultralytics去掉了目标框置信度，把目标框置信度和类别置信度融合到一起了
    # 设置模型输出格式：
    # C表示类别格式
    # ultralytics ONNX一般是 '4+C N' (4 + C行，N列)
    # ultralytics/YOLOv5 旧项目中的yolov5是 'N C+4' (N行，C+4列)
    output_format = 'CN' if outputs.shape[1] == 4 + len(class_names) else 'NC'  
    print(outputs.shape[1], output_format)

    for i, (output, img_path, r, (dw, dh)) in enumerate(zip(outputs, img_paths, ratios, pads)):
        raw_img = cv2.imread(img_path)
        detections = filter_box(output, conf_thres=0.5, iou_thres=0.5, output_format=output_format)
        if detections.shape[0] > 0:
            detections = scale_coords(detections, r, dw, dh, raw_img.shape[:2])
            draw(raw_img, detections)
        cv2.imwrite(f'result_{i+1}.jpg', raw_img)
