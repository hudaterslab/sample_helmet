#!/usr/bin/env python3
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import torchvision
from ultralytics.utils import ops

# [라이브러리 로드]
try:
    from dx_engine import InferenceEngine, InferenceOption
except ImportError:
    print("Error: 'dx_engine' library not found.")
    sys.exit(1)

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
except ImportError:
    print("Error: PyGObject not found.")
    sys.exit(1)

Gst.init(None)

# ==========================================
# 0. Global Settings & Skeleton
# ==========================================
SKELETON = [
    [0,1],[0,2],[1,3],[2,4],[0,5],[0,6],[5,7],[7,9],
    [6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

# ==========================================
# 1. GStreamer Classes
# ==========================================
class GStreamerCapture:
    def __init__(self, source, req_width=1920, req_height=1080, fps=30):
        # 웹캠 요청용 해상도 (RTSP는 무시됨)
        self.req_width = req_width
        self.req_height = req_height
        
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # Webcam: 요청 해상도로 시도
            device_id = int(source)
            cmd = (
                f"v4l2src device=/dev/video{device_id} ! "
                f"image/jpeg,width={req_width},height={req_height},framerate={fps}/1 ! "
                "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=True max-buffers=1 drop=True sync=False"
            )
            print(f"✅ Camera Mode: /dev/video{device_id}")
        else:
            # RTSP: 원본 해상도 그대로 수신 (latency=0 필수)
            cmd = (
                f"rtspsrc location={source} latency=0 ! "
                "decodebin ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=True max-buffers=1 drop=True sync=False"
            )
            print(f"✅ Stream Mode: {source}")

        self.pipeline = Gst.parse_launch(cmd)
        self.sink = self.pipeline.get_by_name('sink')
        if not self.sink:
            print("❌ Failed to create pipeline sink")
            sys.exit(1)

    def start(self): 
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("❌ Error: Unable to set the pipeline to the playing state.")
            sys.exit(1)
    
    def read(self):
        sample = self.sink.emit("pull-sample")
        if not sample: return False, None
        
        buf = sample.get_buffer()
        # [중요] 매 프레임마다 캡슐에서 해상도 확인 (RTSP는 가변적일 수 있음)
        caps = sample.get_caps() 
        structure = caps.get_structure(0)
        h = structure.get_value("height")
        w = structure.get_value("width")
        
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success: return False, None
        
        frame = np.ndarray(shape=(h, w, 3), dtype=np.uint8, buffer=map_info.data)
        # 메모리 복사 (GStreamer 버퍼 해제를 위해 필수)
        frame_copy = frame.copy()
        buf.unmap(map_info)
        return True, frame_copy

    def release(self): self.pipeline.set_state(Gst.State.NULL)

class GStreamerDisplay:
    def __init__(self, width, height, fps=30):
        # 입력된 크기에 딱 맞는 윈도우 생성
        self.width = width
        self.height = height
        cmd = (
            f"appsrc name=source format=3 ! "
            f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
            f"videoconvert ! autovideosink sync=False"
        )
        print(f"✅ Output Pipeline Created: {width}x{height}")
        self.pipeline = Gst.parse_launch(cmd)
        self.source = self.pipeline.get_by_name('source')
        if not self.source: sys.exit(1)

    def start(self): self.pipeline.set_state(Gst.State.PLAYING)
    
    def show(self, frame):
        data = frame.tobytes()
        buf = Gst.Buffer.new_wrapped(data)
        self.source.emit("push-buffer", buf)
        
    def release(self): self.pipeline.set_state(Gst.State.NULL)

# ==========================================
# 2. AI Helper Functions (No Change)
# ==========================================
class YoloConfig:
    def __init__(self, model_info):
        self.model_path = model_info["model"]["path"]
        self.params = model_info["model"]["param"]
        self.classes = model_info["output"]["classes"]
        self.score_threshold = self.params.get("score_threshold", 0.75)
        self.iou_threshold = self.params.get("iou_threshold", 0.45)
        self.input_size = (self.params.get("input_size", 640), self.params.get("input_size", 640))
        self.num_keypoints = self.params.get("kpt_count", 17)

def preprocess_image(image_src, new_shape=(640, 640), fill_color=(114, 114, 114)):
    shape = image_src.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad: image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_src = cv2.copyMakeBorder(image_src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)
    return cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB), ratio, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else: gain = ratio_pad[0][0]; pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]; coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1]); coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1]); coords[:, 3].clamp_(0, img0_shape[0])
    return coords

def postprocess_pose(output, yolo_config, scale, pad, orig_shape):
    if isinstance(output, list): output = output[0]
    if hasattr(output, 'cpu'): dets = output.cpu().numpy().squeeze(0)
    else: dets = output.squeeze(0)
    
    scores = dets[:, 4]
    mask = scores > yolo_config.score_threshold
    dets = dets[mask]; scores = scores[mask]
    
    if len(dets) == 0: return np.array([]), np.array([]), np.array([])

    boxes_xywh = dets[:, :4]
    boxes_xyxy = np.copy(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    
    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), scores.tolist(), yolo_config.score_threshold, yolo_config.iou_threshold)
    if len(indices) == 0: return np.array([]), np.array([]), np.array([])
        
    final_dets = dets[indices.flatten()]
    final_scores = scores[indices.flatten()]
    
    keypoints_raw = final_dets[:, 5:5 + yolo_config.num_keypoints * 3]
    keypoints = keypoints_raw.reshape(-1, yolo_config.num_keypoints, 3)
    keypoints = keypoints[:, :, [1, 2, 0]] 

    pad_w, pad_h = pad
    keypoints[:, :, 0] -= pad_w; keypoints[:, :, 0] /= scale
    keypoints[:, :, 1] -= pad_h; keypoints[:, :, 1] /= scale

    H, W = orig_shape[:2]
    keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, W - 1)
    keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, H - 1)
    
    keypoints = keypoints[:, :, [0, 1, 2]]
    labels = np.zeros((keypoints.shape[0],), dtype=np.int32)
    return keypoints, final_scores, labels

def postprocess_detect(output, config: YoloConfig):
    prediction = torch.from_numpy(output[0])
    if prediction.shape[2] > prediction.shape[1]: prediction = prediction.transpose(1, 2)
    prediction = prediction[0]
    box = ops.xywh2xyxy(prediction[:, :4])
    scores, class_ids = torch.max(prediction[:, 4:], 1, keepdim=True)
    mask = scores.flatten() > config.score_threshold
    box = box[mask]; scores = scores[mask]; class_ids = class_ids[mask]
    if box.shape[0] == 0: return torch.empty((0, 4)), torch.empty(0), torch.empty(0)
    indices = torchvision.ops.nms(box, scores.flatten(), config.iou_threshold)
    return box[indices], scores[indices], class_ids[indices]

# ==========================================
# 3. Dynamic Layout Manager (Adaptive)
# ==========================================
class LayoutManager:
    def __init__(self, input_w, input_h):
        # 입력 영상 크기 저장 (Main Area)
        self.main_w = input_w
        self.main_h = input_h
        
        # Sidebar 너비 설정 (입력 높이에 비례하거나 고정값)
        # 예: 1080p 일때 약 480px 정도가 적당, 비율로 25% 설정
        self.sidebar_w = int(self.main_w * 0.3) 
        if self.sidebar_w < 300: self.sidebar_w = 300 # 최소 너비 보장

        # 전체 캔버스 크기 계산
        self.out_w = self.main_w + self.sidebar_w
        self.out_h = self.main_h
        
        print(f"✅ Layout Initialized: Canvas {self.out_w}x{self.out_h} (Video {self.main_w}x{self.main_h})")
        
        self.snapshot_img = None

    def update_snapshot(self, frame):
        self.snapshot_img = frame.copy()

    def compose(self, live_frame):
        # 1. 빈 캔버스 생성
        canvas = np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        
        # 2. [최적화] 영상 리사이즈 없이 바로 복사 (Zero-Copy Resizing)
        # 입력 영상 크기가 초기화 시점과 같다면 그냥 붙여넣기만 하면 됨
        h, w = live_frame.shape[:2]
        if h == self.main_h and w == self.main_w:
            canvas[0:h, 0:w] = live_frame
        else:
            # 만약 RTSP 해상도가 도중에 바뀌었다면 안전하게 리사이즈 (드문 경우)
            live_resized = cv2.resize(live_frame, (self.main_w, self.main_h))
            canvas[0:self.main_h, 0:self.main_w] = live_resized
        
        # 구분선
        cv2.line(canvas, (self.main_w, 0), (self.main_w, self.out_h), (50, 50, 50), 2)

        # 3. 사이드바 그리기
        sidebar_x = self.main_w
        cv2.rectangle(canvas, (sidebar_x, 0), (self.out_w, self.out_h), (30, 30, 30), -1)
        
        font_scale = max(0.6, self.out_h / 1500.0) # 해상도에 따른 폰트 크기 조절
        cv2.putText(canvas, "SAFETY LOG", (sidebar_x + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, (255, 255, 255), 2)
        cv2.putText(canvas, "Last Event:", (sidebar_x + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

        # 4. 스냅샷 배치
        if self.snapshot_img is not None:
            # 사이드바 너비에 맞춰 스냅샷 축소
            disp_w = self.sidebar_w - 40
            disp_h = int(disp_w * (self.main_h / self.main_w)) # 원본 비율 유지
            
            snap_resized = cv2.resize(self.snapshot_img, (disp_w, disp_h))
            y_pos = 140
            canvas[y_pos : y_pos + disp_h, sidebar_x + 20 : sidebar_x + 20 + disp_w] = snap_resized
            
            cv2.rectangle(canvas, (sidebar_x + 20, y_pos), (sidebar_x + 20 + disp_w, y_pos + disp_h), (0, 255, 0), 2)
            cv2.putText(canvas, "Captured!", (sidebar_x + 20, y_pos + disp_h + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        else:
            cv2.putText(canvas, "Waiting...", (sidebar_x + 50, 200), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 100), 2)

        return canvas

# ==========================================
# 4. Main Logic
# ==========================================
def run_system(source_input):
    det_cfg = YoloConfig({
        "model": { "path": "helmet_3cls_v8.dxnn", "param": { "score_threshold": 0.8, "iou_threshold": 0.3, "input_size": 640 } },
        "output": { "classes": ["helmet", "head", "person"] }
    })
    pose_cfg = YoloConfig({
        "model": { "path": "YOLOV5Pose640_1.dxnn", "param": { "score_threshold": 0.5, "iou_threshold": 0.5, "input_size": 640, "kpt_count": 17 } },
        "output": { "classes": ["person"] }
    })

    print(f"Loading Models... Input Source: {source_input}")
    io = InferenceOption()
    io.set_use_ort(True)
    ie_det = InferenceEngine(det_cfg.model_path, io)
    ie_pose = InferenceEngine(pose_cfg.model_path, io)
    
    # 캡처 시작 (해상도 정보는 아직 모름, 첫 프레임에서 확인)
    cap = GStreamerCapture(source=source_input)
    cap.start()
    
    # [변경] Display 및 Layout은 첫 프레임 수신 후 초기화 (Lazy Init)
    display = None
    layout = None

    print("🚀 Loop Starting...")
    
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    
    TARGET_SAFETY_TIME = 3.0
    safety_timer = 0.0
    prev_event_triggered = False 
    
    try:
        frame_idx = 0
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if frame_idx == 0: time.sleep(1); continue
                continue
            
            # --- [New] 첫 프레임에서 해상도 감지 및 Display 초기화 ---
            if display is None:
                h, w = frame.shape[:2]
                print(f"🎥 Detected Input Resolution: {w}x{h}")
                
                # 입력 해상도에 맞춰 Layout 생성
                layout = LayoutManager(input_w=w, input_h=h)
                
                # 계산된 전체 Canvas 크기대로 Display 생성
                display = GStreamerDisplay(width=layout.out_w, height=layout.out_h)
                display.start()

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            fps = 1 / dt if dt > 0 else 0
            
            # --- AI Processing ---
            img_input, ratio, padding = preprocess_image(frame, det_cfg.input_size[0])
            
            out_det = ie_det.run([img_input])
            boxes_det, scores_det, cids_det = postprocess_detect(out_det, det_cfg)
            
            out_pose = ie_pose.run([img_input])
            final_kpts_pose, scores_pose, _ = postprocess_pose(out_pose, pose_cfg, ratio[0], padding, frame.shape)
            
            vis_frame = frame.copy()
            
            # --- Draw Results ---
            final_boxes_det = []
            if len(boxes_det) > 0:
                final_boxes_det = scale_coords(img_input.shape[:2], boxes_det, vis_frame.shape, (ratio, padding))

            for kpt_person in final_kpts_pose:
                for px, py, conf in kpt_person:
                    if conf > 0.5: cv2.circle(vis_frame, (int(px), int(py)), 3, GREEN, -1)
                for s in SKELETON:
                    p1, p2 = s[0], s[1]
                    if kpt_person[p1][2] > 0.5 and kpt_person[p2][2] > 0.5:
                        cv2.line(vis_frame, (int(kpt_person[p1][0]), int(kpt_person[p1][1])), 
                                            (int(kpt_person[p2][0]), int(kpt_person[p2][1])), YELLOW, 2)

            is_currently_safe = False
            def is_inside_expanded(pt, box):
                x1, y1, x2, y2 = box
                h = y2 - y1
                return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2 + (h * 0.5)

            for i, d_box in enumerate(final_boxes_det):
                cls_id = int(cids_det[i].item())
                label = det_cfg.classes[cls_id]
                if "helmet" in label or "안전모" in label:
                    for kpt_person in final_kpts_pose:
                        head_points_safe = False
                        for head_idx in [0, 1, 2, 3, 4]: 
                            px, py, conf = kpt_person[head_idx]
                            if conf > 0.5 and is_inside_expanded((px, py), d_box):
                                head_points_safe = True; break
                        if head_points_safe:
                            is_currently_safe = True
                            x1, y1, x2, y2 = map(int, d_box)
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), GREEN, 3)
                            break
                    if is_currently_safe: break

            if is_currently_safe: safety_timer += dt
            else: safety_timer = 0.0

            event_triggered = safety_timer >= TARGET_SAFETY_TIME
            if safety_timer > TARGET_SAFETY_TIME + 1.0: safety_timer = TARGET_SAFETY_TIME + 1.0

            if event_triggered and not prev_event_triggered:
                print("📸 Snapshot Taken!")
                layout.update_snapshot(vis_frame)
            
            prev_event_triggered = event_triggered 

            for i, box in enumerate(final_boxes_det):
                x1, y1, x2, y2 = map(int, box)
                cls_id, score = int(cids_det[i].item()), scores_det[i].item()
                label = det_cfg.classes[cls_id]
                color = WHITE
                if "helmet" in label:
                     if not is_currently_safe: color = RED
                     else: continue
                elif "head" in label: color = BLUE
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if event_triggered:
                cv2.putText(vis_frame, "GOOD SAFETY", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, GREEN, 4)
            elif safety_timer > 0.1:
                bar_width = int((safety_timer / TARGET_SAFETY_TIME) * vis_frame.shape[1])
                cv2.rectangle(vis_frame, (0, vis_frame.shape[0]-20), (bar_width, vis_frame.shape[0]), YELLOW, -1)

            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # --- Final Composition ---
            final_output = layout.compose(vis_frame)
            display.show(final_output)
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"FPS: {fps:.1f} | Timer: {safety_timer:.1f}s | Event: {event_triggered}", end='\r')

    except KeyboardInterrupt:
        print("\nStopped by User")
    finally:
        if cap: cap.release()
        if display: display.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safety Helmet AI System")
    parser.add_argument('--input', type=str, default='rtsp://admin:Hu924688@192.168.10.64:554/Streaming/Channels/101', 
                        help="Input source: Device ID (e.g., '0') or RTSP URL / File path")
    args = parser.parse_args()
    
    run_system(args.input)
