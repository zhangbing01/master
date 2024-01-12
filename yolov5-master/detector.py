import cv2
import os
import time
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_segments
from models.common import DetectMultiBackend
import torch
import numpy as np
import multiprocessing as mp


class Detector:
    # def __init__(self, cuda_subid, param_dl, offline, log) -> None:
    #     self.log = log
    #     # weights, device_id, imgsz, conf_thres, iou_thres
    #     if offline:
    #         deviceID = param_dl.deviceID4Video[cuda_subid]
    #     else: # online stream
    #         deviceID = param_dl.deviceID4Stream[cuda_subid]
    #     # init detection
    #     self.device = torch.device("cuda:"+str(deviceID) if torch.cuda.is_available() else "cpu")
    #     self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
    #     # Load model
    #     self.imgsz = param_dl.imgsz
    #     if cuda_subid == 2:
    #         weight = param_dl.weight_headset
    #     else:
    #         weight = param_dl.weight
    #     self.model = DetectMultiBackend(weight, device=self.device, fp16=self.half)
    #     self.imgsz = check_img_size(param_dl.imgsz, s = self.model.stride)  # check img_size
    #     self.model.eval()
    #     self.conf_thres = param_dl.confidence_threshold # # object confidence threshold
    #     if cuda_subid == 2:
    #         self.conf_thres[0] = 0.45
    #     self.iou_thres = param_dl.iou_threshold # IOU threshold for NMS
    #
    #     self.log.warning('[detector] class Detector initialized')
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != 'cpu'
        self.model = DetectMultiBackend("./weights/yolov5s.pt", device=self.device, fp16=self.half)
        self.model.eval()
        self.imgsz = check_img_size(640, s=self.model.stride)  # check img_size
        self.conf_thres = 0.35
        # confTH = [0.35, 0.65, 0.45, 0.45, 0.15, 0.35, 0.85, 0.55]
        self.iou_thres = 0.45

    def transform_image(self, image):
        # if image is None:
        #     self.log.error('[detector] input image for transformation is None')
        #     return
        h0 = image.shape[0]
        w0 = image.shape[1]
        img, ratio, (dw, dh) = letterbox(image, new_shape=self.imgsz, auto=False)
        shapes = [(h0, w0), (ratio, (dw, dh))]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img[np.newaxis, :]
        return img, shapes

    def to_GPU(self, trans_img):
        trans_img = torch.from_numpy(trans_img).to(self.device)
        trans_img = trans_img.half() if self.model.fp16 else trans_img.float()  # uint8 to fp16/32
        trans_img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(trans_img.shape) == 3:
            trans_img = trans_img[None]
        return trans_img

    def detect(self, trans_img, trans_shapes):

        # Run inference
        # trans_img, shapes = self.transform_image(detect_img)

        # trans_end_time = time.time()
        # delta_trans = trans_end_time - start_time

        # trans_img = torch.from_numpy(trans_img).to(self.device)
        # trans_img = trans_img.half() if self.half else trans_img.float()  # uint8 to fp16/32
        # trans_img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if trans_img.ndimension() == 3:
        #     trans_img = trans_img.unsqueeze(0)
        # Inference
        pred = self.model(trans_img, augment=False, visualize=False)

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

        xyxy = []
        for k, det in enumerate(pred):  # detections per image

            nobj = len(det)
            if nobj:
                # Rescale boxes from img_size to im0 size
                det = scale_segments(trans_img.shape[2:], det, trans_shapes[k][0], trans_shapes[k][1])
                det[:, :4] = det[:, :4].round()

                det_cpu_xyxy = np.array(det.detach().cpu(), dtype=np.double)
                xyxy.append(det_cpu_xyxy)
            else:
                xyxy.append(None)

        return xyxy


# if __name__ == '__main__':
#     from core.DayLogger import DayLogger
#     log = DayLogger('main').get_logger()
#     config_file_path = 'config/config.cfg'
#     g_supp = Supportor(log)
#     param = g_supp.get_system_parameters(config_file_path)
#     d = Detector(0,param[1])
#     img = cv2.imread("4819.jpg")
#     tmp_img = d.transform_image(img)
#     trans_img_gpu = d.to_GPU(tmp_img)
#     det_list_xyxy = d.detect(trans_img_gpu,tmp_img)
#     print(det_list_xyxy)
def write(q, cam) -> None:
    """
    :q: Manager.list对象
    :param cam: 摄像头参数
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    while True:
        _, img = cap.read()
        if _:
            # 修改分辨率
            img = cv2.resize(img, (640, 480))
            # 读取视频帧并将其放入队列中
            q.put(img)
            # 如果队列中的帧数超过阈值，移除最早的帧
            q.get() if q.qsize() > 1 else time.sleep(0.01)


def read(q, wind_name) -> None:
    """
    :q: Manager.list对象
    :param wind_name:wind窗口名称
    :return: None
    """
    # 创建特征检测对象
    d = Detector()
    # 对视频数据进行处理
    print('Process to read: %s' % os.getpid())
    while True:
        # 获取视频帧
        frame = q.get()
        tmp_img, tmp_shaps = d.transform_image(frame)
        trans_img_gpu = d.to_GPU(tmp_img)
        det_list_xyxy = d.detect(trans_img_gpu, [tmp_shaps])
        for xyxy in det_list_xyxy:
            if xyxy is not None:
                for x1y1x2y2 in xyxy:
                    # print(x1y1x2y2)
                    if x1y1x2y2[5] == 0:
                        x1 = x1y1x2y2[0].astype("int")
                        y1 = x1y1x2y2[1].astype("int")
                        x2 = x1y1x2y2[2].astype("int")
                        y2 = x1y1x2y2[3].astype("int")
                        # 步骤一：获取交集部分坐标
                        ix_min = max(x1, 60)  # 两个 x_left 中[靠右]的那个
                        iy_min = max(y1, 30)  # 两个 y_top 中[靠下]的那个
                        ix_max = min(x2, 580)  # 两个 x_right 中[靠左]的那个
                        iy_max = min(y2, 450)  # 两个 y_bottom 中[靠上]的那个

                        # 步骤二：计算交集部分的宽和高
                        # 注：可能出现宽度或高度为负数的情况，此时两个框没有交集，交并比为0
                        iw = max(ix_max - ix_min, 0.0)
                        ih = max(iy_max - iy_min, 0.0)

                        # 步骤三：计算交集部分面积
                        inters = iw * ih
                        rate = inters / ((580 - 60) * (450 - 30))
                        print(rate)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("human", frame)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    mp.set_start_method(method='spawn', force=True)
    queue = mp.Queue()
    processes = []
    processes.append(mp.Process(target=write, args=(
        queue, "rtsp://admin:admin123@192.168.12.200:554/cam/realmonitor?channel=1&subtype=0")))
    processes.append(mp.Process(target=read, args=(
        queue, "rtsp://admin:admin123@192.168.12.200:554/cam/realmonitor?channel=1&subtype=0")))
    for process in processes:
        process.daemon = False
        process.start()
    for process in processes:
        process.join()



