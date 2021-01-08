import detectron2
import numpy as np
import os, cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog, DatasetCatalog

filename = "TFL-DK38-Vic-616C-D040-----02-07h38m30s-19.06.2020.avi"
vid_path = "/home/nikkhadijah/Data/Vic-On-Train-CCTV/"+filename
video = cv2.VideoCapture(vid_path)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
basename = os.path.basename(vid_path)

cfg = get_cfg()

# # load a weights file from online resources such as model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Inference with a panoptic segmentation model
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

predictor = DefaultPredictor(cfg)

results = []
frame_id = 0

while(video.isOpened()):
    ret, frame = video.read()

    if ret is True:

        print (frame_id)

        panoptic = predictor(frame)
        panoptic_seg, segments_info = predictor(frame)["panoptic_seg"]

        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=2)

        instances = panoptic["instances"]
        category_detections = instances[instances.pred_classes == 0]
        confident_detections = category_detections[category_detections.scores > 0.5]

        # print (confident_detections.pred_classes)
        print (len(confident_detections.pred_classes))
        print (confident_detections.pred_boxes)
        output_pred_boxes = confident_detections.pred_boxes
        for i in output_pred_boxes.__iter__():
            bbox = i.cpu().numpy()
            print(bbox)
        
        # instances.gt_boxes = confident_detections.pred_boxes[0]
        # box1 = confident_detections.pred_boxes[0]
        # print (box1.shape)
        # print (len(box1[1]))
        # print (confident_detections.scores)
        
        out = v.draw_instance_predictions(confident_detections.to("cpu"))
        cv2.imshow('frame', out.get_image()[:, :, ::-1])

        # confident_detections.to("cpu")
        # print (confident_detections.pred_boxes.to("cpu"))
        # print (confident_detections.pred_boxes.numpy())

        # kitti format
        # for n_target in len(confident_detections.pred_classes)
        #     target_id = n_target + 1
        #     if confident_detections.pred_classes 
            # results.append((frame_id, online_tlwhs, online_ids))

        frame_id = frame_id + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else :
        

        video.release()
        cv2.destroyAllWindows()
