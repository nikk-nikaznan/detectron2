import detectron2
import numpy as np
import os, cv2
import sys

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog, DatasetCatalog

filename = "video.avi"
kitti_filename = "Export_video_Detectron2_results.txt"
vid_path = "/home/"+filename
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
        
        pred_class = []
        for i in confident_detections.pred_classes.__iter__():
            pred_class.append(i.cpu().numpy())
        
        out = v.draw_instance_predictions(confident_detections.to("cpu"))
        cv2.imshow('frame', out.get_image()[:, :, ::-1])

        # # kitti format
        for n_target in range (len(pred_class)):
            frame = frame_id
            target_id = n_target + 1            
            type_id = pred_class[n_target]
            output_pred_boxes = confident_detections.pred_boxes[n_target]
            for i in output_pred_boxes.__iter__():
                bbox = i.cpu().numpy()
            
            # print (bbox)
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]

            print (frame, target_id, type_id, x1, y1, x2, y2)
            results.append((frame, target_id, type_id, x1, y1, x2, y2))

        frame_id = frame_id + 1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else :
        
        save_format = '{frame} {target_id} {type_id} 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
        with open(kitti_filename, 'w') as f:
            for frame, target_id, type_id, x1, y1, x2, y2 in results:
                if type_id == 0:
                    type_id = "person"
                line = save_format.format(frame=frame, target_id=target_id, type_id=type_id, x1=x1, y1=y1, x2=x2, y2=y2)
                f.write(line)

        video.release()
        cv2.destroyAllWindows()
