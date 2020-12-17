import detectron2
import numpy as np
import os, cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from detectron2.data import MetadataCatalog, DatasetCatalog

vid_path = "/home/nikkhadijah/Data/Track/VideoFiles/GNE_663_110520_0714_0739_Cam03.mp4"
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

while(video.isOpened()):
    ret, frame = video.read()

    # outputs = predictor(frame)
    # print(outputs["instances"].pred_classes)
    # print(len(outputs["instances"].pred_classes))

    panoptic = predictor(frame)
    # panoptic_seg, segments_info = predictor(frame)["panoptic_seg"]
    
    # print (segments_info[].pred_class)
    print(panoptic["instances"].pred_classes)
    print(len(panoptic["instances"].pred_classes))


    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    instances = panoptic["instances"]
    category_detections = instances[instances.pred_classes == 0]
    confident_detections = category_detections[category_detections.scores > 0.5]
    
    out = v.draw_instance_predictions(confident_detections.to("cpu"))
    cv2.imshow('frame', out.get_image()[:, :, ::-1])


    # # print ('coco_2017_val')
    # visualizer = Visualizer(frame[:, :, ::-1], MetadataCatalog.get('coco_2017_val'), scale=1)
    
    # instances = outputs["instances"]
    # # category_detections = instances[instances.pred_classes == 1]
    # # confident_detections = instances[instances.scores > 0.5]
    
    # img_output = visualizer.draw_instance_predictions(instances.to("cpu"))

    # cv2.imshow('frame', img_output.get_image()[:, :, ::-1])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Display the resulting frame
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()