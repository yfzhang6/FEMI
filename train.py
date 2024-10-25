import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'\ultralytics\cfg\models\v8\yolov8.yaml')
    #model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset/Switch_data.yaml',
                patience=300,
                cache=False,
                imgsz=640,
                epochs=100,
                batch=16,
                close_mosaic=0,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                #resume='runs/train/exp17/weights/last.pt', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )

#ultralytics/cfg/models/v5/yolov5.yaml
#ultralytics/cfg/models/v3/yolov3-tiny.yaml
#ultralytics/cfg/models/v8/yolov8-efficientViT.yaml
#ultralytics/cfg/models/v8/yolov8-C2f-Faster-EMA-bifpn-P2-DA.yaml
#ultralytics/cfg/models/v8/yolov8-C2f-Faster-bifpn-P2.yaml


#ultralytics/cfg/models/v5/yolov5-GhostHGNetV2.yaml
#ultralytics/cfg/models/v5/yolov5-fasternet.yaml
#ultralytics/cfg/models/v8/yolov8-GhostHGNetV2.yaml





#ultralytics/cfg/models/v8/yolov8-C2f-Faster-EMA-bifpn.yaml
#ultralytics/cfg/models/v8/yolov8-C2f-Faster-EMA-bifpn-P2.yaml
#ultralytics/cfg/models/v8/yolov8-C2f-DAttention.yaml
#ultralytics/cfg/models/v8/yolov8.yaml--inner-iou




#ultralytics/cfg/models/v8/yolov8s-C2faster-bifpn.yaml
#ultralytics/cfg/models/v8/yolov8s-C2f-DA-Faster.yaml
#ultralytics/cfg/models/v8/yolov8s-bifpn-DAttention.yaml
#ultralytics/cfg/models/v8/yolov8s-C2faster-bifpn-DA.yaml

#ultralytics/cfg/models/v8/yolov8n-bifpn-P2.yaml
#ultralytics/cfg/models/v8/yolov8n-bifpn-P2-faster.yaml
#ultralytics/cfg/models/v8/yolov8n-bifpn-P2-DA.yaml
#ultralytics/cfg/models/v8/yolov8n-p2.yaml

#ultralytics/cfg/models/v8/yolov8-bifpn-P2-CloAtt.yaml，爆显存
#ultralytics/cfg/models/v8/yolov8n-bifpn-P2-DBB.yaml
#ultralytics/cfg/models/v8/yolov8-bifpn-P2-DBB-DA.yaml

#ultralytics/cfg/models/v8/yolov8-C2f-DAttention.yaml
#ultralytics/cfg/models/v8/yolov8-bifpn-DBB.yaml

#E:\0moguiv8\yolov8-main\ultralytics\cfg\models\v8\yolov8-FE-BIMAFPN.yaml