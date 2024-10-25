import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt') # select your model.pt path
    model.predict(source='',
                project='runs/detect',
                name='exp',
                save=True,
                visualize=False # visualize model features maps

                )
