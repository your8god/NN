from imageai import Detection

from .model import Model


def choose_model(model_kind, type='img'):
    match type:
        case 'img':
            detection = Detection.ObjectDetection() 
        case 'video':
            detection = Detection.VideoObjectDetection()
        case _:
            raise AttributeError()

    match model_kind:
        case Model.RetinaNet:
            detection.setModelTypeAsRetinaNet()
        case Model.YOLOv3:
            detection.setModelTypeAsYOLOv3()
        case Model.TinyYOLOv3:
            detection.setModelTypeAsTinyYOLOv3()

    detection.setModelPath(model_kind.path)
    detection.loadModel()
    return detection
