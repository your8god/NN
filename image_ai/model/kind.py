from imageai import Classification

from .model import Model


def choose_model(model_kind):
    prediction = Classification.ImageClassification()

    match model_kind:
        case Model.MobileNetV2:
            prediction.setModelTypeAsMobileNetV2()
        case Model.RestNet50:
            prediction.setModelTypeAsResNet50()
        case Model.InceptionV3:
            prediction.setModelTypeAsInceptionV3()
        case Model.DenseNet121:
            prediction.setModelTypeAsDenseNet121()

    prediction.setModelPath(model_kind.path)
    prediction.loadModel()
    return prediction