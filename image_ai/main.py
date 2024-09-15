from imageai import Classification

from model.model import Model
from model.kind import choose_model
from data.preset import PictureStorage
from translatator import tr 


for item in Model:
    model = choose_model(item)
    print(item)
    for picture in PictureStorage:
        predictions, probabilities = model.classifyImage(picture.path, result_count=3)
        print([tr(i) for i in predictions], probabilities)
