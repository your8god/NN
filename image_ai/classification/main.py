from model.model import Model
from model.kind import choose_model
from image_ai.data.preset import PictureStorage
from image_ai.translatator import tr 


for item in Model:
    model = choose_model(item)
    print(item)
    for picture in list(PictureStorage)[6:]:
        predictions, probabilities = model.classifyImage(picture.path, result_count=3)
        print([tr(i) for i in predictions], probabilities)
    print()
 