from model.model import Model
from model.kind import choose_model
from image_ai.data.preset import PictureStorage
from image_ai.translatator import tr 


def all_objects():
    for item in Model:
        model = choose_model(item)
        print(item)
        for picture in PictureStorage:
            try:
                res = model.detectObjectsFromImage(
                    input_image=picture.path, 
                    output_image_path=picture.output_path(str(item)),
                    minimum_percentage_probability=30
                )
                print(res)
            except:
                print(picture)
        print()
    

def custom_objects(*args):
     for item in Model:
        model = choose_model(item)
        custom = model.CustomObjects(**{obj_name: True for obj_name in args})
        print(item)
        for picture in PictureStorage:
            try:
                res = model.detectObjectsFromImage(
                    custom_objects=custom,
                    input_image=picture.path, 
                    output_image_path=picture.output_path(f'c_{str(item)}'),
                    minimum_percentage_probability=30
                )
                print(res)
            except:
                print(picture)
        print()


all_objects()
custom_objects('person', 'dog')
