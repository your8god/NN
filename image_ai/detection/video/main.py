from image_ai.detection.model.model import Model
from image_ai.detection.model.kind import choose_model
from image_ai.data.preset import VideoStorage
from image_ai.translatator import tr 


for item in Model:
        model = choose_model(item, type='video')
        print(item.name)
        for video in VideoStorage:
            try:
                res = model.detectObjectsFromVideo(
                    input_file_path=video.path, 
                    output_file_path=video.output_path(item.name),
                    minimum_percentage_probability=30,
                    frame_detection_interval=20,
                    log_progress=True,
                    display_percentage_probability=False
                )
                print(res)
            except:
                print(video.name)
        print()