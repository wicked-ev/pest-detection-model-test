import os

import supervision as sv
from dotenv import load_dotenv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# before running this code make sure you set and create your environment variables (.env file)
# make sure to download the weights and image samples from the dataset for testing
load_dotenv()

weights_path = os.getenv("WEIGHTS_PATH")
output_save = os.getenv("OUTPUT_SAVE_PATH")
image_sample_path = os.path.join(
    os.getenv("SAMPLE_TEST_PATH"),
    "ash_weevil_2_png.rf.hZxgta7G4n8alrNt0cnu.jpg",
)


def main():
    model = RFDETRNano(pretrain_weights=weights_path)
    image = Image.open(image_sample_path)

    model.optimize_for_inference()
    detections = model.predict(image, threshold=0.5)

    labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
    annotated_image = sv.BoxAnnotator().annotate(image, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    annotated_image.show()
    file_count = sum(
        1
        for item in os.listdir(output_save)
        if os.path.isfile(os.path.join(output_save, item))
    )
    annotated_image.save(os.path.join(output_save, f"output{file_count + 1}.jpg"))


if __name__ == "__main__":
    main()
