import os
import sys

import cv2
import supervision as sv
from dotenv import load_dotenv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# before running this code make sure you set and create your environment variables (.env file)
# make sure to download the weights and image samples from the dataset for testing
load_dotenv()


def imageDectection(model):

    output_save = os.getenv("OUTPUT_SAVE_PATH")
    image_sample_path = os.path.join(
        os.getenv("SAMPLE_TEST_PATH"), "ash_weevil_2_png.rf.hZxgta7G4n8alrNt0cnu.jpg"
    )
    image = Image.open(image_sample_path)
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


def videoDetection(model, video_path):

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_path}")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = model.predict(frame_rgb, threshold=0.5)

        labels = [COCO_CLASSES[int(class_id) + 1] for class_id in detections.class_id]

        annotated_frame = sv.BoxAnnotator().annotate(frame_bgr, detections)
        annotated_frame = sv.LabelAnnotator().annotate(
            annotated_frame, detections, labels
        )

        cv2.imshow("RF-DETR Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    weights_path = os.getenv("WEIGHTS_PATH")
    model = RFDETRNano(pretrain_weights=weights_path)
    model.optimize_for_inference()
    path = {"-v": os.getenv("VIDEO_SAMPLE_TEST"), "-s": os.getenv("VIDEO_STREAM_TEST")}
    args = sys.argv[1:]
    if args[0] == "-v":
        print("run video detection")
        videoDetection(model, path["-v"])
    elif args[0] == "-s":
        print(" run stream detection")
        videoDetection(model, path["-s"])
    else:
        print("run image detection")
        imageDectection(model)


if __name__ == "__main__":
    main()
