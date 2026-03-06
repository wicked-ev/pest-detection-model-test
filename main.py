import os
import sys
import time
import cv2
import supervision as sv
from dotenv import load_dotenv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# before running this code make sure you set and create your environment variables (.env file)
# make sure to download the weights and image samples from the dataset for testing
load_dotenv()
output_save = os.getenv("OUTPUT_SAVE_PATH")
file_count = len(os.listdir(output_save))


def imageDectection(model):

    image_sample_path = os.path.join(
        os.getenv("SAMPLE_TEST_PATH"), "ash_weevil_2_png.rf.hZxgta7G4n8alrNt0cnu.jpg"
    )
    try:
        image = Image.open(image_sample_path)
        detections = model.predict(image, threshold=0.5)

        labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
        annotated_image = sv.BoxAnnotator().annotate(image, detections)
        annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
        annotated_image.show()

        file_count += 1
        annotated_image.save(os.path.join(output_save, f"output{file_count + 1}.jpg"))
    finally:
        image.close()

def videoDetection(model, video_path):

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_path}")
    FRAME_SKIP = 2
    frame_count = 0
    last_detections, last_lables = None, []
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            time.sleep(0.001)  # so the loop wont max out the cpu
            break

        if frame_count % FRAME_SKIP == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            last_detections = model.predict(frame_rgb, threshold=0.5)
            last_lables = [
                COCO_CLASSES[int(class_id) + 1] for class_id in last_detections.class_id
            ]
        frame_count += 1

        annotated_frame = box_annotator.annotate(frame_bgr, last_detections)
        annotated_frame = label_annotator.annotate(
            annotated_frame, last_detections, last_lables
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

    if not args:
        print("run image detection")
        imageDectection(model)        
    elif args[0] == "-v":
        print("run video detection")
        videoDetection(model, path["-v"])
    elif args[0] == "-s":
        print("run stream detection")
        videoDetection(model, path["-s"])

if __name__ == "__main__":
    main()
