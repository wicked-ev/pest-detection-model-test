import os
import sys
import time
import threading
import queue
import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from PIL import Image
from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

# Before running this code make sure you set and create your environment variables (.env file)
# Make sure to download the weights and image samples from the dataset for testing
load_dotenv()

OUTPUT_SAVE_PATH = os.getenv("OUTPUT_SAVE_PATH")
SAMPLE_TEST_PATH = os.getenv("SAMPLE_TEST_PATH")
WEIGHTS_PATH     = os.getenv("WEIGHTS_PATH")

# --- Pi 3 B+ tuning knobs ---
# Inference input size: smaller = faster, less accurate
INFERENCE_WIDTH  = 320
INFERENCE_HEIGHT = 320
# Process every Nth frame to reduce CPU load
FRAME_SKIP = 4
# Cap display FPS so the main thread doesn't spin
TARGET_FPS = 15


def _get_labels(detections):
    """
    Convert detection class IDs to human-readable label strings.
    COCO_CLASSES is 0-indexed, so class_id maps directly — no offset needed.
    (The +1 offset in the original videoDetection was the source of wrong labels.)
    """
    return [COCO_CLASSES[int(cid)] for cid in detections.class_id]


def _resize_for_inference(frame_bgr):
    """
    Downscale a BGR frame to the inference resolution.
    Returns the resized BGR frame as a PIL Image (RGB).
    Using INTER_NEAREST is the fastest resize filter — acceptable for detection.
    """
    small = cv2.resize(
        frame_bgr,
        (INFERENCE_WIDTH, INFERENCE_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


def imageDetection(model):
    """Run detection on a single test image, display and save the annotated result."""
    image_path = os.path.join(
        SAMPLE_TEST_PATH,
        "ash_weevil_2_png.rf.hZxgta7G4n8alrNt0cnu.jpg",
    )

    image = None  # defined before try so finally is always safe
    try:
        image = Image.open(image_path)

        # Resize to inference resolution before predicting
        image_small = image.resize(
            (INFERENCE_WIDTH, INFERENCE_HEIGHT), Image.BILINEAR
        )

        detections = model.predict(image_small, threshold=0.5)
        labels = _get_labels(detections)

        # Annotate on the small inference frame (faster than annotating full-res)
        frame = np.array(image_small)
        frame = sv.BoxAnnotator().annotate(frame, detections)
        frame = sv.LabelAnnotator().annotate(frame, detections, labels)

        # Convert back to PIL to show/save
        annotated_pil = Image.fromarray(frame)
        annotated_pil.show()

        # Derive the next file index from what already exists in the output dir
        file_count = len(os.listdir(OUTPUT_SAVE_PATH))
        save_path = os.path.join(OUTPUT_SAVE_PATH, f"output{file_count + 1}.jpg")
        annotated_pil.save(save_path)
        print(f"Saved: {save_path}")

    finally:
        if image is not None:
            image.close()


def videoDetection(model, video_path):
    """
    Run detection on a video file or a stream URL/device index.

    Architecture for Pi 3 B+:
      - Producer thread: reads frames as fast as possible and drops old ones.
      - Main thread: runs inference every FRAME_SKIP frames, annotates, and displays.
    This keeps the capture pipeline from stalling while inference is running.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_path}")

    # --- Producer thread ---------------------------------------------------
    # Uses a queue of depth 1 so we always hold the *latest* frame.
    frame_queue = queue.Queue(maxsize=1)
    stop_event  = threading.Event()

    def _capture_loop():
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                stop_event.set()
                break
            # Drop stale frame if the main thread hasn't consumed the last one
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)

    capture_thread = threading.Thread(target=_capture_loop, daemon=True)
    capture_thread.start()
    # -----------------------------------------------------------------------

    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    last_detections = None
    last_labels     = []
    frame_count     = 0
    frame_interval  = 1.0 / TARGET_FPS

    try:
        while not stop_event.is_set():
            loop_start = time.monotonic()

            # Block up to 1 s for a new frame; avoids busy-waiting
            try:
                frame_bgr = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Run inference only on every Nth frame to reduce CPU load
            if frame_count % FRAME_SKIP == 0:
                pil_small       = _resize_for_inference(frame_bgr)
                last_detections = model.predict(pil_small, threshold=0.5)
                last_labels     = _get_labels(last_detections)

            frame_count += 1

            # Annotate using the most recent valid detections
            if last_detections is not None:
                annotated = box_annotator.annotate(frame_bgr, last_detections)
                annotated = label_annotator.annotate(annotated, last_detections, last_labels)
            else:
                annotated = frame_bgr

            cv2.imshow("RF-DETR Detection", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Soft FPS cap — sleep only if we finished faster than the target
            elapsed = time.monotonic() - loop_start
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        # Always clean up even if an exception occurs
        stop_event.set()
        capture_thread.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Load model once; optimize_for_inference() applies ONNX / int8 optimizations
    model = RFDETRNano(pretrain_weights=WEIGHTS_PATH)
    model.optimize_for_inference()

    path_map = {
        "-v": os.getenv("VIDEO_SAMPLE_TEST"),
        "-s": os.getenv("VIDEO_STREAM_TEST"),
    }
    args = sys.argv[1:]

    if not args:
        print("Running image detection…")
        imageDetection(model)
    elif args[0] in path_map:
        mode = "video" if args[0] == "-v" else "stream"
        print(f"Running {mode} detection…")
        videoDetection(model, path_map[args[0]])
    else:
        print(f"Unknown argument: {args[0]}")
        print("Usage: script.py [-v | -s]")
        sys.exit(1)


if __name__ == "__main__":
    main()