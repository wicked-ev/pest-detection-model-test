this project is to test the yolov8_pest_detection we have that runs on RF-DETR (Nano) architecture

this project uses UV as package manager instead of PIP make sure to install UV first:
- uv (https://docs.astral.sh/uv/getting-started/installation/)
- you need to run `uv sync` to install dependencies and create a virtual environment
- then to run the project `uv python main.py` 

**BUT before running this code make sure first to**:
- download the weights from roboflow platform
- download sample images for testing
- create .env file with for weights, sample images and output path's

links:
- weights (https://app.roboflow.com/pest-detection-bed8r/yolov8_pest_detection/models/yolov8_pest_detection/6)
- dataset (https://app.roboflow.com/pest-detection-bed8r/yolov8_pest_detection/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true