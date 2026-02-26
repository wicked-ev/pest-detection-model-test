this project is to test the yolov8_pest_detection we have that runs on RF-DETR (Nano) architecture

this project uses UV as package manager instead of PIP make sure to install UV first:
- uv (https://docs.astral.sh/uv/getting-started/installation/)
- you need to run `uv sync` to install dependencies and create a virtual environment
- to run the code `uv python main.py` this will run the model on image detection
- to run the model on video detection use the flag `-v`, `uv python main.py -v`
- and for video stream `uv python main.py -s`
- make sure to setup the video path/links in your .env file before running

**BUT before running this code make sure first to**:
- download the weights from roboflow platform
- download sample images for testing
- create .env file with for weights, sample images and output path's

links:
- weights (https://app.roboflow.com/pest-detection-bed8r/yolov8_pest_detection/models/yolov8_pest_detection/6)
- dataset (https://app.roboflow.com/pest-detection-bed8r/yolov8_pest_detection/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
