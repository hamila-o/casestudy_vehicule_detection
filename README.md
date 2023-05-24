# casestudy_vehicule_detection
This application was made as part of a use-case study for a job interview for the position of research engineer (in machine learning). 
The application developed in this project is a proof-of-concept of an object detection model for cars, trucks, and motorcycles from 2D-RGB images. The code is in Python programming language (Python 3). The model used is the YOLOv5s pretrained model (link: https://github.com/ultralytics/yolov5). The dataset employed to test the model is a modified version of the Vehicles-OpenImages Dataset containing only the relevant classes (link: https://public.roboflow.com/object-detection/vehicles-openimages/1). The execution environment is Ubuntu. The project comes as a single program called 'predict.py'.

<br /> 

To install the necessary dependencies, run the following commands:
```bash
git clone https://github.com/hamila-o/casestudy_vehicule_detection
cd casestudy_vehicule_detection
bash setup.sh
```
The program uses the YOLOv5s model to process the images in the 'data/images' folder and produces outputs in a folder named 'output'. To execute the program, run the following command (values between brackets are optional):

```bash
python predict.py [-h] [-i INPUT] [-o OUTPUT] [-e LABELS]
```
The program comes with a few options:
<br /> 
- **-i, --input**: Specify the input directory (in the POSIX format) of the images. The default value is './data/images/'.
- **-o, --output**: Specify the output directory (in the POSIX format) for the predictions. The default value is './output/'.
- **-e, --labels**: Specify the labels directory (in the POSIX format) for the evaluation. Evaluations are stored in the output directory. The default value is './data/labels/'.



