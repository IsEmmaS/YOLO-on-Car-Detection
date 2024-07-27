# YOLO (You Only Look Once) Object Detection

##  YOLO object detection algorithm in PyTorch on Custom data.
- 🌋 (This is a work in progress....).
- Use data Dataset download from [Kaggle](https://www.kaggle.com/datasets/sshikamaru/car-object-detection).

### Network structure from [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg), Use [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) algorithm include:

* **convolutional** 🥱 Just Convolution Layer.
* **shortcut** : Implement skip connection.
* **upsample** : Up Sample the Image.
* **route** : Merge feature in differ layers.
* **YOLO** : Get the Result of the network.
* **net** : Total network configuration info.