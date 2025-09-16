# DFINE-onnx-tensorrt
D-FINE(Transfomer object detection) training, onnx, tensorrt.

# Training
* conda create -n dfine python=3.11.9
* conda activate dfine
* prepare dataset(COCO format)
* * pip install opencv-python
* * pip install scikit-learn
* * pip install tqdm
* * python yolo2coco.py --root_dir ./dataset/train --save_path train.json
* * python yolo2coco.py --root_dir ./dataset/valid --save_path valid.json
* * python yolo2coco.py --root_dir ./dataset/test --save_path test.json
* * yolo2coco.py (from https://github.com/z1069614715/objectdetection_script/tree/master/mmdet-course)
* Get D-FINE
* * git clone https://github.com/Peterande/D-FINE.git
* * cd D-FINE
* * pip install -r requirements.txt
* Modify the D-FINE configuration file(Take size S(N S M L X) as an example. )
* * vim configs/dfine/custom/dfine_hgnetv2_s_custom.yml 
* * * total_batch_size, for example 64.
* * * total_batch_size, for example 64.
