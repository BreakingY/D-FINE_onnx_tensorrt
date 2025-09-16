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
* * pip install matplotlib
* * pip install onnx onnxsim onnxruntime
* Modify the D-FINE configuration file(Take size S(N S M L X) as an example. )
* * vim configs/dfine/custom/dfine_hgnetv2_s_custom.yml 
* * * total_batch_size, for example 64.
* * * total_batch_size, for example 128.
```
__include__: [
  '../../dataset/custom_detection.yml',
  '../../runtime.yml',
  '../include/dataloader.yml',
  '../include/optimizer.yml',
  '../include/dfine_hgnetv2.yml',
]
```
* * vim configs/dataset/custom_detection.yml
* * * num_classes: 2
* * * train_dataloader->img_folder: PATH/D-FINE_onnx_tensorrt/dataset/train/images
* * * train_dataloader->ann_file: PATH/D-FINE_onnx_tensorrt/dataset/train/annotations/train.json
* * * val_dataloader->img_folder: PATH/D-FINE_onnx_tensorrt/dataset/valid/images
* * * val_dataloader->ann_file: PATH/D-FINE_onnx_tensorrt/dataset/valid/annotations/valid.json
* * vim configs/runtime.yml
* * * Keep the default
* * vim configs/dfine/include/dataloader.yml
* * * Keep the default, image size 640 x 640
* * vim configs/dfine/include/optimizer.yml
* * * Keep the default
* * vim configs/dfine/include/dfine_hgnetv2.yml
* * * Keep the default, image size 640 x 640
* Training
* * export model=s  # n s m l x
* * CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml --use-amp --seed=0

# ONNX TODO
* python tools/deployment/export_onnx.py --check -c configs/dfine/custom/dfine_hgnetv2_s_custom.yml -r ./output/dfine_hgnetv2_s_custom/best_stg1.pth
* python dfine_onnx_inference.py

# TensorRT TODO
* /data/sunkx/TensorRT-8.5.1.7/bin/trtexec --onnx=./output/dfine_hgnetv2_s_custom/best_stg1.onnx --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:4x3x640x640 --saveEngine=best.engine --fp16
* make
* ./dfine_trt_inference ./best.engine ./test.jpg


