
### Requirements
- PyTorch 1.8
- Python 3.8
- Opencv 4.4

### Before Start

We'll call the directory that you cloned ERFNet-CULane-PyTorch as `$ERFNet_ROOT`.

### Testing
1. Download our trained models to `./weight`
    
2. Run test script
    ```Shell
    cd $ERFNet_ROOT
    python demo.py
    ```
    
### Training
1. Download the pre-trained model
    
2. Training ERFNet model
    ```Shell
    cd $ERFNet_ROOT
    python train_cus.py
    ```
   