class Config:
    model_weights = "E:/deep/res50fasterrcnn-model-120-mAp-0.9283537864685059.pth"
    test_root_dir = 'E:/deep/dataloader/test'
    image_path = "E:/deep/image.jpg"
    gpu_id = '0'
    num_classes = 3 + 1
    data_root_dir = " "


test_cfg = Config()
