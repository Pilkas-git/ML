import numpy as np

class Config:
    resume = ''#'E:/deep/res50fasterrcnn-modelv2-80-mAp-0.8895748257637024.pth' 
    start_epoch = 0 
    use_dr_loss = 1

    # data transform parameter
    min_size = 448
    max_size = 1024 
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    # anchor parameters
    anchor_size = [32, 64, 128, 256, 512]
    anchor_ratio = [0.5, 1.0, 2.0]

    # roi align parameters
    roi_out_size = [7, 7]
    roi_sample_rate = 2

    # rpn process parameters
    rpn_pre_nms_top_n_train = 2000
    rpn_post_nms_top_n_train = 2000

    rpn_pre_nms_top_n_test = 1000
    rpn_post_nms_top_n_test = 1000

    '''rpn_nms_thresh = 0.7
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5'''

    rpn_nms_thresh = 0.5
    rpn_fg_iou_thresh = 0.5
    rpn_bg_iou_thresh = 0.4
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5

    # remove low threshold target    
    box_score_thresh = 0.05     # mAp min_score 0.01 max overlap=0.45  top_k 0.45
    #box_score_thresh = 0.01     # faster rcnn min_score 0.05 max overlap=0.5  top_k 100
    box_nms_thresh = 0.5
    #box_nms_thresh = 0.45
    box_detections_per_img = 100
    #box_detections_per_img = 200
    box_fg_iou_thresh = 0.45
    box_bg_iou_thresh = 0.45
    box_batch_size_per_image = 512
    box_positive_fraction = 0.25
    bbox_reg_weights = None

    num_epochs = 1000  # train epochs
    eval_after = 20

    # learning rate parameters
    lr = 0.005   #starts with 0.001 then they reduced after 50k/350k iterations
    momentum = 0.9   #from paper on Faster RCNN We use a momentum of 0.9 and a weight decay of 0.0005 [37].
    weight_decay = 0.0005   # too much model never fits. To little -> stop model earlyer to avoid overfitting

    # learning rate schedule
    lr_gamma = 0.33
    lr_dec_step_size = 80

    batch_size = 2
    device_name = 'cuda'

    num_class = 3 + 1
    voc_labels = ('apple', 'snowman', 'motorcycle')  # class orders in oidv6 dataset 1=014j1m -> 2=0152hh -> 3=04_sv
    class_Labels = [ "/m/014j1m", "/m/04_sv", "/m/0152hh"] # Apple, Motorcycle, snowman
    train_root_dir = 'E:/deep/dataloader/Train'
    valid_root_dir = 'E:/deep/dataloader/Validation'
    model_save_dir = "E:/deep"


cfg = Config()
