class LineSqueezeConfig:
    det_model_path = "./weights/line_squeeze/det_v3.onnx"
    ocr_model_path = "./weights/line_squeeze/rec_ppocrv5en_v1.onnx"
    ocr_metadata_path = "./weights/common/official/PP-en_rec_ppocr_v5/inference.yml"
    det_nc = 2
    det_conf_threshold = 0.5
    det_nms_threshold = 0.5
