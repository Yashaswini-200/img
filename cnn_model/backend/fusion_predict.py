from cnn_model.predict_cnn import predict_image as cnn_predict
from fft_metadata_predict import predict_image as fft_predict

def fused_predict(img_path):
    cnn_score = cnn_predict(img_path)
    fft_score = fft_predict(img_path)
    final_score = (cnn_score + fft_score) / 2
    label = "AI-Generated" if final_score > 0.5 else "Real"
    return label, final_score
