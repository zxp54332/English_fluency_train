import onnx
import onnxruntime
from regression_model import Wav2Vec2ForRegression
from transformers import AutoConfig, AutoFeatureExtractor
import torchaudio
import torch
import pandas as pd
from datetime import datetime


# 載入原本訓練好的模型
max_length = 16000 * 15
model_path = "results_240.0kSR_3Classes_divide10_mae/checkpoint-83400"
config = AutoConfig.from_pretrained(model_path)
model = Wav2Vec2ForRegression.from_pretrained(
    model_path,
    config=config,
)

# 設定一個輸入，connected speech的輸入為 (batch_size, waveform)
x = torch.randn(1, 18654)


# 輸出成 onnx 格式
# 這裡要注意的是 dynamic_axes，如果輸入的維度是動態的，在這邊要進行設定
# 這裡我們設定 第0維 batch_size 跟第1維 waveform，是動態的

torch.onnx.export(
    model,
    x,
    f="torch-model.onnx",
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size", 1: "length"}, "output": {0: "batch_size"}},
    do_constant_folding=True, 
    opset_version=13, 
)


# 載入輸出好的onnx模型
ort_session = onnxruntime.InferenceSession("torch-model.onnx")
onnx_model = onnx.load("torch-model.onnx")
onnx.checker.check_model(onnx_model)
print(ort_session.get_inputs()[0].name)

# 這裡的 'input' 對應到上面的 `input_names`
ort_inputs = {"input": torch.randn(1, 18654).numpy()}
ort_outs = ort_session.run(None, ort_inputs)


test_data = pd.read_csv("test.csv")
df = pd.DataFrame(test_data)
test_data = test_data["filepath"]
with torch.no_grad():
    for i in range(len(test_data)-2490):
        waveform, sample_rate = torchaudio.load(test_data[i])
        #fluency_score = model(waveform).logits.item()
        print(waveform.numpy().dtype)
        ort_inputs = {"input":waveform.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        print(ort_outs)
        print(fluency_score)

