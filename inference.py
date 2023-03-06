#%%
from regression_model import Wav2Vec2ForRegression
from transformers import AutoConfig, AutoFeatureExtractor
import torchaudio
import torch
import pandas as pd
from datetime import datetime


max_length = 16000 * 15
model_path = "results_15.0s_3classes_2022_08_21_ 6_58_27PM/checkpoint-66720"
config = AutoConfig.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForRegression.from_pretrained(
    model_path,
    config=config,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def preprocess_function(waveform):
    inputs = feature_extractor(
        waveform,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
    )
    return inputs["input_values"]


test_data = pd.read_csv("test.csv")
df = pd.DataFrame(test_data)
test_data = test_data["filepath"]
score = []
with torch.no_grad():
    for i in range(len(test_data)):
        waveform, sample_rate = torchaudio.load(test_data[i])
        encoded = torch.FloatTensor(preprocess_function(waveform)).squeeze(0)
        encoded = encoded.to(device)
        fluency_score = model(encoded).logits.item()
        score.append(fluency_score)

df["predict"] = score
df = df.sort_values(by=["fluency"])
t = datetime.now().strftime("%Y_%m_%d_%l_%M%p")
df = df.reindex(
    columns=["filepath", "fluency", "predict", "sentence"]
)
df.to_csv(f"score_{t}.csv", index=False)


#%%
# count miss predict
c1_miss_predict = 0
c2_miss_predict = 0
c3_miss_predict = 0
total = 0

for i in range(len(df["predict"])):
    if df.iat[i, 1] == 0.1 and df.iat[i, 2] > 0.29:
        c1_miss_predict += 1
        total += 1
        continue
    elif df.iat[i, 1] == 0.5:
        if df.iat[i, 2] < 0.30 or df.iat[i, 2] > 0.7:
            c2_miss_predict += 1
            total += 1
            continue
    elif df.iat[i, 1] == 0.9:
        if df.iat[i, 2] < 0.7 or df.iat[i, 2] > 1.10:
            c3_miss_predict += 1
            total += 1


c1 = [c1_miss_predict]
c2 = [c2_miss_predict]
c3 = [c3_miss_predict]

dic = {"c1_miss": c1, "c2_miss": c2, "c3_miss": c3, "total": total}
miss_predict = pd.DataFrame(dic)
miss_predict.to_csv(f"miss_predict_{t}.csv", index=False)

#%%
# confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import evaluate

df = pd.read_csv(f"score_{t}.csv")
df["fluency"].loc[df["fluency"] == 0.1] = 0
df["fluency"].loc[df["fluency"] == 0.5] = 1
df["fluency"].loc[df["fluency"] == 0.9] = 2


df["predict"].loc[df["predict"] > 0.7] = 2
df["predict"].loc[(df["predict"] >= 0.3) & (df["predict"] <= 0.7)] = 1
df["predict"].loc[df["predict"] < 0.3] = 0


f1_metric = evaluate.load("f1")
f1_results = f1_metric.compute(
    references=df["fluency"], predictions=df["predict"], average="weighted"
)
print("f1 score", f1_results["f1"])
cf_arrays = []
array = confusion_matrix(df["fluency"], df["predict"])
cf_arrays.append(array)
for array in cf_arrays:
    df_cm = (pd.DataFrame(array, index=["0", "1", "2"], columns=["0", "1", "2"]),)

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm[0], annot=True, cmap="Blues_r")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.show()
