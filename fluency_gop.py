#%%
import pandas as pd
import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
from utils import getEngGOPresult
from os.path import basename
import os
import httpx
from datetime import datetime

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="metricgan-u5000sync/",
    savedir="metricgan-u5000sync/",
    hparams_file="hyperparams.yml",
)

df = pd.read_csv("score_2022_08_31_ 6_08PM.csv")
scores = []
new_scores = []
new_predict = []

client = httpx.Client()
rate = 16000
# for i, result in tqdm(df.iterrows(), total=2500):
for i in range(2500):
    predict_fluency, text, file = float(df.iat[i, 2]), df.iat[i, 3], df.iat[i, 0]
    noisy = enhance_model.load_audio(file).unsqueeze(0)
    print(noisy.shape)
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.0]))
    print(enhanced.shape)
    file = basename(file)
    torchaudio.save(f"./{file}", enhanced, rate, format="wav")

    try:
        res = getEngGOPresult(f"./{file}", text, client)
        score = res.get("gop").get("GOP_scores")
    except Exception:
        score = 0

    if predict_fluency > 0.7:
        predict_fluency = 3
    elif predict_fluency >= 0.3 and predict_fluency <= 0.7:
        predict_fluency = 2
    else:
        predict_fluency = 1

    new_score = score * predict_fluency
    new_scores.append(new_score)
    print(i + 1)
    print("gop_score", score)
    print("fluency level", predict_fluency)
    print("fluency_score", new_score)

    if new_score > 160:
        new_score = 0.9
    elif new_score >= 80 and new_score <= 160:
        new_score = 0.5
    else:
        new_score = 0.1

    new_predict.append(new_score)
    scores.append(score)
    os.remove(f"./{file}")
client.close()

df["fluency_gop_predict"] = new_predict
df["new_score"] = new_scores
df["gop_score"] = scores
df = df.reindex(
    columns=[
        "filepath",
        "fluency",
        "fluency_gop_predict",
        "gop_score",
        "new_score",
        "sentence",
    ]
)

t = datetime.now().strftime("%Y_%m_%d_%l_%M%p")
df.to_csv(f"fluency_gop_test_{t}.csv", index=False)

#%%
# count miss predict
c1_miss_predict = 0
c2_miss_predict = 0
c3_miss_predict = 0
total = 0

for i in range(len(df["fluency"])):
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
miss_predict.to_csv(f"fix_miss_predict_{t}.csv", index=False)
