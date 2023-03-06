import pandas as pd
import json
import glob


with open("train-all-info.json") as f:
    train_data = json.load(f)

train_filepath = []
train_fluency = []
train_sentence = []
text_df = pd.read_csv("train/text.txt", sep="\t", header=None, nrows=2500)
text_df.columns = ["file_name", "sentence"]
count = 0

for k, v in train_data.items():
    if train_data[k]["fluency"] in [0, 1, 2, 3, 4, 5, 6]:
        train_data[k]["fluency"] = 1
    elif train_data[k]["fluency"] in [7, 8]:
        train_data[k]["fluency"] = 5
    else:
        train_data[k]["fluency"] = 9
    train_fluency.append(train_data[k]["fluency"] / 10.0)
    train_filepath.append(glob.glob(f"train/{k}" + ".WAV")[0])
    train_sentence.append(text_df.iat[count, 1])
    count += 1

dict = {"filepath": train_filepath, "fluency": train_fluency, "sentence": train_sentence}
df = pd.DataFrame(dict)
df.to_csv("train.csv", index=False)  # index = False 移除多出來的 Unnamed index


with open("test-all-info.json") as f:
    test_data = json.load(f)

test_filepath = []
test_fluency = []
test_sentence = []
text_df = pd.read_csv("test/text.txt", sep="\t", header=None, nrows=2500)
text_df.columns = ["file_name", "sentence"]
count = 0

for k, v in test_data.items():
    if test_data[k]["fluency"] in [0, 1, 2, 3, 4, 5, 6]:
        test_data[k]["fluency"] = 1
    elif test_data[k]["fluency"] in [7, 8]:
        test_data[k]["fluency"] = 5
    else:
        test_data[k]["fluency"] = 9
    test_fluency.append(test_data[k]["fluency"] / 10.0)
    test_filepath.append(glob.glob(f"test/{k}" + ".WAV")[0])
    test_sentence.append(text_df.iat[count, 1])
    count += 1

dict = {"filepath": test_filepath, "fluency": test_fluency, "sentence": test_sentence}
df = pd.DataFrame(dict)
df.to_csv("test.csv", index=False)
