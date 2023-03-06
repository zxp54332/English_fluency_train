import json
import glob
import os


with open("train-all-info.json") as f:
    data = json.load(f)
train_count = 0
for k, v in data.items():
    file = glob.glob(f"data/temp/{k}" + ".WAV")  # 抓取檔按路徑
    file_name = file[0].split("/")[-1]
    os.replace(file[0], "data/train/" + file_name)  # 移動檔案
    train_count += 1
file = glob.glob("data/speechocean762/train/text*")
os.replace(file[0], "data/train/text.txt")

with open("test-all-info.json") as f:
    data = json.load(f)
test_count = 0
for k, v in data.items():
    file = glob.glob(f"data/temp/{k}" + ".WAV")  # 抓取檔按路徑
    file_name = file[0].split("/")[-1]
    os.replace(file[0], "data/test/" + file_name)  # 移動檔案
    test_count += 1
file = glob.glob("data/speechocean762/test/text*")
os.replace(file[0], "data/test/text.txt")

print("train data:", train_count)
print("test data:", test_count)
