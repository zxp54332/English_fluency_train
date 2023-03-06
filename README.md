# English_fluency_train
## 專案位置 145主機
```bash
cd /media/DATA/laurence/English_fluency_train
conda activate fluency
source .env
```
* 若建立新的虛擬環境則需安裝
``python3.8 -m pip install -r requirement.txt``
  * 到 `wav2vec.py` 的 `MLflowCallback` 定義處，將 `on_save` 方法註解，避免每個 checkpoint 都上傳 MLflow
## Download Data
* 下載 corpus 後自動分成 train/test 資料夾
```bash
./download_data.sh

|- train/test folder/
|   |- file1.WAV
|   |- file2.WAV
|   |- ...
```
## Prepare CSV Data
* 在 `create_csv_file.py` 修改想要的 label，並產生 csv file (train.csv、test.csv)
```bash
python3.8 create_csv_file.py
```
## Training
* 在 `wav2vec.py` 中可透過 ``max_length`` 修改輸入 wav 的秒數
* 執行時會產生 cache_data 快取資料夾，若修改訓練 csv data，則需移除此快取資料夾
```bash
python3.8 wav2vec.py
```
* OONX 架構在此測試無明顯加速，因此不執行 `reduced_model.py`，執行產生`torch-model.onnx`，不影響原本模型
## Inference
* **分開執行各個功能，若要一次執行請確認代碼**
* 請確認 ``model_path`` 及 ``max_length`` 是否正確
* 建立 score.csv、miss_predict.csv
```bash
python3.8 inference.py
```
## Analysis Data
* **分開執行各個功能，若要一次執行請確認代碼**
* 分析 training、testing data
* 分析 predict label -- 請在 ``count predict logics`` 中確認 csv 檔名
* 計算 mean、std
* 新增 count gop_fix predict logics 分析 (2022/08/24)
  * 先執行 fluency_gop.py 得到 fluency_gop_test.csv
```bash
python3.8 analysis_data.py
```
## Local GOP testing
* <font color=red> **New score = (Fluency label) * (GOP score)** </font>
  * **Fluency label：1~3**
  * **GOP score：0~100** 
* 將 inference 產生的 score.csv 寫入 `fluency_gop.py`。以下範例：
  ```python
  df = pd.read_csv("score_2022_08_23_ 5_45AM.csv")
  ```
* 產生 `fluency_gop_test.csv`、`fix_miss_predict.csv` 可配合 `analysis_data.py` 共同分析
  ```bash
  python3.8 fluency_gop.py
  ```
## MLflow
* **查看 Experiments 中的 Fluency_eng**
## Pytorch
* 版本可能影響預測結果，盡量不要跨版本測試 ( torch=1.10.1+cu102 訓練不起 )
* 加入 GOP testing 後升級至 `torch=1.11.0+cu102`、`torchaudio==0.11.0+cu102`、`torchvision==0.12.0+cu102`

  ```bash
  python3.8 pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
  ```
* Fluency web 為 `1.12.1` cpu 版
