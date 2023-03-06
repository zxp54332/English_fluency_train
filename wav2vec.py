from transformers import AutoFeatureExtractor
from transformers import TrainingArguments, Trainer, EvalPrediction, AutoConfig
from regression_model import Wav2Vec2ForRegression
from datasets import load_dataset, Audio, load_from_disk
import os
import evaluate
from datetime import datetime

## 移至 MLflowCallback 的 906 行 on_save 全部註解，如此不會在每個 epoch 上傳 model
from transformers.integrations import MLflowCallback
from transformers import EarlyStoppingCallback
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


# load data
if os.path.exists("cache_data/"):
    train_dataset = load_from_disk("cache_data/train")
    eval_dataset = load_from_disk("cache_data/test")
    print("load file complete")
else:
    data_files = {
        "train": "train.csv",
        "validation": "test.csv",
    }
    dataset = load_dataset(
        "csv",
        data_files=data_files,
    )
    train_dataset = dataset["train"]
    train_dataset = train_dataset.cast_column("filepath", Audio(sampling_rate=16000))
    eval_dataset = dataset["validation"]
    eval_dataset = eval_dataset.cast_column("filepath", Audio(sampling_rate=16000))
    # save data
    train_dataset.save_to_disk("cache_data/train/")
    eval_dataset.save_to_disk("cache_data/test/")
    print("save file complete")

train_label_list = train_dataset.unique("fluency")
train_label_list.sort()  # Let's sort it for determinism
eval_label_list = eval_dataset.unique("fluency")
eval_label_list.sort()
num_labels = len(train_label_list)
print("train_dataset", train_dataset["filepath"][0])
print("train_labels", train_label_list)
print("test_labels", eval_label_list)
print(f"A classification problem with {num_labels} classes: {train_label_list}")


# preprocessing data
max_length = 16000 * 15


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["filepath"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        padding=True,
    )
    return inputs


feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
encoded = train_dataset.map(
    preprocess_function, remove_columns=(["filepath", "sentence"]), batched=True
)
encoded = encoded.rename_column("fluency", "label")
encoded_val = eval_dataset.map(
    preprocess_function, remove_columns=(["filepath", "sentence"]), batched=True
)
encoded_val = encoded_val.rename_column("fluency", "label")
print("encoded", encoded)


# define metrics
#mse_metric = evaluate.load("mse")
metric = evaluate.combine(["mse", "mae"])


def compute_metrics(eval_pred: EvalPrediction):
    # print(eval_pred.predictions)
    # print(eval_pred.label_ids)
    metrics = metric.compute(
        predictions=eval_pred.predictions, references=eval_pred.label_ids
    )
    # print(mae) -- {'mae': 0.xxxx}
    return {"mse": metrics["mse"], "mae": metrics["mae"]}


# load pretrained model
config = AutoConfig.from_pretrained("facebook/wav2vec2-base")
config.num_labels = 1  # regression loss 設 1
config.problem_type = "regression"
model = Wav2Vec2ForRegression.from_pretrained("facebook/wav2vec2-base", config=config)


# training arguments
t = datetime.now().strftime("%Y_%m_%d_%l_%M_%S%p")
output_dir = f"./results_{max_length/16000}s_{num_labels}classes_{t}"
training_args = TrainingArguments(
    # no_cuda=True,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    num_train_epochs=100,
    save_total_limit=20,
    #metric_for_best_model="eval_mse",
    # load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded,
    eval_dataset=encoded_val,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
)


# mlflow record
with mlflow.start_run(experiment_id="85", run_name="torch_1.11.0_smooth_L1Loss") as run:

    trainer.train()
    # trainer.train(resume_from_checkpoint='')

    # save best model if load_best_model_at_end=True or save last model
    trainer.save_model(output_dir=f"{output_dir}/latest_model")

    # model evaluation
    eval_results = trainer.predict(encoded_val)
    print(eval_results.metrics)

    # classification
    predict = []
    fluency = []
    class1 = []
    class2 = []
    class3 = []
    for i in range(len(eval_results.predictions)):
        if eval_results.predictions[i][0] > 0.7:
            predict.append(2)
        elif (
            eval_results.predictions[i][0] >= 0.3
            and eval_results.predictions[i][0] <= 0.7
        ):
            predict.append(1)
        elif eval_results.predictions[i][0] < 0.3:
            predict.append(0)

    for i in range(len(eval_results.label_ids)):
        if eval_results.label_ids[i] < 0.11:  # ids 0.1 實際為 0.1000000014901161
            fluency.append(0)
            class1.append(eval_results.predictions[i][0])
        elif eval_results.label_ids[i] == 0.5:
            fluency.append(1)
            class2.append(eval_results.predictions[i][0])
        elif eval_results.label_ids[i] > 0.89:  # ids 0.9 實際為 0.899999976158142
            fluency.append(2)
            class3.append(eval_results.predictions[i][0])

    # print("fluency", fluency)
    # print("predict", predict)
    # print(eval_results.label_ids)

    # f1 score
    f1_metric = evaluate.load("f1")
    f1 = f1_metric.compute(
        predictions=predict,
        references=fluency,
        average="weighted",
    )
    print("f1 score", f1["f1"])

    # count predict logic
    fig = plt.figure(figsize=(30, 4))
    ax1 = plt.subplot(221)
    plt.hist(class1, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
    for p in ax1.patches:
        ax1.annotate(
            str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
        )
    ax1.set_title("class1", fontsize=16)
    ax1.spines["top"].set_visible(False)  # 刪除外框
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="best", fontsize=11)

    ax2 = plt.subplot(222)
    plt.hist(class2, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
    for p in ax2.patches:
        ax2.annotate(
            str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
        )
    ax2.set_title("class2", fontsize=16)
    ax2.spines["top"].set_visible(False)  # 刪除外框
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="best", fontsize=11)

    ax3 = plt.subplot(223)
    plt.hist(class3, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
    for p in ax3.patches:
        ax3.annotate(
            str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
        )
    ax3.set_title("class3", fontsize=16)
    ax3.spines["top"].set_visible(False)  # 刪除外框
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="best", fontsize=11)

    fig.tight_layout()
    plt.savefig(f"{output_dir}/predict_logic.png")

    # confusion matrix
    dict = {"fluency": fluency, "predict": predict}
    df = pd.DataFrame(dict)
    cf_arrays = []
    array = confusion_matrix(df["fluency"], df["predict"])
    cf_arrays.append(array)
    for array in cf_arrays:
        df_cm = (pd.DataFrame(array, index=["0", "1", "2"], columns=["0", "1", "2"]),)

        fig1 = plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm[0], annot=True, cmap="Blues_r")
        plt.xlabel("Pred")
        plt.ylabel("True")
        plt.savefig(f"{output_dir}/confusion_matrix.png")

    # log
    mlflow.log_metrics(eval_results.metrics)
    mlflow.log_metric("test_f1_score", f1["f1"])
    mlflow.log_figure(fig, "predict_logic.png")
    mlflow.log_figure(fig1, "confusion_matrix.png")
    MLflowCallback()._ml_flow.pyfunc.log_model(
        "model",
        artifacts={"model_path": f"{output_dir}/latest_model"},
        python_model=MLflowCallback()._ml_flow.pyfunc.PythonModel(),
    )

print("--------------------mlflow recording finish--------------------")
