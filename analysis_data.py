#%%
import pandas as pd
import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

## train data ##
train_data = pd.read_csv("train.csv")
# print(train_data)
train_data["fluency"].value_counts().plot(kind="bar", ax=ax1)
for p in ax1.patches:
    ax1.annotate(
        str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02), fontsize=13
    )
ax1.set_title("train", fontsize=16)
ax1.spines["top"].set_visible(False)  # 刪除外框
ax1.spines["right"].set_visible(False)
ax1.legend(loc="best", fontsize=11)

## test data ##
test_data = pd.read_csv("test.csv")
# print(test_data)
test_data["fluency"].value_counts().plot(kind="bar", ax=ax2)
for p in ax2.patches:
    ax2.annotate(
        str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02), fontsize=13
    )
ax2.set_title("test", fontsize=16)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.legend(loc="best", fontsize=11)

plt.suptitle("Data set", fontsize=20)
plt.show()

# %%
## count predict logics
import pandas as pd
import matplotlib.pyplot as plt

c1 = []
c2 = []
c3 = []
df = pd.read_csv("fluency_gop_test_2022_08_31_ 7_25PM.csv")
for i in range(len(df)):
    if df.iat[i, 1] == 0.1:
        c1.append(df.iat[i, 2])
    elif df.iat[i, 1] == 0.5:
        c2.append(df.iat[i, 2])
    elif df.iat[i, 1] == 0.9:
        c3.append(df.iat[i, 2])

fig = plt.figure(figsize=(30, 4))
ax1 = plt.subplot(221)
plt.hist(c1, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
for p in ax1.patches:
    ax1.annotate(
        str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
    )
ax1.set_title("class1", fontsize=16)
ax1.spines["top"].set_visible(False)  # 刪除外框
ax1.spines["right"].set_visible(False)
ax1.legend(loc="best", fontsize=11)

ax2 = plt.subplot(222)
plt.hist(c2, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
for p in ax2.patches:
    ax2.annotate(
        str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
    )
ax2.set_title("class2", fontsize=16)
ax2.spines["top"].set_visible(False)  # 刪除外框
ax2.spines["right"].set_visible(False)
ax2.legend(loc="best", fontsize=11)

ax3 = plt.subplot(223)
plt.hist(c3, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
for p in ax3.patches:
    ax3.annotate(
        str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
    )
ax3.set_title("class3", fontsize=16)
ax3.spines["top"].set_visible(False)  # 刪除外框
ax3.spines["right"].set_visible(False)
ax3.legend(loc="best", fontsize=11)

fig.tight_layout()
plt.show()

#%%
# calculate std、mean
import numpy as np

c1 = np.array(c1)
c2 = np.array(c2)
c3 = np.array(c3)
print("c1 std:", np.std(c1, ddof=1))
print("c1 mean:", np.mean(c1))
print("c2 std:", np.std(c2, ddof=1))
print("c2 mean:", np.mean(c2))
print("c3 std:", np.std(c3, ddof=1))
print("c3 mean:", np.mean(c3))

#%%
## count gop_fix predict logics
import pandas as pd
import matplotlib.pyplot as plt

c1 = []
c2 = []
c3 = []
df = pd.read_csv("fluency_gop_test.csv")
for i in range(len(df)):
    if df.iat[i, 1] == 0.1:
        c1.append(df.iat[i, 2])
    elif df.iat[i, 1] == 0.5:
        c2.append(df.iat[i, 2])
    elif df.iat[i, 1] == 0.9:
        c3.append(df.iat[i, 2])

fig = plt.figure(figsize=(30, 4))
ax1 = plt.subplot(221)
plt.hist(c1, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
for p in ax1.patches:
    ax1.annotate(
        str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
    )
ax1.set_title("c1", fontsize=16)
ax1.spines["top"].set_visible(False)  # 刪除外框
ax1.spines["right"].set_visible(False)
ax1.legend(loc="best", fontsize=11)

ax2 = plt.subplot(222)
plt.hist(c2, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
for p in ax2.patches:
    ax2.annotate(
        str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
    )
ax2.set_title("c2", fontsize=16)
ax2.spines["top"].set_visible(False)  # 刪除外框
ax2.spines["right"].set_visible(False)
ax2.legend(loc="best", fontsize=11)

ax3 = plt.subplot(223)
plt.hist(c3, bins=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05])
for p in ax3.patches:
    ax3.annotate(
        str(p.get_height()), (p.get_x() * 1.05, p.get_height() * 1.05), fontsize=13
    )
ax3.set_title("c3", fontsize=16)
ax3.spines["top"].set_visible(False)  # 刪除外框
ax3.spines["right"].set_visible(False)
ax3.legend(loc="best", fontsize=11)

fig.tight_layout()
plt.show()
