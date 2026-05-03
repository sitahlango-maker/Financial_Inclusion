import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ===============================
# 1. ALIGN FEATURES SAFELY
# ===============================
expected_features = model_pooled.feature_names_in_

X_test_aligned = X_test.copy()

for col in expected_features:
    if col not in X_test_aligned.columns:
        X_test_aligned[col] = 0

X_test_aligned = X_test_aligned[expected_features]

# ===============================
# 2. POOLED MODEL
# ===============================
pooled_probs = model_pooled.predict_proba(X_test_aligned)[:, 1]
pooled_pred = (pooled_probs >= 0.5).astype(int)

# ===============================
# 3. EXPERT ROUTING MODEL
# ===============================
expert_probs = np.zeros(len(X_test_aligned))
expert_pred = np.zeros(len(X_test_aligned))

for i in range(len(X_test_aligned)):
    row = X_test_aligned.iloc[[i]]

    country = gating_model.predict(row)[0]

    if country in experts:
        prob = experts[country].predict_proba(row)[0, 1]
    else:
        prob = pooled_probs[i]

    expert_probs[i] = prob
    expert_pred[i] = int(prob >= 0.5)

# ===============================
# 4. METRICS FUNCTION
# ===============================
def cm_metrics(y_true, y_pred, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Model": name,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

# ===============================
# 5. RESULTS TABLE
# ===============================
results = [
    cm_metrics(y_test, pooled_pred, "Pooled Model"),
    cm_metrics(y_test, expert_pred, "Expert Routing")
]

df_cm = pd.DataFrame(results).round(4)

print("\n=== CONFUSION MATRIX SUMMARY ===\n")
print(df_cm.to_string(index=False))

# ===============================
# 6. CONFUSION MATRIX PLOT (POOLED)
# ===============================
tn, fp, fn, tp = confusion_matrix(y_test, pooled_pred).ravel()

cm = np.array([[tn, fp],
               [fn, tp]])

fig, ax = plt.subplots(figsize=(5, 4))

# FORCE CLEAN WHITE BACKGROUND
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

im = ax.imshow(cm, cmap="Blues")

ax.set_title("Confusion Matrix - Pooled Model", color="black")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["No Access", "Access"], color="black")
ax.set_yticklabels(["No Access", "Access"], color="black")

# annotations
for i in range(2):
    for j in range(2):
        ax.text(
            j, i, cm[i, j],
            ha="center",
            va="center",
            color="black",
            fontsize=12
        )

plt.colorbar(im)
plt.tight_layout()
plt.show()

# ===============================
# 7. EXPORT TABLE IMAGE (WHITE BACKGROUND FIXED)
# ===============================
fig2, ax2 = plt.subplots(figsize=(10, 3))

fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

ax2.axis("off")

table = ax2.table(
    cellText=df_cm.values,
    colLabels=df_cm.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# enforce visibility
for key, cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_facecolor("white")
    cell.get_text().set_color("black")

plt.title(
    "Confusion Matrix Performance Summary",
    fontsize=14,
    color="black"
)

plt.savefig(
    "confusion_matrix_summary.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("Saved: confusion_matrix_summary.png")
