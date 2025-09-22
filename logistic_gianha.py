import numpy as np
import plotly.express as px
import pandas as pd

# 1)  Cột: [Area, Bedrooms, Distance, Price]

data_full = np.array([
    [65,  2,  8,  2.5],
    [80,  3, 12,  3.2],
    [70,  2,  5,  4.1],
    [55,  1,  6,  2.1],
    [90,  3, 15,  3.5],
    [110, 4, 10,  5.9],  # nếu ảnh là 5.8 hãy sửa lại
    [45,  1,  7,  1.9],
    [75,  2,  4,  2.9],
    [100, 3,  4,  3.7],
    [85,  3, 11,  3.8],
    [60,  2, 13,  2.4],
    [120, 4,  6,  8.2],
    [70,  2, 14,  2.7],
    [50,  1,  8,  2.0],
    [95,  3,  5,  6.5],
    [80,  2, 16,  3.0],
    [130, 4,  9,  9.5],
    [65,  2, 11,  2.6],
    [75,  3,  7,  3.4],
    [105, 3, 12,  4.9],
    [58,  2, 15,  2.3],
    [88,  3, 10,  4.2],
    [68,  2,  6,  3.0],
    [92,  3,  8,  4.5],
    [115, 4, 13,  6.0],
    [52,  1,  9,  2.2],
    [78,  3, 14,  3.1],
    [83,  3,  5,  5.0],
    [98,  3, 10,  4.8],
    [72,  2,  7,  3.3],
], dtype=float)

# Target nhị phân theo ngưỡng giá
threshold = 4.0
labels = np.where(data_full[:, 3] >= threshold, 1, -1)  # 1: đắt, -1: rẻ


X = data_full[:, :3]  # (Area, Bedrooms, Distance)
y = labels            # (-1 hoặc 1)


# 2) Đưa vào DataFrame cho tiện & vẽ 3D

df = pd.DataFrame(X, columns=["Area", "Bedrooms", "Distance"])
df["Price"] = data_full[:, 3]
df["Label"] = y

# Vẽ 3D: trục Z là Price, tô màu theo Label
fig = px.scatter_3d(
    df, x="Area", y="Bedrooms", z="Price",
    color=df["Label"].map({1:"Expensive", -1:"Cheap"}),
    symbol="Label", opacity=0.85, title="Housing Data (3D)"
)
fig.update_layout(scene=dict(xaxis_title="Area (m²)",
                             yaxis_title="Bedrooms",
                             zaxis_title="Price"))
fig.show(config={"displaylogo": False, "modeBarButtonsToAdd": ["fullscreen"]})


# 3) Chuẩn hóa dữ liệu (z-score)

X = (X - X.mean(axis=0)) / X.std(axis=0)


# 4) Hàm sigmoid & loss

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss(y_i, z):
    # y_i ∈ {-1, 1}
    return np.log(1 + np.exp(-y_i * z)).item()


# 5) Khởi tạo tham số

num_samples, n_features = X.shape
W = np.random.randn(n_features, 1) * 0.01
b = 0.0


# 6) SGD Training

learning_rate = 0.1
epochs = 200

for epoch in range(epochs):
    loss_epoch = 0.0
    indices = np.random.permutation(num_samples)
    for i in indices:
        x_i = X[i].reshape(-1, 1)   # (3,1)
        y_i = y[i]                  # -1 hoặc 1

        # logit
        z = float(W.T @ x_i + b)

        # loss
        loss_epoch += logistic_loss(y_i, z)

        # gradient theo nhãn ±1
        denom = (1 + np.exp(y_i * z))
        grad_w = -(y_i * x_i) / denom
        grad_b = -(y_i) / denom

        # update
        W -= learning_rate * grad_w
        b -= learning_rate * grad_b

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Avg Loss = {loss_epoch / num_samples:.4f}")


# 7) Hàm dự đoán

def predict(X_input):
    z = X_input @ W + b
    return np.where(z >= 0, 1, -1).flatten()


# 8) Đánh giá

y_pred = predict(X)
acc = np.mean(y_pred == y)

print("\nFinal Weights:", W.flatten(), "Bias:", b)
print("Predicted:", y_pred)
print("True     :", y)
print(f"Final Accuracy: {acc:.4f}")
