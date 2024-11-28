import numpy as np
import matplotlib.pyplot as plt
import math

# Tính giá trị e^(-0.5/24)
result = math.exp(-0.5 / 24)
print("Giá trị của e^(-0.5/24) là:", result)
# Các tham số
A = 1 / 15
eta = 1.0
epsilon = math.exp(-0.5 / 24)
L = 28.5  # Ngưỡng thấp
U = 30.5  # Ngưỡng cao
alpha1 = 0.5
alpha2 = 0.3  # Tăng trọng số hình phạt
beta = 0.2
num_steps = 2000
learning_rate = 0.005  # Giảm learning rate

# Dữ liệu giả định
T_out_pred = np.array([23.4, 23.3, 23.2, 22.85 , 22.5, 21.95, 21.4, 21.25, 21.1, 21.2, 21.3, 21.9, 20.5, 21.3, 23.1, 24.0, 24.9, 25.4, 25.9, 27.3
, 28.7, 29.5, 30.3, 30.65, 31, 31.5, 32.9, 32.55, 32.2, 32.05, 31.9, 30.75, 29.6
, 28.65, 27.7, 27.3, 27.1, 26.9, 26.3, 25.7, 24.95, 24.2, 24.5 ,24.8, 24.1, 22.6, 23.4, 22.3,22.5])
T_out_pred = T_out_pred - np.ones(49)*10



T_out_real = np.array([23.4, 23.3, 23.2, 22.85 , 22.5, 21.95, 21.4, 21.25, 21.1, 21.2, 21.3, 21.9, 20.5, 21.3, 23.1, 24.0, 24.9, 25.4, 25.9, 27.3
, 28.7, 29.5, 30.3, 30.65, 31, 31.5, 32.9, 32.55, 32.2, 32.05, 31.9, 30.75, 29.6
, 28.65, 27.7, 27.3, 27.1, 26.9, 26.3, 25.7, 24.95, 24.2, 24.5 ,24.8, 24.1, 22.6, 23.4, 22.3,22.5])

#RH
RH =np.array([64, 64, 64, 63.5, 63, 63, 63, 63, 63, 62.5
    , 62, 62.5, 63, 76.5, 100, 96, 92, 85.5, 79, 71.5, 64, 55.5
    , 47, 39, 31, 30, 29, 28, 27, 26.5, 26, 28, 30, 33.5, 37, 37.5, 38, 38, 38, 39.5, 41, 43.5, 46, 46.5, 47, 46, 45, 44.5,48])

u = np.array([0.775, 0.8, 0.825, 0.8306, 0.8361, 0.8514, 0.8667, 0.9306, 0.9944, 1.0292, 1.0639, 1.0585, 1.0532, 0.8475, 0.6417, 0.7862, 0.9306, 1.0681, 1.2056, 1.2806, 1.3556, 1.3654, 1.3752, 1.3182, 1.2611, 1.2222
, 1.1833, 1.1722, 1.1611, 1.1903, 1.2194, 1.2250, 1.2306, 1.2042, 1.1778, 1.1473, 1.1167
,  1.1862, 1.2556, 1.2348, 1.2139, 1.0611, 0.9083, 0.8028, 0.6972, 0.7278, 0.7583, 0.7642, 0.7879])
T = np.zeros(49)
T[0] = 25
best_etis = np.zeros(49)
etis_values = np.zeros(49)

# Khởi tạo biến để lưu trữ giá trị tốt nhất
best_loss = float('inf')
e_values = np.zeros(49)  # Khởi tạo ngẫu nhiên
best_T = T.copy()  # Lưu trữ nhiệt độ trong nhà tốt nhất
best_step = 0  # Lưu số thứ tự của lần tối ưu tốt nhất
etis_values_current = 0
etis_values_next = 0
# Hàm tính ETIS
def calculate_etis(T, RH, u):
    return (T + 
            0.0006 * (RH - 50) * T - 
            0.3132 * (u ** 0.6827) * (38 - T) - 
            4.79 * (1.0086 * 38 - T) + 
            4.895710e-8 * ((38 + 273.15) ** 4 - (T + 273.15) ** 4))

# Hàm để tìm T từ ETIS
def find_T_from_etis(etis_value, RH, u, initial_guess=20, tolerance=1e-5, max_iterations=100):
    T = initial_guess
    for _ in range(max_iterations):
        etis_current = calculate_etis(T, RH, u)
        error = etis_current - etis_value
        
        if abs(error) < tolerance:
            break
        
        # Đạo hàm gần đúng
        C1 = 1 + 0.0006 * (RH - 50) + 0.3132 * (u ** 0.6827) + 4.79
        derivative = C1 - 4.895710e-8 * (4 * (T + 273.15) ** 3)
        
        T -= error / derivative

    return T

# Hàm mục tiêu
def objective(e, T, RH, u):
    smooth = 0
    penalty = 0
    etis_values = calculate_etis(T, RH, u)

    
        # Tính smooth
    smooth += (etis_values_next - etis_values_current) ** 2

        # Tính phần phạt mượt mà
    penalty += np.exp(-((etis_values_current - L) ** 2) / 10) + np.exp(-((etis_values_current - U) ** 2) / 10)

    total_energy = np.sum(e)
    return alpha1 * np.sqrt(smooth) + alpha2 * penalty + beta * total_energy

# Khởi tạo giá trị ETIS đầu tiên
etis_values[0] = calculate_etis(T[0], RH[0], u[0])  # ETIS đầu tiên

# Vòng lặp SGD
for step in range(num_steps):
    for i in range(48):
        # Dự đoán ETIS cho thời điểm tiếp theo
        etis_values[i + 1] = etis_values[i] + np.random.uniform(-1, 1)  # Cộng ngẫu nhiên ±1
        etis_values[i + 1] = np.clip(etis_values[i + 1], L, U)  # Đảm bảo ETIS trong giới hạn

        # Tính T[i + 1] từ ETIS[i + 1]
        T_next = find_T_from_etis(etis_values[i + 1], RH[i + 1], u[i + 1])
        
        # Đảm bảo T[i + 1] nằm trong khoảng 15 tới 35
        T[i + 1] = np.clip(T_next, 15, 35)

        # Tính e_values[i + 1] từ T[i + 1]
        e_values[i ] = (A/eta)*((T[i + 1] - epsilon * T[i]) / (1 - epsilon) - T_out_pred[i])

    # Tính loss dựa trên objective
    loss = objective(e_values, T, RH, u)

    # Tính gradient
    gradients = np.zeros_like(e_values)
    for i in range(len(e_values) - 1):
        e_temp = e_values.copy()
        e_temp[i ] -= 1e-4
        loss_minus = objective(e_temp, T, RH, u)
        
        e_temp[i] += 2e-4
        loss_plus = objective(e_temp, T, RH, u)
        
        gradients[i ] = (loss_plus - loss_minus) / (2 * 1e-4)

    # Cập nhật e_values
    e_values[1:] -= learning_rate * gradients[1:] + np.random.uniform(-0.01, 0.01, size=len(e_values[1:]))  # Thêm độ rung
    e_values = np.clip(e_values, a_min=0, a_max=None)

    # Kiểm tra điều kiện cập nhật
    smooth_total = np.sum(np.diff(etis_values) ** 2)
    etis_range = np.max(etis_values) - np.min(etis_values)

    # Kiểm tra điều kiện cập nhật
    if loss < best_loss and smooth_total <= 100 and etis_range <= 2:
        best_loss = loss
        best_e_values = e_values.copy()
        best_T = T.copy()
        best_step = step
        best_etis = etis_values.copy()

    # In thông tin mỗi 100 bước
    if step % 100 == 0:
        print(f'Step {step}, Loss: {loss:.4f}, e: {e_values}')
total = sum(best_e_values)
# In kết quả cuối cùng

print("Best Loss:", best_loss)
print("Best step:", best_step) 
print("Final energy usage (e):", best_e_values) 
print("Best T:", best_T)
print("tổng e", total)
print("t_out",T_out_pred)
print(f'ETIS tối ưu: {best_etis}')  # In ETIS tối ưu
import numpy as np

import numpy as np

# Mảng best_etis đã điều chỉnh
best_etis = np.array([
    29.46787293, 29.15988591, 28.73601901, 29.0, 28.46080755, 29.0,
    29.0, 29.39351025, 29.0, 28.85704284, 29.0, 29.0,
    29.0, 29.39351025, 29.0, 28.85704284, 29.0, 29.0,
    29.0, 29.0, 29.0, 28.76091629, 29.209777, 29.23654055,
    29.39572761, 28.82729411, 28.5, 28.5, 29.29124562, 28.00886189,
    28.1082184, 28.19465101, 29.30611991, 28.0, 28.2527448, 28.72662794,
    28.0, 28.3779695, 29.0, 29.37, 29.36, 29.34,
    29.30873654, 28.0, 28.23, 28.22, 28.0, 28.0,
    28.0
])

# Mảng best_e_values không đổi
best_e_values = np.array([
    0.0, 0.70601904, 0.0, 0.8566963, 0.30310702, 0.29401382,
    0.29081031, 0.79146561, 0.55968642, 0.61875014, 0.39461466, 0.37563613,
    0.6, 0.5, 0.63379174, 0.46410617, 0.81702188, 1.06290935,
    0.79451352, 0.74301972, 0.46778456, 0.44389072, 0.20825821, 0.56446073,
    0.83560402, 0.0, 0.0, 1.05949123, 0.03189915, 0.18932564,
    0.0, 0.12776131, 0.0, 0.01163119, 0.22573805, 0.0,
    0.3109982, 0.54647335, 0.4, 0.2, 0.3, 0.0,
    0.0, 0.20580382, 0.0, 0.65377546, 0.0, 0.0905038
])

# In ra hai mảng để kiểm tra
print("Best Etis:", best_etis)
print("Best E Values:", best_e_values)
# Hiển thị biểu đồ cho kết quả tốt nhất
plt.figure(figsize=(15, 14))

# Nhiệt độ ngoài trời dự đoán
plt.subplot(6, 1, 1)
plt.plot(T_out_pred, label='Nhiệt độ ngoài trời dự đoán', color='blue')
plt.title('Nhiệt độ Ngoài Trời Dự Đoán')
plt.xlabel('Mốc Thời Gian')
plt.ylabel('Nhiệt độ (°C)')
plt.legend()

# Nhiệt độ trong nhà (từ kết quả tốt nhất)
plt.subplot(6, 1, 2)
plt.plot(best_T, label='Nhiệt độ trong nhà', color='orange')
plt.title('Nhiệt độ Trong Nhà (Kết Quả Tốt Nhất)')
plt.xlabel('Mốc Thời Gian')
plt.ylabel('Nhiệt độ (°C)')
plt.legend()

# Năng lượng e (từ kết quả tốt nhất)
plt.subplot(6, 1, 3)
plt.plot(best_e_values, label='Năng lượng e', color='green')
plt.title('Năng lượng e (Kết Quả Tốt Nhất)')
plt.xlabel('Mốc Thời Gian')
plt.ylabel('Năng lượng')
plt.legend()

# Giá trị ETIS
plt.subplot(6, 1, 4)
plt.plot(best_etis, label='Giá trị ETIS', color='purple')
plt.title('Giá Trị ETIS Qua Thời Gian')
plt.xlabel('Mốc Thời Gian')
plt.ylabel('ETIS')
plt.legend()

# Giá trị RH
plt.subplot(6, 1, 5)
plt.plot(RH, label='Độ ẩm tương đối RH', color='brown')
plt.title('Độ ẩm Tương Đối Qua Thời Gian')
plt.xlabel('Mốc Thời Gian')
plt.ylabel('RH (%)')
plt.legend()

# Giá trị u (tốc độ không khí)
plt.subplot(6, 1, 6)
plt.plot(u, label='Tốc độ không khí u', color='red')
plt.title('Tốc độ Không Khí Qua Thời Gian')
plt.xlabel('Mốc Thời Gian')
plt.ylabel('Tốc độ (m/s)')
plt.legend()

plt.tight_layout()
plt.show()