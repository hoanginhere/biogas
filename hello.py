import numpy as np
import matplotlib.pyplot as plt
import math

# Tham số và dữ liệu ban đầu
A = 1 / 15
eta = 1.0
epsilon = math.exp(-0.5 / 24)
L = 28.5  # Ngưỡng thấp cho ETIS
U = 30.5  # Ngưỡng cao cho ETIS

# Dữ liệu đầu vào
T_out_pred = np.zeros(49)

# 12 phần tử đầu tiên từ 20 đến 22
T_out_pred[:12] = np.linspace(20, 22, 12)

# Tăng dần từ 22 đến 26
T_out_pred[12:25] = np.linspace(22, 26, 13)

# Giảm dần từ 26 về 20
T_out_pred[25:] = np.linspace(26, 20, 24)

RH = np.array([64, 64, 64, 63.5, 63, 63, 63, 63, 63, 62.5, 62, 62.5, 63, 76.5, 100, 96, 92, 85.5, 79, 71.5, 64, 55.5,
               47, 39, 31, 30, 29, 28, 27, 26.5, 26, 28, 30, 33.5, 37, 37.5, 38, 38, 38, 39.5, 41, 43.5, 46, 46.5, 
               47, 46, 45, 44.5, 48])

u = np.array([0.775, 0.8, 0.825, 0.8306, 0.8361, 0.8514, 0.8667, 0.9306, 0.9944, 1.0292, 1.0639, 1.0585, 1.0532, 0.8475, 
              0.6417, 0.7862, 0.9306, 1.0681, 1.2056, 1.2806, 1.3556, 1.3654, 1.3752, 1.3182, 1.2611, 1.2222, 1.1833, 
              1.1722, 1.1611, 1.1903, 1.2194, 1.2250, 1.2306, 1.2042, 1.1778, 1.1473, 1.1167, 1.1862, 1.2556, 1.2348, 
              1.2139, 1.0611, 0.9083, 0.8028, 0.6972, 0.7278, 0.7583, 0.7642, 0.7879])
        # Bổ sung đầy đủ giá trị

loss_minus = np.zeros(49)
loss_plus = np.zeros(49)
best_loss_e= np.zeros(49)
best_loss_smooth= np.zeros(49)
best_loss_pen = np.zeros(49)
# Hàm tính ETIS
def calculate_etis(T, RH, u):
    return (T + 
            0.0006 * (RH - 50) * T - 
            0.3132 * (u ** 0.6827) * (38 - T) - 
            4.79 * (1.0086 * 38 - T) + 
            4.895710e-8 * ((38 + 273.15) ** 4 - (T + 273.15) ** 4))

# Hàm tính nhu cầu năng lượng (cập nhật giá trị đơn tại mỗi thời điểm)
def calculate_energy_demand(T, RH, u, i):
    energy_demand = (A / eta) * ((T[i] - epsilon * T[i-1]) / (1 - epsilon) - T_out_pred[i-1])
    if energy_demand < 0:
        energy_demand = 0
        T[i] = 0.8 * T[i-1] + (1 - 0.8) * T_out_pred[i-1]
    return energy_demand

# Hàm tối ưu hóa từng thành phần
def optimize_component_per_step(func, T_initial, RH, u, T_tout, learning_rate=0.005, num_steps=100):
    T_opt = T_initial.copy()
    T_best = T_initial.copy()  # Biến lưu trữ giá trị tối ưu nhất
    loss_best = float('inf')   # Khởi tạo độ lỗi tối ưu nhất là vô cùng lớn

    for step in range(num_steps):
        for i in range(1, len(T_opt)):
            # Sao chép nhiệt độ hiện tại để tính toán gradient
            T_temp = T_opt.copy()
            T_temp[i] -= 1e-4
            loss_minus[i] = func(T_temp, RH, u, i)

            T_temp[i] += 2e-4
            loss_plus[i] = func(T_temp, RH, u, i)

            # Tính gradient và cập nhật T_opt
            gradient = (loss_plus - loss_minus) / (2 * 1e-4)
            T_opt[i] -= learning_rate * gradient[i]

            # Tính toán ETIS và điều chỉnh nếu cần thiết
            etis_value = calculate_etis(T_opt, RH, u)[i]
            if etis_value < L or etis_value > U:
                adjustment = L - etis_value if etis_value < L else U - etis_value
                T_opt[i] += adjustment * 0.1
                T_opt[i] = np.clip(T_opt[i], 15, 35)

            # Điều chỉnh năng lượng, chỉ cho phép tăng nhiệt độ khi cần
            if i > 0:
                energy_value = calculate_energy_demand(T_opt, RH, u, i)

        # Tính toán độ lỗi tổng thể sau mỗi bước
        current_loss = func(T_opt, RH, u, i)
        if current_loss < loss_best:
            loss_best = current_loss
            T_best = T_opt.copy()  # Lưu trữ T_opt hiện tại là tốt nhất

    return T_best


def fitness_function(T, RH, u):
    etis_values = calculate_etis(T, RH, u)
    energy_demand_values = np.zeros(len(T))
    for i in range(1, len(T)):
        energy_demand_values[i] = calculate_energy_demand(T, RH, u, i)
    penalty = np.sum(np.maximum(L - etis_values, 0)) + np.sum(np.maximum(etis_values - U, 0))
    return np.sum(np.square(energy_demand_values)) + penalty

T = np.zeros(49)

T_energy_opt = optimize_component_per_step(calculate_energy_demand, T, RH, u, T_out_pred)
T_smooth_opt = optimize_component_per_step(lambda T, RH, u, i: np.sqrt(np.sum(np.diff(calculate_etis(T, RH, u)) ** 2)), T, RH, u, T_out_pred)
T_penalty_opt = optimize_component_per_step(lambda T, RH, u, i: np.exp(-((calculate_etis(T, RH, u)[i] - L) ** 2) / 10) + np.exp(-((calculate_etis(T, RH, u)[i] - U) ** 2) / 10), T, RH, u, T_out_pred)

w_energy = 0.8
w_smooth = 0.1
w_penalty = 0.1

lower_bound = np.min([T_energy_opt, T_smooth_opt, T_penalty_opt])
upper_bound = np.max([T_energy_opt, T_smooth_opt, T_penalty_opt])

step_size = 0.1
best_fitness = float('inf')
best_temperature = None

for t in np.arange(lower_bound, upper_bound, step_size):
    T_test = np.full_like(T, t)
    current_fitness = fitness_function(T_test, RH, u)
    
    if current_fitness < best_fitness:
        best_fitness = current_fitness
        best_temperature = T_test

energy_demand = np.zeros(len(best_temperature) - 1)
for i in range(1, len(best_temperature)):
    energy_demand[i-1] = calculate_energy_demand(best_temperature, RH, u, i)

etis_values = calculate_etis(best_temperature, RH, u)

print(f"Balanced temperature: {best_temperature}")
print(f"Energy demand for each time: {energy_demand}")
print(f"bestTe: {T_energy_opt}")
print(f"bestTetis: {T_smooth_opt}")
print(f"bestTpen: {T_penalty_opt}")



plt.subplot(3, 1, 1)
plt.plot(best_temperature, label='T cân bằng')
plt.xlabel('Thời điểm')
plt.ylabel('Nhiệt độ')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(energy_demand, label='Nhu cầu năng lượng')
plt.xlabel('Thời điểm')
plt.ylabel('Năng lượng')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(etis_values, label='etis')
plt.xlabel('Thời điểm')
plt.ylabel('etis')
plt.legend()
plt.tight_layout()
plt.show()
