import numpy as np
import matplotlib.pyplot as plt
import math

# Tham số và dữ liệu ban đầu
A = 1 / 15
eta = 1.0
epsilon = math.exp(-0.5 / 24)
L = 28.5  # Ngưỡng thấp cho ETIS
U = 35  # Ngưỡng cao cho ETIS
alpha1 = 0.5
beta = 0.5
target_objective = 87

# Dữ liệu đầu vào
T_out_pred = np.array([23.4, 23.3, 23.2,24.85 , 21.5, 22.95, 21.4, 21.25, 21.1, 21.2, 21.3, 21.9, 20.5, 21.3, 23.1, 24.0, 24.9, 25.4, 25.9, 27.3, 28.7, 29.5, 30.3, 30.65, 31, 31.95, 32.9, 32.55, 32.2, 32.05, 31.9, 30.75, 29.6, 28.65, 27.7, 27.3, 27.1, 26.9, 26.3, 25.7, 24.95, 24.2, 24.5 ,24.8, 24.1, 22.6, 23.4, 22.3])
RH = np.array([64, 63, 64, 64, 63.5, 63, 63, 63, 63, 63, 62.5, 62, 62.5, 63, 76.5, 100, 96, 92, 85.5, 79, 71.5, 64, 55.5, 47, 39, 31, 30, 29, 28, 27, 26.5, 26, 28, 30, 33.5, 37, 37.5, 38, 38, 38, 39.5, 41, 43.5, 46, 46.5, 47, 46, 45, 44.5])
u = np.array([0.775, 0.78, 0.8, 0.825, 0.8306, 0.8361, 0.8514, 0.8667, 0.9306, 0.9944, 1.0292, 1.0639, 1.0585, 1.0532, 0.8475, 0.6417, 0.7862, 0.9306, 1.0681, 1.2056, 1.2806, 1.3556, 1.3654, 1.3752, 1.3182, 1.2611, 1.2222, 1.1833, 1.1722, 1.1611, 1.1903, 1.2194, 1.2250, 1.2306, 1.2042, 1.1778, 1.1473, 1.1167, 1.1862, 1.2556, 1.2348, 1.2139, 1.0611, 0.9083, 0.8028, 0.6972, 0.7278, 0.7583, 0.7642])
loss_minus = np.zeros(49)
T_out_pred = T_out_pred -10 
loss_plus = np.zeros(49)
# Hàm tính ETIS
def calculate_etis(T, RH, u):
    return (T + 
            0.0006 * (RH - 50) * T - 
            0.3132 * (u ** 0.6827) * (38 - T) - 
            4.79 * (1.0086 * 38 - T) + 
            4.895710e-8 * ((38 + 273.15) ** 4 - (T + 273.15) ** 4))

# Hàm tính nhu cầu năng lượng
def calculate_energy_demand(T, RH, u, i):
    energy_demand = (A / eta) * ((T[i] - epsilon * T[i-1]) / (1 - epsilon) - T_out_pred[i-1])
    if energy_demand < 0:
        energy_demand = 0
        T[i] = 0.3* T[i-1] + (1 - 0.3) * T_out_pred[i-1]
    return energy_demand

# Hàm tính giá trị mục tiêu
def objective(T, RH, u):
    etis_values = calculate_etis(T, RH, u)
    energy_demand_values = np.zeros(len(T))
    for i in range(1, len(T)):
        energy_demand_values[i] = calculate_energy_demand(T, RH, u, i)

    smooth_values = np.zeros(len(T))
    for i in range(1, len(etis_values)):
        smooth_values[i] = (etis_values[i] - etis_values[i - 1]) ** 2

    E_min, E_max = np.min(energy_demand_values), np.max(energy_demand_values)
    S_min, S_max = np.min(smooth_values), np.max(smooth_values)

    percent_E = 100 * (1 - (energy_demand_values - E_min) / (E_max - E_min)) if E_min != E_max else np.full(len(T), 100)
    percent_S = 100 * (1 - (smooth_values - S_min) / (S_max - S_min)) if S_min != S_max else np.full(len(T), 100)

    objective_values = alpha1 * percent_S + beta * percent_E
    return objective_values, percent_S, percent_E

# Hàm tối ưu hóa từng thành phần với ràng buộc ETIS liên tiếp
def optimize_component_per_step_with_etis_constraint(func, T_initial, RH, u, T_tout, learning_rate=0.005, num_steps=100):
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
# Tối ưu hóa nhiệt độ với mục tiêu đạt 90%
def optimize_temperatures_with_objective(RH, u, T_out_pred):
    T_initial = np.full(len(RH), 22.0)


    # Tính toán nhiệt độ tối ưu ban đầu
    T_opt_energy = optimize_component_per_step_with_etis_constraint(calculate_energy_demand, T_initial, RH, u, T_out_pred)
    T_smoothness_values = optimize_component_per_step_with_etis_constraint(lambda T, RH, u, i: np.sqrt(np.sum(np.diff(calculate_etis(T, RH, u)) ** 2)), T_initial, RH, u, T_out_pred)
    E_min = np.zeros(len(T_opt_energy))
    for i in range(1, len(T_opt_energy)):
        E_min[i] = calculate_energy_demand(T_opt_energy, RH, u, i)
    #E_min = np.min(calculate_energy_demand(T_opt_energy, RH, u, np.arange(len(T_opt_energy))))
    E_max = 10
    S_min = np.min(np.diff(calculate_etis(T_smoothness_values,RH,u))**2)
    S_max = 5

    # Vòng lặp để điều chỉnh giá trị T
    # Vòng lặp để điều chỉnh giá trị T
    T_opt_adjusted = T_initial.copy()  # Khởi tạo T_opt_adjusted từ T_initial

    for i in range(len(T_opt_adjusted)):
      best_value = None
      best_objective = float('inf')  # Khởi tạo giá trị tốt nhất lớn vô cực

    # Thử từng giá trị từ 15 đến 35 với bước 0.05
      for T_value in np.arange(15, 35.05, 0.05):
        T_opt_adjusted[i] = T_value  # Cập nhật giá trị T[i]

        # Tính toán giá trị ETIS cho T_opt_adjusted
        etis_value = calculate_etis(T_opt_adjusted, RH, u)[i]

        # Kiểm tra xem ETIS có vượt ngưỡng không
        if L <= etis_value <= U:
            # Tính toán các giá trị mục tiêu
            objective_values, percent_S, percent_E = objective(T_opt_adjusted, RH, u)
             # 90% của giá trị mục tiêu
            
            # Tính độ gần gũi với mục tiêu
            objective_difference = abs(target_objective - objective_values[i])

            # Nếu giá trị mục tiêu này tốt hơn giá trị tốt nhất hiện tại
            if objective_difference < best_objective:
                best_objective = objective_difference
                best_value = T_value  # Lưu lại giá trị T tốt nhất

    # Cập nhật giá trị T[i] với giá trị tốt nhất tìm được
      if best_value is not None:  # Kiểm tra nếu có giá trị tốt nhất hợp lệ
        T_opt_adjusted[i] = best_value

# Trả về giá trị T đã tối ưu
    return T_opt_adjusted, objective_values, target_objective
'''import pandas as pd

# Các tham số để thay đổi
alpha1_values = np.arange(0, 1.1, 0.1)  # Từ 0 tới 1 với bước 0.1
target_objective_values = np.arange(0, 101, 10)  # Từ 0 tới 100 với bước 10

# Danh sách lưu trữ kết quả
results = []

# Vòng lặp qua các giá trị alpha1 và target_objective
for alpha1 in alpha1_values:
    for target_objective in target_objective_values:
        beta = 1 - alpha1  # Cập nhật beta

        # Tối ưu hóa nhiệt độ
        T_opt_final, objective_values, _= optimize_temperatures_with_objective(RH, u, T_out_pred)

        # Tính giá trị năng lượng và ETIS cho đồ thị
        energy_demand_values = np.zeros(len(T_out_pred))
        for i in range(1, len(T_out_pred)):
            energy_demand_values[i] = calculate_energy_demand(T_opt_final, RH, u, i)
        total_energy_demand = np.sum(energy_demand_values)
        etis_values = calculate_etis(T_opt_final, RH, u)
        smooth_values = np.zeros(len(etis_values))

        for i in range(1, len(etis_values)):
            smooth_values[i] = (etis_values[i] - etis_values[i - 1]) ** 2
        total_smooth = np.sum(smooth_values)
        print("Tổng nhu cầu năng lượng:", total_energy_demand)
        print("Tổng giá trị smooth:", total_smooth)
        print("alpha1:", alpha1)
        print("Beta:", beta)
        print("Target_object:", target_objective)

        # Lưu kết quả
        results.append({
            'alpha1': alpha1,
            'beta': beta,
            'target_objective': target_objective,
            'total_energy_demand': total_energy_demand,
            'total_smooth': total_smooth
        })

# Tạo DataFrame từ kết quả
results_df = pd.DataFrame(results)
results_df.to_csv('results2.csv', index=False)
# Hiển thị bảng kết quả
print(results_df)'''
T_energy_opt = 0
# Tính toán và tối ưu hóa nhiệt độ
T_opt_final, objective_values,target_objective = optimize_temperatures_with_objective(RH, u, T_out_pred)

# Tính giá trị năng lượng và ETIS cho đồ thị
energy_demand_values = np.zeros(len(T_out_pred))
for i in range(1, len(T_out_pred)):
    energy_demand_values[i] = calculate_energy_demand(T_opt_final, RH, u, i)
total_energy_demand = np.sum(energy_demand_values)
etis_values = calculate_etis(T_opt_final, RH, u)
smooth_values = np.zeros(len(etis_values))

for i in range(1, len(etis_values)):
    smooth_values[i] = (etis_values[i] - etis_values[i - 1]) ** 2
print("Giá trị smooth:", smooth_values)

total_smooth = np.sum(smooth_values)
print("Tổng nhu cầu năng lượng:", total_energy_demand)
print("Tổng giá trị smooth:", total_smooth)

# Vẽ kết quả
def plot_results(T_out_pred, T_opt_final, energy_demand_values, etis_values,objective_values,alpha1,beta,target_objective, total_energy_demand, total_smooth):
    plt.figure(figsize=(12, 8))

    # Vẽ nhiệt độ đầu ra và tối ưu
    plt.subplot(5, 1, 1)
    plt.plot(T_out_pred, label='T_out_pred', color='blue')
    plt.plot(T_opt_final, label='T_opt_final', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    # Vẽ nhu cầu năng lượng
    plt.subplot(5, 1, 2)
    plt.plot(energy_demand_values, label='Energy Demand', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Energy Demand (kWh)')
    plt.legend()

    # Vẽ ETIS
    plt.subplot(5, 1, 3)
    plt.plot(etis_values, label='ETIS', color='purple')
    plt.xlabel('Time Step')
    plt.ylabel('ETIS')
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(objective_values, label='OBJ', color='red')
    plt.xlabel('Time Step')
    plt.ylabel('Obj')
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.text(0.3, 0.6, f'Alpha: {alpha1}', transform=plt.gca().transAxes, fontsize=12, ha='center', color='black')
    plt.text(0.3, 0.4, f'Beta: {beta}', transform=plt.gca().transAxes, fontsize=12, ha='center', color='black')
    plt.text(0.3, 0.2, f'%Obj: {target_objective}', transform=plt.gca().transAxes, fontsize=12, ha='center', color='black')
    plt.text(0.7, 0.4, f'Sum_e: {total_energy_demand}', transform=plt.gca().transAxes, fontsize=12, ha='center', color='black')
    plt.text(0.7, 0.2, f'Sum_smooth: {total_smooth}', transform=plt.gca().transAxes, fontsize=12, ha='center', color='black')



    plt.tight_layout()
    plt.show()
print("Nhiệt độ ngaoif trời:", T_out_pred)
print("Nhiệt độ tối ưu :", T_opt_final)
print("Nhu cầu năng lượng:", energy_demand_values)
print("Giá trị ETIS:", etis_values)

# Vẽ kết quả
plot_results(T_out_pred, T_opt_final, energy_demand_values, etis_values,objective_values, alpha1, beta,target_objective,total_energy_demand,total_smooth)