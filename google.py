import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('results.csv')

# Làm tròn các giá trị alpha1 và beta tới 3 chữ số thập phân
data['alpha1'] = data['alpha1'].round(3)
data['beta'] = data['beta'].round(3)

# Hiển thị DataFrame trong console
print(data)

# Tạo bảng nhiệt cho total_smooth
plt.figure(figsize=(12, 8))

# Tạo bảng nhiệt cho total_smooth
smooth_heatmap_data = data.pivot_table(index='target_objective', columns=['alpha1', 'beta'], 
                                        values='total_smooth')

# Tạo biểu đồ nhiệt với giá trị được làm tròn tới 3 chữ số thập phân
sns.heatmap(smooth_heatmap_data, cmap='YlGnBu', annot=True, fmt='.3f')

plt.title('Total Smooth by Target Objective, Alpha and Beta')
plt.xlabel('Alpha and Beta')
plt.ylabel('Target Objective')

# Điều chỉnh góc xoay của nhãn trục x
plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Để tránh nhãn bị cắt
plt.show()