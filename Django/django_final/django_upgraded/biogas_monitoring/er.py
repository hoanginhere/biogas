import sqlite3
import matplotlib.pyplot as plt

# Kết nối tới cơ sở dữ liệu SQLite
db_file = 'db.sqlite3'  # Thay thế bằng tên tệp SQLite của bạn
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Tạo hình vẽ
fig, ax = plt.subplots(figsize=(10, 6))

# Lấy danh sách các bảng
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Vẽ các bảng
y = 0
for table in tables:
    table_name = table[0]
    ax.text(0.5, y, table_name, horizontalalignment='center', fontsize=12, fontweight='bold')
    y -= 1

    # Lấy thông tin về các cột
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        col_name = col[1]
        ax.text(0.5, y, f'  - {col_name}', horizontalalignment='center', fontsize=10)
        y -= 0.5

    # Lấy thông tin về khóa ngoại
    cursor.execute(f"PRAGMA foreign_key_list({table_name});")
    foreign_keys = cursor.fetchall()
    for fk in foreign_keys:
        referenced_table = fk[3]  # Bảng được tham chiếu
        ax.text(0.5, y, f'FK -> {referenced_table}', horizontalalignment='center', fontsize=10, color='red')
        y -= 0.5

# Tùy chỉnh hình vẽ
ax.axis('off')
plt.title('Database Schema', fontsize=14, fontweight='bold')
plt.tight_layout()

# Hiển thị hình vẽ
plt.show()

# Đóng kết nối
conn.close()