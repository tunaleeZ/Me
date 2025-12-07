import csv
xs, zs = [], []
with open('wrist_trajectory_xyz_cm.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['hand_X_cm'] == "" or row['hand_Z_cm'] == "":
            continue
        xs.append(float(row['hand_X_cm']))
        zs.append(float(row['hand_Z_cm']))

import matplotlib.pyplot as plt
plt.plot(xs, zs, marker='.')
plt.xlabel('X (cm)  - trái / phải camera')
plt.ylabel('Z (cm)  - gần / xa camera')
plt.axis('equal')
plt.grid(True)
plt.show()

# import csv
# import matplotlib.pyplot as plt

# # 1. Khởi tạo danh sách cho 3 trục
# xs, ys, zs = [], [], []

# # 2. Đọc file CSV
# csv_file = 'wrist_trajectory_xyz_cm.csv'

# try:
#     with open(csv_file, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             # Kiểm tra dữ liệu hợp lệ cho cả 3 trục
#             if (row['hand_X_cm'] == "" or 
#                 row['hand_Z_cm'] == "" or 
#                 row.get('hand_Y_cm', "") == ""): # Dùng .get để tránh lỗi nếu file cũ chưa có Y
#                 continue
            
#             # RealSense: X (Trái/Phải), Y (Lên/Xuống), Z (Gần/Xa)
#             x_val = float(row['hand_X_cm'])
#             y_val = float(row['hand_Y_cm']) 
#             z_val = float(row['hand_Z_cm'])
            
#             xs.append(x_val)
#             # Lưu ý: Trong Computer Vision, trục Y thường hướng xuống dưới (dương).
#             # Để vẽ lên đồ thị cho thuận mắt (lên là dương), ta có thể đảo dấu (-y_val).
#             # Ở đây tôi giữ nguyên, bạn có thể thêm dấu '-' nếu thấy ngược.
#             ys.append(y_val) 
#             zs.append(z_val)

#     # 3. Vẽ đồ thị 3D
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d') # <-- Tạo trục 3D

#     # Vẽ đường quỹ đạo
#     # Tham số: ax.plot(trục_ngang, trục_sâu, trục_đứng)
#     # Ở đây ta map: X -> X đồ thị, Z -> Y đồ thị, Y -> Z đồ thị (độ cao)
#     ax.plot(xs, zs, ys, marker='.', linestyle='-', color='blue', label='Quỹ đạo cổ tay')

#     # Đánh dấu điểm bắt đầu (Màu xanh lá) và kết thúc (Màu đỏ)
#     if xs:
#         ax.scatter(xs[0], zs[0], ys[0], color='green', s=50, label='Bắt đầu')
#         ax.scatter(xs[-1], zs[-1], ys[-1], color='red', s=50, label='Kết thúc')

#     # 4. Thiết lập nhãn trục
#     ax.set_xlabel('X (cm): Trái / Phải')
#     ax.set_ylabel('Z (cm): Gần / Xa (Depth)')
#     ax.set_zlabel('Y (cm): Lên / Xuống')

#     # Đảo ngược trục Z của đồ thị (tương ứng trục Y thực tế) 
#     # vì trong ảnh, tọa độ pixel (0,0) ở góc trên cùng bên trái, nên Y càng lớn càng đi xuống.
#     # Lệnh này giúp đồ thị hiển thị "đúng chiều" không gian thực.
#     ax.invert_zaxis() 

#     ax.legend()
#     plt.title("Quỹ đạo Cổ tay 3D (RealSense)")
#     plt.show()

# except FileNotFoundError:
#     print(f"Không tìm thấy file {csv_file}. Hãy chắc chắn bạn đã ghi log trước.")
# except KeyError as e:
#     print(f"Lỗi dữ liệu: File CSV thiếu cột {e}. Hãy cập nhật code ghi log để thêm cột Y.")