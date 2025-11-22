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
