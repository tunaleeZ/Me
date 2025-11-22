import csv
import matplotlib.pyplot as plt

xs_left, zs_left = [], []
xs_right, zs_right = [], []

with open('wrist_trajectory_xyz_cm.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # body trái
        if row['left_X_cm'] != "" and row['left_Z_cm'] != "":
            xs_left.append(float(row['left_X_cm']))
            zs_left.append(float(row['left_Z_cm']))

        # body phải
        if row['right_X_cm'] != "" and row['right_Z_cm'] != "":
            xs_right.append(float(row['right_X_cm']))
            zs_right.append(float(row['right_Z_cm']))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 5))

# Body trái
axL.plot(xs_left, zs_left, marker='.')
axL.set_title('Body trái')
axL.set_xlabel('X (cm)')
axL.set_ylabel('Z (cm)')
axL.axis('equal')
axL.grid(True)

# Body phải
axR.plot(xs_right, zs_right, marker='.')
axR.set_title('Body phải')
axR.set_xlabel('X (cm)')
axR.set_ylabel('Z (cm)')
axR.axis('equal')
axR.grid(True)

plt.tight_layout()
plt.show()
