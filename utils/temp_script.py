import os
cmd = []
for amt in [60, 180, 300, 600]:
    cmd.append(f"python3 ML_Model/ANN_06.py {amt}")
    cmd.append(f"rclone copyto ~/dev/SRA2023/orbitsim/saved_models/ANN_06_{amt}_VP=250 box:orbitsim/saved_models/ANN_06_{amt}_VP=250")


cmd = ";".join(cmd)

print(f"CMD: '{cmd}'")


os.system(cmd)