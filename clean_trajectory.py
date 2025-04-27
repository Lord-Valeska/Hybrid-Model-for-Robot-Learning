import pandas as pd

# 1. load
xyz = pd.read_csv('data/cleaned_xyz_test.csv')  
enc = pd.read_csv('data/received_data_test.csv', header=None,
                  names=['traj','encoder1','encoder2','encoder3'])
cmd = pd.read_csv('data/command_test.csv',   header=None,
                  names=['traj','angle1','angle2','angle3'])

# 2. count original trajectories
n_traj_xyz = xyz['traj'].nunique()

# 3. restrict enc & cmd to only trajs in xyz
valid = xyz['traj'].unique()
enc = enc[enc['traj'].isin(valid)]
cmd = cmd[cmd['traj'].isin(valid)]

# 4. add per-traj row-indices
xyz['idx'] = xyz.groupby('traj').cumcount()
enc['idx'] = enc.groupby('traj').cumcount()
cmd['idx'] = cmd.groupby('traj').cumcount()

# 5. merge on traj + idx
merged = (
    xyz
    .merge(enc, on=['traj','idx'], how='inner')
    .merge(cmd, on=['traj','idx'], how='inner')
)

# 6. compute metrics
n_rows = len(merged)
n_traj = merged['traj'].nunique()

# 7. remap traj → 0,1,... in order of appearance
merged['traj'] = pd.factorize(merged['traj'])[0]

# 8. drop helper idx and reorder
merged = merged.drop(columns='idx')[
    ['traj',
     'encoder1','encoder2','encoder3',
     'X','Y','Z',
     'angle1','angle2','angle3']
]

# 9. save
merged.to_csv('data/merged_data_test.csv', index=False)

# 10. report
print(f"Original cleaned_xyz_1.csv contained {n_traj_xyz} unique trajectories.")
print(f"Saved {n_rows} rows across {n_traj} unique trajectories (renumbered 0–{n_traj-1}) to merged_data.csv")
