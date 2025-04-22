# validate_pcc.py
import pandas as pd
from models import solve_end_effector, angle_to_length, d

def validate_pcc(csv_path: str):
    # 1. 读取 CSV，假设无 header
    df = pd.read_csv(csv_path, header=None)
    df.columns = ['traj', 'enc1', 'enc2', 'enc3']
    
    failures = []   # 存储失败的 (traj, row_index, 错误信息)
    results = []    # 存储成功的 (traj, row_index, x, y, z)
    
    # 2. 按轨迹分组
    for traj_id, group in df.groupby('traj'):
        group = group.reset_index(drop=True)
        # 第一行：实际绝对角度
        init_angles = group.loc[0, ['enc1','enc2','enc3']].values.astype(float)
        
        # 遍历该轨迹所有点
        for idx, row in group.iterrows():
            if idx == 0:
                angles = init_angles
            else:
                increments = row[['enc1','enc2','enc3']].values.astype(float)
                angles = init_angles + increments
            
            # 3. 角度→长度
            lengths = [angle_to_length(a) for a in angles]
            
            # 4. 反算末端位姿
            try:
                x, y, z, *_ = solve_end_effector(lengths, d)
                results.append((traj_id, idx, x, y, z))
            except Exception as e:
                failures.append((traj_id, idx, str(e)))
    
    # 5. 汇总报告
    if failures:
        print("以下点无法通过 PCC 求解：")
        for traj_id, idx, err in failures:
            print(f"  轨迹 {traj_id}，第 {idx} 行 → {err}")
    else:
        print("所有轨迹的所有点均可由 PCC 模型成功反算。")

    # 可选：将结果保存到 CSV
    res_df = pd.DataFrame(results, columns=['traj','point','x','y','z'])
    res_df.to_csv('pcc_results.csv', index=False)
    print("反算成功的位姿已保存至 pcc_results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", "-c", type=str, default="command_all.csv",
        help="输入的编码器数据 CSV 路径（无 header）"
    )
    args = parser.parse_args()
    validate_pcc(args.csv)
