import pandas as pd
import io

o3_data = """scene,valid_disparity_pixels,valid_ground_truth_pixels,mae,rmse,bad_1px
artroom1,892100,1922559,2.366439,5.998957,0.456409
artroom2,1134861,2008475,2.079891,5.795401,0.426600
bandsaw1,1132724,2002219,1.231792,2.425939,0.354005
bandsaw2,549148,2004851,2.948895,7.337898,0.481833
chess1,694127,1767536,2.791344,8.611643,0.518237
chess2,820411,1891420,2.462869,8.233381,0.413022
chess3,1099814,1835761,1.796834,6.495133,0.292957
curule1,1204039,1888228,1.524969,4.898473,0.274517
curule2,1171613,1882860,1.767609,6.284306,0.203161
curule3,1367461,1936434,1.355209,4.020841,0.206492
ladder1,1217694,1964242,2.110065,3.836563,0.480199
ladder2,1268832,1945108,2.034110,3.910171,0.485464
octogons1,513362,1908316,2.584306,6.489880,0.460119
octogons2,693476,1962501,2.158193,4.684907,0.389472
pendulum1,863769,1937363,1.714076,5.028061,0.284699
pendulum2,646959,1993429,3.660179,9.368093,0.514296
podium1,126902,1832241,7.304799,19.756601,0.524466
skates1,760770,1544848,1.592723,8.455462,0.306613
skates2,1574268,1651689,1.216418,4.578375,0.256082
skiboots1,1375794,1800669,1.996400,7.607337,0.280656
skiboots2,954302,1790824,2.770043,9.041889,0.522916
skiboots3,1453773,1901789,2.069880,7.657701,0.405431
traproom1,988947,1700060,2.669948,7.695451,0.431218
traproom2,1268249,1880921,1.848709,5.497290,0.344742"""
o3_df = pd.read_csv(io.StringIO(o3_data))

o4_data = """scene,fold,token_grid,valid_disparity_pixels,valid_ground_truth_pixels,mae,rmse,bad_1px,mean_confidence,estimated_working_set_mb
artroom1,0,540x960,1728276,1922559,1.990668,6.465405,0.324700,0.025512,387.597656
artroom2,1,540x960,1809379,2008475,2.203735,6.517442,0.379445,0.020012,387.597656
bandsaw1,2,540x960,1801722,2002219,1.838542,4.179800,0.311525,0.018766,387.597656
bandsaw2,3,540x960,1459880,2004851,3.822492,6.713164,0.606072,0.011676,387.597656
chess1,4,540x960,1430109,1767536,6.980379,15.953807,0.501521,0.016367,387.597656
chess2,0,540x960,1388894,1891420,7.570698,16.668667,0.417943,0.019887,387.597656
chess3,1,540x960,1607890,1835761,5.262598,13.102783,0.314536,0.019574,387.597656
curule1,2,540x960,1657369,1888228,3.233602,7.752802,0.300748,0.022735,387.597656
curule2,3,540x960,1646479,1882860,2.912904,8.336313,0.278679,0.020704,387.597656
curule3,4,540x960,1789882,1936434,1.789131,4.801649,0.255860,0.023260,387.597656
ladder1,0,960x540,1831725,1964242,1.979687,3.739090,0.433736,0.027462,387.597656
ladder2,1,960x540,1820684,1945108,1.944396,3.875507,0.403665,0.024095,387.597656
octogons1,2,540x960,1710410,1908316,1.893183,5.518516,0.292879,0.017197,387.597656
octogons2,3,540x960,1868871,1962501,1.933283,4.427618,0.306344,0.014136,387.597656
pendulum1,4,540x960,1624088,1937363,1.672324,4.465377,0.268553,0.018711,387.597656
pendulum2,0,540x960,1611742,1993429,2.770417,6.086673,0.422494,0.021788,387.597656
podium1,1,540x960,1064641,1832241,12.776092,21.652277,0.723903,0.010684,387.597656
skates1,2,540x960,1272877,1544848,1.928083,5.934153,0.361377,0.016127,387.597656
skates2,3,540x960,1646503,1651689,1.048141,2.841086,0.273711,0.020369,387.597656
skiboots1,4,540x960,1782507,1800669,1.477846,5.069563,0.215019,0.019487,387.597656
skiboots2,0,540x960,1410655,1790824,3.428607,11.708120,0.462070,0.023610,387.597656
skiboots3,1,540x960,1667516,1901789,3.076530,10.017492,0.455651,0.021737,387.597656
traproom1,2,540x960,1429786,1700060,3.483291,10.288795,0.324611,0.019615,387.597656
traproom2,3,540x960,1660768,1880921,2.606062,8.483561,0.296306,0.017545,387.597656"""
o4_df = pd.read_csv(io.StringIO(o4_data))

fold_data = """fold,scene_count,scenes_with_ground_truth,mean_mae,mean_rmse,mean_bad_1px
0,5,5,3.548015,8.933591,0.412189
1,5,5,5.052670,11.033100,0.455440
2,5,5,2.475340,6.734813,0.318228
3,5,5,2.464576,6.160348,0.352222
4,4,4,2.979920,7.572599,0.310238"""
fold_df = pd.read_csv(io.StringIO(fold_data))

def print_df_metrics(df, label):
    best_scene_mae = df.loc[df['mae'].idxmin(), 'scene']
    worst_scene_mae = df.loc[df['mae'].idxmax(), 'scene']
    print(f"1. {label} Summary:")
    print(f"  Scene count: {len(df)}")
    print(f"  Mean MAE: {df['mae'].mean():.4f}, Median MAE: {df['mae'].median():.4f}")
    print(f"  Mean RMSE: {df['rmse'].mean():.4f}")
    print(f"  Mean bad_1px: {df['bad_1px'].mean():.4f}")
    print(f"  Best scene (MAE): {best_scene_mae} ({df['mae'].min():.4f})")
    print(f"  Worst scene (MAE): {worst_scene_mae} ({df['mae'].max():.4f})")

print_df_metrics(o3_df, "O3")

print("\n2. O3 Specific Scenes:")
for s in ['artroom2', 'chess3', 'ladder1']:
    row = o3_df[o3_df['scene'] == s].iloc[0]
    print(f"  {s}: MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, bad_1px={row['bad_1px']:.4f}")

def print_df_metrics_o4(df, label):
    best_scene_mae = df.loc[df['mae'].idxmin(), 'scene']
    worst_scene_mae = df.loc[df['mae'].idxmax(), 'scene']
    print(f"\n3. {label} Summary:")
    print(f"  Scene count: {len(df)}")
    print(f"  Mean MAE: {df['mae'].mean():.4f}, Median MAE: {df['mae'].median():.4f}")
    print(f"  Mean RMSE: {df['rmse'].mean():.4f}")
    print(f"  Mean bad_1px: {df['bad_1px'].mean():.4f}")
    print(f"  Best scene (MAE): {best_scene_mae} ({df['mae'].min():.4f})")
    print(f"  Worst scene (MAE): {worst_scene_mae} ({df['mae'].max():.4f})")

print_df_metrics_o4(o4_df, "O4")

print("\n4. O4 Fold Summary:")
for _, row in fold_df.iterrows():
    print(f"  Fold {int(row['fold'])}: count={int(row['scene_count'])}, mean MAE={row['mean_mae']:.4f}, mean RMSE={row['mean_rmse']:.4f}, mean bad_1px={row['mean_bad_1px']:.4f}")

print("\n5. O4 Specific Scenes:")
for s in ['artroom2', 'chess3', 'ladder1']:
    row = o4_df[o4_df['scene'] == s].iloc[0]
    print(f"  {s}: Fold {int(row['fold'])}, MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, bad_1px={row['bad_1px']:.4f}")

common_scenes = list(set(o3_df['scene']) & set(o4_df['scene']))
common_o3 = o3_df[o3_df['scene'].isin(common_scenes)].sort_values('scene').reset_index(drop=True)
common_o4 = o4_df[o4_df['scene'].isin(common_scenes)].sort_values('scene').reset_index(drop=True)

print(f"\n6. O3 vs O4 Comparison (Common Scenes: {len(common_scenes)}):")
print(f"  Mean MAE: O3={common_o3['mae'].mean():.4f}, O4={common_o4['mae'].mean():.4f}")
print(f"  Mean RMSE: O3={common_o3['rmse'].mean():.4f}, O4={common_o4['rmse'].mean():.4f}")
print(f"  Mean bad_1px: O3={common_o3['bad_1px'].mean():.4f}, O4={common_o4['bad_1px'].mean():.4f}")

mae_win = (common_o4['mae'] < common_o3['mae']).sum()
rmse_win = (common_o4['rmse'] < common_o3['rmse']).sum()
bad_win = (common_o4['bad_1px'] < common_o3['bad_1px']).sum()

print(f"  O4 Wins count: MAE={mae_win}, RMSE={rmse_win}, bad_1px={bad_win}")
