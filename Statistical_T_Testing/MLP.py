import numpy as np
from scipy import stats

def paired_test(mae_with_exog, mae_without_exog):
    t_stat, p_value = stats.ttest_rel(mae_with_exog, mae_without_exog)
    
    diff = mae_with_exog - mae_without_exog
    cohen_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return t_stat, p_value, cohen_d

mae_with_outliers = np.array([1590.65, 9831.43, 1557.82, 1666.54, 1578.36])
mae_without_outliers = np.array([1717.49, 1737.77, 1702.57, 10625.21, 1661.67])

mae_with_stable = np.array([1590.65, 1557.82, 1666.54, 1578.36])
mae_without_stable = np.array([1717.49, 1737.77, 1702.57, 1661.67])

t_stat, p_value, cohen_d = paired_test(mae_with_outliers, mae_without_outliers)
print("Including outliers:")
print(f"Two-sided paired t-test: t = {t_stat}, p = {p_value}")
print(f"Cohen's d = {cohen_d}\n")

t_stat, p_value, cohen_d = paired_test(mae_with_stable, mae_without_stable)
print("Excluding outliers:")
print(f"Two-sided paired t-test: t = {t_stat}, p = {p_value}")
print(f"Cohen's d = {cohen_d}")
