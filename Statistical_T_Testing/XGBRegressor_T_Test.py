import numpy as np
from scipy import stats

mae_without_exog = np.array([1601.45, 1597.95, 1607.47, 1580.54, 1596.56])
mae_with_exog = np.array([1652.51, 1644.11, 1635.32, 1647.16, 1654.32])

t_stat, p_value = stats.ttest_rel(mae_with_exog, mae_without_exog)

diff = mae_with_exog - mae_without_exog
cohen_d = np.mean(diff) / np.std(diff, ddof=1)

print(f"Two-sided paired t-test: t = {t_stat}, p = {p_value}")
print(f"Cohen's d = {cohen_d}")
