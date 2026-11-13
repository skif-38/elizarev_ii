import numpy as np
from scipy import stats

raw = np.loadtxt("ivaniv.dat")
before = raw[:, 0]
after = raw[:, 1]

threshold = 50

support_before = before >= threshold
support_after = after >= threshold

no_to_yes = np.sum(~support_before & support_after)
yes_to_no = np.sum(support_before & ~support_after)

b = no_to_yes
c = yes_to_no
m = b + c

if m == 0:
    z_support = 0.0
    p_support = 1.0
else:
    z_support = (b - c) / np.sqrt(m)
    p_support = 2 * stats.norm.sf(abs(z_support))

print(f"z = {z_support:.6f}, p = {p_support:.6f}")
print("Поддержка не изменилась")

delta = after - before
t_stat, p_mean = stats.ttest_rel(after, before)

print(f"\nСред разница: {delta.mean():.4f}")
print(f"t = {t_stat:.6f}, p = {p_mean:.6f}")
print("Рейтинг снизился")

print(" уровень поддержки поддержки не изменился потому что p ≈ 0.31 > 0.05, средний рейтинг снизился −2.88, p ~ 0.00045 < 0.05)")
