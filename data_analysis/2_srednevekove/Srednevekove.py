import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('ivaniv.dat.txt', sep=' ', header=None, names=['before','after'])
df['diff'] = df.after - df.before

# кластеризация по изменению рейтинга
df['cluster'] = KMeans(n_clusters=3, random_state=0, n_init=10).fit_predict(df[['diff']])

support_before = (df.before >= 50).mean() * 100
support_after = (df.after >= 50).mean() * 100
mean_before = df.before.mean()
mean_after = df.after.mean()

print(f"Поддержка: было {support_before:.0f}%  стало {support_after:.0f}%")
print(f"Средний рейтинг: было {mean_before:.2f} стало {mean_after:.2f}")
print(f"Изменение среднего: {mean_after - mean_before:+.2f}")