import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score)


def load(filepath):
    # чтение тхт файла
    with open(filepath) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if 'Freq' in l and 'Range' in l)
    rows = []
    for l in lines[start + 1:]:
        p = l.split()
        if len(p) >= 3:
            try:
                rows.append([float(x) for x in p[:9]])
            except ValueError:
                pass
    cols = ['Freq', 'Range', 'Pol', 'MPA', 'Amp',
            'Doppler', 'Az', 'Zn', 'PGH']
    return pd.DataFrame(rows, columns=cols)


def cluster(filepath):
    # DBSCAN
    df = load(filepath)
    X = df[['Freq', 'Range']].values
    X = X[~np.isnan(X).any(axis=1)]
    X = StandardScaler().fit_transform(X)
    n = len(X)
    ms_list = [2, int(np.log(n)) + 1, int(np.sqrt(n)), 4]
    best_score, best_labels, best_params = -1, None, None
    for ms in ms_list:
        nn = NearestNeighbors(n_neighbors=min(ms, n - 1)).fit(X)
        kdist = np.sort(nn.kneighbors(X)[0][:, -1])
        eps_list = [np.quantile(kdist, q) for q in (0.85, 0.90, 0.95)]
        for eps in eps_list:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            k = len(set(labels) - {-1})
            if k < 2:
                continue
            m = labels != -1
            Xv, lv = X[m], labels[m]
            sil = silhouette_score(Xv, lv)
            dbi = davies_bouldin_score(Xv, lv)
            chi = calinski_harabasz_score(Xv, lv)
            noise = np.mean(labels == -1)
            ari = adjusted_rand_score(
                lv, KMeans(k, n_init=10, random_state=0).fit_predict(Xv))
            score = (sil + 1 / (1 + dbi) + chi / (chi + 1)
                     + (1 - noise) + ari) / 5
            if score > best_score:
                best_score, best_labels = score, labels
                best_params = (eps, ms, k, sil, dbi, chi, noise, ari)
    eps, ms, k, sil, dbi, chi, noise, ari = best_params
    print(f'{filepath}')
    print(f'  eps={eps:.3f}, min_samples={ms}')
    print(f'  кластеров={k}, шум={100*noise:.1f}%')
    print(f'  silhouette={sil:.3f}, davies_bouldin={dbi:.3f}, '
          f'calinski_harabasz={chi:.1f}, ari={ari:.3f}')
    print(f'  score={best_score:.3f}')
    df['cluster'] = np.where(np.isnan(df[['Freq', 'Range']]).any(axis=1),
                             -1, best_labels)
    return df