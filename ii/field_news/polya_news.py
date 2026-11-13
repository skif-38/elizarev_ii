# Загрузка данных
from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups(subset='all')
y=data.target

# TFIDF эмбеддинг слов с использованнием TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_df=0.8, min_df=3, stop_words='english')
X_tfidf=vectorizer.fit_transform(data.data)
print(f"Размерность TF-IDF: {X_tfidf.shape}")

# Уменьшение размерности и визуализация
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


svd=TruncatedSVD(n_components=100)
x_svd=svd.fit_transform(X_tfidf)
tsne=TSNE(n_components=2, perplexity=40)
x_tsne=tsne.fit_transform(x_svd)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap='tab20', s=5, alpha=0.7)
plt.title("t-SNE визуализация (цвет = исходная категория)")
plt.show()



# Кластеризация KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sopt=-1
kopt=-1
copt=None

for k in range(12, 28):
    clf=KMeans(n_clusters=k, n_init=10, random_state=42)
    c=clf.fit_predict(x_svd)

    s=silhouette_score(x_svd, c)
    print(f"k={k}, Силуэт={s:.4f}")

    if s>sopt:
        sopt=s
        kopt=k
        copt=c

print(f"Оптимальное число кластеров: {kopt}")
print(f"Лучший силуэт: {sopt:.4f}")

# Сравнение с исходной разметкой датасета
from sklearn.metrics import adjusted_rand_score

ari=adjusted_rand_score(y, copt)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
# Визуализация результата
plt.figure(figsize=(16, 7))
plt.subplot(1, 2, 1)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap='tab20', s=5, alpha=0.7)
plt.title("Исходные категории (20 классов)")
plt.subplot(1, 2, 2)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=copt, cmap='tab20', s=5, alpha=0.7)
plt.title(f"Результат KMeans (k={kopt})")
plt.show()

# Низкий ARI (около 0.1): В похожих темах используются одни и те же слова,
#и метод считает только частоту слов, поэтому похожие темы слились в одну общую кучу,
#которую почти невозможно нормально разделить на 20 исходных категорий.

# Почему K=25 (или другое число), а не 20: Так как тексты перемешались в 1 большую кучу,
# алгоритму KMeans было проще нарезать её на случайные кусочки, чем угадать 20 реальных размытых тем.
# Силуэт тоже очень низкий (около 0.06), что подтверждает, что кластеры сильно перекрываются.