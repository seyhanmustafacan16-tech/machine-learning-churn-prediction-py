"""
Proje Amacı: Müşteri ayrılma (churn) tahmini için veri üretiminden model testine kadar uzanan temel makine öğrenmesi akışının (pipeline) kurulması.
Kullanılan Kütüphaneler: pandas, numpy, scikit-learn
Çalıştırma Adımları:
    1. Gerekli kütüphanelerin bilgisayarda yüklü olduğundan emin olun. (pip install pandas numpy scikit-learn).
    2. Bu Python dosyasını terminalden veya kullandığınız IDE (VS Code vb.) üzerinden çalıştırın.
    3. Kod sırasıyla sentetik veri üretecek, ön işleme yapacak, modelleri eğitip sonuçları terminale yazdıracaktır.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

np.random.seed(42) # seed: Her çalıştırmada aynı veriyi almak için kullanılır.

veri_sayisi = 150 # 150 Adet sentetik (yapay) veri üretilir.

data = {
    'yas': np.random.randint(18, 70, veri_sayisi),
    'gelir': np.random.randint(30000, 150000, veri_sayisi),
    'abonelik_suresi': np.random.randint(1, 60, veri_sayisi),
    'destek_talebi_sayisi': np.random.randint(0, 10, veri_sayisi),
    'sehir': np.random.choice(['Istanbul', 'Ankara', 'Izmir', 'Bursa'], veri_sayisi),
    'uyelik_tipi': np.random.choice(['Standart', 'Premium'], veri_sayisi),
    'churn': np.random.choice([0, 1], veri_sayisi, p=[0.7, 0.3]) # Churn: 0 = Müşteri kalır (%70), 1 = Müşteri Ayrılır (%30)
}

df = pd.DataFrame(data)

print("=== VERI SETININ ILK 5 SATIRI ===")
print(df.head())

print("\n=== SATIR VE SUTUN SAYISI ===")
print(f"Toplam Satir: {df.shape[0]}, Toplam Sutun: {df.shape[1]}")

print("\n=== HEDEF DEGISKEN (CHURN) DAGILIMI ===")
print(df['churn'].value_counts())
print("-" * 50)



# --- VERİ ÖN İŞLEME AŞAMASI ---

# Öznitelik Üretimi (Feature Engineering)
df['destek_talebi_var_mi'] = df['destek_talebi_sayisi'].apply(lambda x: 1 if x > 0 else 0) # Müşterinin hiç destek talebi olmuş mu? (0: Hayır, 1: Evet)
print("\n'destek_talebi_var_mi' ozniteligi uretildi.")

print("\n=== EKSIK DEGER KONTROLU ===")
print(df.isnull().sum())

# Kategorik Değişkenleri Dönüştürme (One-Hot Encoding)
df = pd.get_dummies(df, columns=['sehir', 'uyelik_tipi'], drop_first=True, dtype=int) # drop_first=True -> gereksiz sütun tekrarını (dummy variable trap) önler.

# Sayısal Değişkenlerde Ölçekleme
sayisal_sutunlar = ['yas', 'gelir', 'abonelik_suresi', 'destek_talebi_sayisi']
scaler = StandardScaler()
df[sayisal_sutunlar] = scaler.fit_transform(df[sayisal_sutunlar])

print("\n=== ON ISLEME SONRASI ILK 5 SATIR ===")
print(df.head())
print("-" * 40)



# --- TRAIN, VALIDATION VE TEST BÖLME ---

X = df.drop('churn', axis=1)
y = df['churn']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"\n=== VERI BOLME ISLEMI ===")
print(f"Egitim (Train) seti boyutu: {X_train.shape[0]}")
print(f"Dogrulama (Validation) seti boyutu: {X_val.shape[0]}")
print(f"Test seti boyutu: {X_test.shape[0]}")



# --- MODEL EĞİTİMİ ---

print("\n=== MODELLER EGITILIYOR ===")

# 1. Model: Lojistik Regresyon
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)

# 2. Model: KNN (K-Nearest Neighbors)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

print("Logistic Regression ve KNN modelleri basariyla egitildi.")
print("-" * 40)



# --- VALIDASYON VE MODEL SEÇİMİ ---

print("\n=== VALIDATION (DOGRULAMA) KARSILASTIRMASI ===")

# Lojistik Regresyon Validation Performansı
y_val_pred_log = log_model.predict(X_val)
val_acc_log = accuracy_score(y_val, y_val_pred_log)
print(f"Logistic Regression Validation Dogrulugu (Accuracy): {val_acc_log:.2f}")

# KNN Validation Performansı
y_val_pred_knn = knn_model.predict(X_val)
val_acc_knn = accuracy_score(y_val, y_val_pred_knn)
print(f"KNN Validation Dogrulugu (Accuracy): {val_acc_knn:.2f}")

# Performansa göre en iyi modeli otomatik olarak seçiyoruz.
if val_acc_log >= val_acc_knn:
    secilen_model = log_model
    secilen_model_adi = "Logistic Regression"
else:
    secilen_model = knn_model
    secilen_model_adi = "KNN"

print(f"\nSecilen Model: {secilen_model_adi} (Validation performansi daha iyidir.)")



# --- TEST SETİ DEĞERLENDİRMESİ ---

print(f"\n=== {secilen_model_adi} TEST SETI METRIKLERI ===")
y_test_pred = secilen_model.predict(X_test)

print("Confusion Matrix (Karmasiklik Matrisi):")
print(confusion_matrix(y_test, y_test_pred))

print(f"Accuracy (Dogruluk):  {accuracy_score(y_test, y_test_pred):.2f}")
print(f"Precision (Kesinlik): {precision_score(y_test, y_test_pred, zero_division=0):.2f}")
print(f"Recall (Duyarlilik):  {recall_score(y_test, y_test_pred, zero_division=0):.2f}")
print(f"F1-Score:             {f1_score(y_test, y_test_pred, zero_division=0):.2f}")

print("\n=== SONUC YORUMU ===")
print(f">>> Model Secimi: Validation metriklerine bakildiginda {secilen_model_adi} modeli daha basarili olmus ve test setine alinmistir.")
print(f">>> Neden?: {secilen_model_adi} algoritmasi, urettigimiz 150 satirlik sentetik verideki matematiksel oruntuleri daha iyi kavramistir.")
print(">>> Genel Degerlendirme: Veri seti kucuk ve rastgele uretilmis oldugu icin test metriklerinde dalgalanmalar olabilir.")
print("    Ancak veri okuma, on isleme, model egitimi ve test adimlarindan olusan pipeline uctan uca hatasiz calismaktadir.")

