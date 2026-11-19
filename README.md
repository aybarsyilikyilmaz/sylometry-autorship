# Stylometry & Authorship Attribution (Yazarlık Tespiti ve Biçembilim)

Bu proje, 19. ve erken 20. yüzyıl İngiliz ve Amerikan edebiyatından **7 yazar** ve her yazardan **8 kitap** (toplam **56 kitap**) kullanılarak geliştirilmiş, nesne yönelimli (OOP) ve kural tabanlı bir **Yazarlık Tespiti (Authorship Attribution)** sistemidir.

Projede hazır kütüphanelerin (örn. `TfidfVectorizer` veya hazır stilometri paketleri) yerine; metin temizleme, parçalama motoru, 156 boyutlu stilometrik özellik çıkarıcı ve Burrows' Delta sınıflandırıcısı sıfırdan Python sınıfları olarak tasarlanmıştır.

---

## 📚 Veri Seti ve Yazarlar (7 Yazar × 8 Kitap)

Karışıklık matrisinde (confusion matrix) hem tür/dönem yakınlığını hem de üslup ayrışmalarını inceleyebilmek amacıyla seçilen yazarlar:

1. **Jane Austen**: Romantik / Erken 19. yy, toplumsal hiciv ve diyalog yapısı.
2. **Charlotte Brontë**: Viktorya dönemi, gotik / romantik üslup (Austen ile dönem ve üslup benzerliği test edilir).
3. **Charles Dickens**: Viktorya dönemi, uzun betimlemeli ve zengin sözcük dağarcıklı cümleler.
4. **Arthur Conan Doyle**: Dedektiflik ve macera, diyalog ağırlıklı ve aksiyon odaklı anlatım.
5. **Mark Twain**: Amerikan İngilizcesi, yerel ağız ve mizahi anlatım tarzı.
6. **H.G. Wells**: Bilim kurgu, analitik, nesnel ve görece kısa cümle yapıları.
7. **Oscar Wilde**: Aforizmalı, yoğun noktalama kullanımı ve nüktedan retorik.

---

## 🏗️ Proje Mimarisi

```
stylometry-authorship/
├── data/
│   ├── raw/                  # {author}/{book_title}.txt formatında 56 ham kitap
│   └── processed/            # features_500.csv, features_1000.csv, features_2000.csv, features_5000.csv
├── figures/                  # 300 DPI grafik çıktıları
│   ├── confusion_matrix.png
│   ├── chunk_accuracy.png
│   └── pca_clusters.png
├── src/
│   ├── __init__.py
│   ├── corpus.py             # TextCleaner & CorpusLoader sınıfları
│   ├── features.py           # StylometricExtractor sınıfı (156 özellik + sınıf dengeleme)
│   ├── delta.py              # BurrowsDelta sınıfı (Z-skor profil & Manhattan mesafesi)
│   ├── evaluator.py          # ExperimentRunner sınıfı (GroupKFold CV & Logistic Regression)
│   └── viz.py                # Görselleştirme fonksiyonları (Isı haritası, çizgi grafik, 2D PCA)
├── main.py                   # Uçtan uca deneyleri çalıştıran ana betik
├── requirements.txt
└── README.md
```

---

## 🧮 156 Özellikli Stilometri Vektörü

Her metin parçası (chunk) için tam 156 ayırt edici stilometrik özellik çıkarılır:

1. **İşlevsel Kelime Sıklıkları (150 Özellik)**:
   Korpus genelinde en sık geçen 150 kelimenin parça içi bağıl sıklığı:
   $$\text{Frekans} = \frac{\text{Kelime Sayısı}}{\text{Parçadaki Toplam Kelime}}$$

2. **Cümle Yapısı (2 Özellik)**:
   - Ortalama cümle uzunluğu (kelime sayısı cinsinden).
   - Cümle uzunluklarının standart sapması.

3. **Kelime Çeşitliliği / Zenginliği (1 Özellik)**:
   - Type-Token Ratio (TTR):
     $$\text{TTR} = \frac{\text{Benzersiz Kelime Sayısı (Type)}}{\text{Toplam Kelime Sayısı (Token)}}$$
     *(Sabit parça uzunlukları ile hesaplandığından metin boyutu yanlılığı ortadan kalkar).*

4. **Noktalama İşaretleri Sıklığı (5 Özellik)**:
   - Virgül (`,`), noktalı virgül (`;`), tire (`-` / `—`), ünlem (`!`), soru işareti (`?`) karakterlerinin bin kelime başına sıklığı:
     $$\text{Punctuation Rate} = \frac{\text{İşaret Sayısı}}{\text{Toplam Kelime}} \times 1000$$

5. **Meta Bilgi Sütunları (2 Sütun)**:
   - `author`: Yazar etiketi (hedef değişken).
   - `book`: Kitap adı (GroupKFold gruplama anahtarı).

---

## ⚖️ Sınıf Dengeleme (Downsampling)

Yazarların ürettikleri toplam parça sayısındaki dengesizlikleri ("emin değilsen Dickens de" yanlılığı) önlemek amacıyla:
- Tüm kitaplar parçalandıktan sonra en az parçaya sahip yazarın parça sayısı $N_{min}$ tespit edilir.
- Diğer yazarlardan `random_state=42` ile tam $N_{min}$ adet parça rastgele seçilerek dengeli bir veri kümesi elde edilir.

---

## 🔬 Değerlendirme Metodolojisi

1. **GroupKFold Çapraz Doğrulama**:
   Gruplar `book` sütununa göre ayrılır. Böylece aynı kitabın parçaları asla hem eğitim hem de test setinde yer almaz; modelin kitap ezberlemesi (data leakage) engellenir.
2. **Kıyaslanan Modeller**:
   - **Burrows' Delta**: Z-skor profili ve Manhattan mesafesi tabanlı sınıflandırıcı.
   - **Logistic Regression / LinearSVC**: `StandardScaler` ile ölçeklendirilmiş lineer model.
3. **Parça Boyutu Deneyleri**:
   500, 1000, 2000 ve 5000 kelimelik parça boyutlarının doğruluğa etkisi incelenir.

---

## 🚀 Kurulum ve Çalıştırma

```bash
# Sanal ortam oluşturma ve etkinleştirme
python3 -m venv .venv
source .venv/bin/activate

# Gereksinimlerin yüklenmesi
pip install -r requirements.txt

# Ana deneylerin yürütülmesi
python main.py
```
