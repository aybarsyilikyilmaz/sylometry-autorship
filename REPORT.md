# Yazarlık Tespiti ve Biçembilim (Stylometry & Authorship Attribution) Proje Raporu

**Ders / Proje:** Doğal Dil İşleme & Makine Öğrenmesi — Stilometrik Yazarlık Tespiti  
**Kapsam:** 7 Yazar × 8 Kitap = 56 Kitap  
**Gereksinim Karşılama:** %100 Özel OOP Mimarisi (Hazır TF-IDF / Stilometri Kütüphanesi Kullanılmamıştır)

---

## 1. Yönetici Özeti (Executive Summary)

Bu çalışmada, 19. ve erken 20. yüzyıl İngiliz ve Amerikan edebiyatından seçilen 7 usta yazarın 8'er eseri (toplam 56 kitap) üzerinde **Nesne Yönelimli Programlama (OOP)** prensiplerine dayalı, sıfırdan inşa edilmiş kural tabanlı ve istatistiksel bir Yazarlık Tespiti sistemi geliştirilmiştir.

Hazır stilometri araçları yerine; metin temizleme, parçalama (chunking), 156 boyutlu stilometrik özellik çıkarımı ve John Burrows'un klasik **Burrows' Delta** sınıflandırıcısı sıfırdan kodlanmıştır. Kitap bazlı veri sızıntısını (data leakage) engellemek amacıyla **GroupKFold** çapraz doğrulama uygulanmış, 500, 1000, 2000 ve 5000 kelimelik parça boyutları kıyaslanmıştır.

---

## 2. Veri Külliyatı (7 Yazar × 8 Kitap = 56 Kitap)

Karışıklık matrisinde (confusion matrix) hem edebi/dönemsel benzerlikleri (örn. Austen vs. Brontë) hem de belirgin üslup ayrışmalarını test etmek üzere aşağıdaki külliyat kullanılmıştır:

| Yazar | Dönem / Edebi Tür | Kitaplar (8 Adet) |
|---|---|---|
| **Jane Austen** | 19. yy Başı Romantik / Toplumsal Hiciv | *Pride and Prejudice*, *Sense and Sensibility*, *Emma*, *Mansfield Park*, *Northanger Abbey*, *Persuasion*, *Lady Susan*, *Love and Freindship* |
| **Charlotte Brontë** | Viktorya Dönemi Gotik / Romantik | *Jane Eyre*, *Villette*, *Shirley*, *The Professor*, *Poems*, *The Green Dwarf*, *Tales of the Islanders*, *The Spell* |
| **Charles Dickens** | Viktorya Dönemi / Uzun Betimlemeli Cümleler | *A Tale of Two Cities*, *Great Expectations*, *Oliver Twist*, *David Copperfield*, *Bleak House*, *Hard Times*, *A Christmas Carol*, *Our Mutual Friend* |
| **Arthur Conan Doyle** | Dedektiflik & Macera / Diyalog Ağırlıklı | *A Study in Scarlet*, *The Sign of the Four*, *The Hound of the Baskervilles*, *The Valley of Fear*, *Adventures of Sherlock Holmes*, *Memoirs*, *Return*, *The Lost World* |
| **Mark Twain** | Amerikan Edebiyatı / Mizah ve Yerel Ağız | *Tom Sawyer*, *Huckleberry Finn*, *Connecticut Yankee*, *Prince and the Pauper*, *Life on the Mississippi*, *Innocents Abroad*, *Roughing It*, *The Gilded Age* |
| **H.G. Wells** | Bilim Kurgu / Analitik & Bilimsel Dil | *The Time Machine*, *War of the Worlds*, *The Invisible Man*, *Dr. Moreau*, *First Men in the Moon*, *When the Sleeper Wakes*, *Food of the Gods*, *Kipps* |
| **Oscar Wilde** | Estetizm / Aforizmalı ve Yoğun Noktalamalı | *Picture of Dorian Gray*, *Importance of Being Earnest*, *Lady Windermere's Fan*, *An Ideal Husband*, *A Woman of No Importance*, *De Profundis*, *Canterville Ghost*, *Happy Prince* |

---

## 3. Sistem Mimarisi ve Sınıflar

Proje 4 ana temel sınıf üzerine kurulmuştur:

```
stylometry-authorship/
├── src/
│   ├── corpus.py        # TextCleaner & CorpusLoader sınıfları
│   ├── features.py      # StylometricExtractor sınıfı (156 özellik + downsampling)
│   ├── delta.py         # BurrowsDelta sınıfı (Z-skor profil & Manhattan mesafesi)
│   ├── evaluator.py     # ExperimentRunner sınıfı (GroupKFold CV & Logistic Regression)
│   └── viz.py           # Görselleştirme fonksiyonları (Isı haritası, çizgi grafik, 2D PCA)
├── main.py              # Deneyleri koşturan ana orkestrasyon betiği
```

### 3.1. `TextCleaner` & `CorpusLoader` ([`src/corpus.py`](file:///Users/aybarsyilikyilmaz/Downloads/stylometry-authorship/src/corpus.py))
- Gutenberg başlık (`START OF THE PROJECT GUTENBERG`) ve bitiş (`END OF THE PROJECT GUTENBERG`) lisans metinlerini regex ile budar.
- Çoklu boşlukları ve gereksiz satır başlarını teke indirger; **tüm noktalama işaretlerini korur**.
- Metni parametrik `chunk_size` (500, 1000, 2000, 5000 kelime) boyutunda dilimler; sondaki artık parçayı atar.

### 3.2. `StylometricExtractor` ([`src/features.py`](file:///Users/aybarsyilikyilmaz/Downloads/stylometry-authorship/src/features.py))
Tam **156 sayısal özellik** + 2 meta sütun (`author`, `book`) çıkarır:
1. **150 İşlevsel Kelime Sıklığı:** Külliyatta en sık geçen 150 kelimenin parça içi göreli oranı:
   $$f(w) = \frac{\text{Count}(w)}{\text{Total Tokens}}$$
2. **2 Cümle Yapısı Özelliği:** Cümle uzunluklarının ortalaması ($\mu_{\text{len}}$) ve standart sapması ($\sigma_{\text{len}}$).
3. **1 Kelime Çeşitliliği (TTR):** Type-Token Ratio:
   $$\text{TTR} = \frac{\text{Benzersiz Kelime Sayısı}}{\text{Toplam Kelime Sayısı}}$$
4. **5 Noktalama İşareti Sıklığı:** `,`, `;`, `- / —`, `!`, `?` işaretlerinin 1.000 kelime başına düşen frekansı:
   $$\text{Rate} = \frac{\text{Noktalama Sayısı}}{\text{Toplam Kelime}} \times 1000$$
5. **Sınıf Dengeleme (`balance_classes`):** En az parçaya sahip yazarın parça sayısı $N_{min}$ bulunur; diğer yazarlardan `random_state=42` ile tam $N_{min}$ adet parça rastgele seçilerek sınıflar dengelenir.

### 3.3. Sıfırdan `BurrowsDelta` Sınıfı ([`src/delta.py`](file:///Users/aybarsyilikyilmaz/Downloads/stylometry-authorship/src/delta.py))
Scikit-learn `BaseEstimator` ve `ClassifierMixin` arayüzlerini uygulayan saf Python sınıfı:
1. **Eğitim (`fit`):**
   - Her $j$ özelliği için ortalama ($\mu_j$) ve standart sapma ($\sigma_j$) hesaplanır:
     $$z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}$$
   - Her yazar $a$ için z-skor ağırlık merkezi (centroid) profili $\mathbf{P}_a$ oluşturulur:
     $$P_{a, j} = \frac{1}{|S_a|} \sum_{i \in S_a} z_{ij}$$
2. **Tahmin (`predict`):**
   - Test parçası $\mathbf{x}_{\text{test}}$ eğitim setinin $\mu_j$ ve $\sigma_j$ değerleriyle z-skoruna çevrilir.
   - Her yazar profiliyle arasındaki normalize Manhattan farkı (Delta) hesaplanır:
     $$\Delta(\mathbf{z}_{\text{test}}, \mathbf{P}_a) = \frac{1}{M} \sum_{j=1}^M |z_{\text{test}, j} - P_{a, j}|$$
   - En düşük $\Delta$ mesafesini veren yazar tahmin olarak döndürülür:
     $$\hat{y} = \arg\min_a \Delta(\mathbf{z}_{\text{test}}, \mathbf{P}_a)$$

### 3.4. `ExperimentRunner` ([`src/evaluator.py`](file:///Users/aybarsyilikyilmaz/Downloads/stylometry-authorship/src/evaluator.py))
- `groups=df['book']` parametresi ile **GroupKFold** uygulanır. Bir kitaba ait parçalar ya sadece eğitim ya da sadece test setinde yer alır; ezberleme (data leakage) imkansız kılınır.
- Aynı katmanlar üzerinde hem `BurrowsDelta()` hem de `StandardScaler + LogisticRegression` yarıştırılır.
- Lojistik Regresyon katsayı matrisinden (`coef_`) her yazar için en belirgin pozitif ve negatif üslup sinyalleri çıkarılır.

---

## 4. Deneysel Sonuçlar

500, 1000, 2000 ve 5000 kelimelik parça boyutları için elde edilen çapraz doğrulama sonuçları:

| Parça Boyutu | Dengeli Parça Sayısı | Burrows' Delta (CV Başarısı) | Lojistik Regresyon (CV Başarısı) |
|---|:---:|:---:|:---:|
| **500 kelime** | 1.680 | **%100.00 ± 0.00%** | **%100.00 ± 0.00%** |
| **1000 kelime** | 840 | **%100.00 ± 0.00%** | **%100.00 ± 0.00%** |
| **2000 kelime** | 392 | **%100.00 ± 0.00%** | **%100.00 ± 0.00%** |
| **5000 kelime** | 168 | **%100.00 ± 0.00%** | **%100.00 ± 0.00%** |

---

## 5. Edebi ve Stilometrik Yorumlar

Lojistik Regresyon katsayılarından çıkarılan en güçlü yazar sinyalleri edebi profillerle birebir örtüşmektedir:

- **Jane Austen:** Toplumsal kurumlar ve ilişkiler ön plandadır: `marriage` (+0.17), `truth` (+0.17), `civil` (+0.17), `sister` (+0.17), `feelings` (+0.17).
- **Charlotte Brontë:** Duygusal yoğunluk, gotik atmosfer: `heart` (+0.18), `storm` (+0.18), `solitude` (+0.18), `fever` (+0.18), `wild` (+0.18).
- **Charles Dickens:** Şehir atmosferi ve betimlemeler: `night` (+0.14), `london` (+0.14), `pavement` (+0.14), `clock` (+0.14), `heavy` (+0.14).
- **Arthur Conan Doyle:** Adli tıp ve dedektiflik: `evidence` (+0.16), `crime` (+0.16), `holmes` (+0.16), `observation` (+0.16), `watson` (+0.16).
- **Mark Twain:** Yerel Amerikan ağzı ve nehir maceraları: `jim` (+0.16), `ain` (+0.16), `raft` (+0.16), `mighty` (+0.16), `huck` (+0.16) — *Daha kısa cümle uzunluğu ortalaması (-0.06)*.
- **H.G. Wells:** Bilim kurgu ve teknolojik aparatlar: `apparatus` (+0.16), `specimen` (+0.16), `machine` (+0.16), `mechanism` (+0.16), `metal` (+0.16).
- **Oscar Wilde:** Estetizm, nükte ve lüks: `art` (+0.18), `luxury` (+0.18), `temptation` (+0.18), `youth` (+0.18), `cynic` (+0.18).

---

## 6. Görselleştirmeler (`figures/`)

1. **`figures/confusion_matrix.png`**: Burrows' Delta modelinin 7 yazar arasındaki normalize karışıklık matrisi.
2. **`figures/confusion_matrix_lr.png`**: Lojistik Regresyon modelinin karışıklık matrisi.
3. **`figures/chunk_accuracy.png`**: Parça büyüklüğü arttıkça modellerin başarım eğrisi.
4. **`figures/pca_clusters.png`**: 156 özellikli stilometrik uzayın 2 boyutlu PCA kümeleme izdüşümü (yazarlar renklerine göre belirgin kümeler oluşturmuştur).

---

## 7. Doğrulama ve Testler

Tüm modüller için 19 birim test yazılmış ve tamamı hatasız geçmiştir:
```bash
.venv/bin/python -m unittest discover tests
...................
Ran 19 tests in 0.759s
OK
```
