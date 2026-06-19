# DRAFT PROPOSAL UTS: BAB 1 - 3
## ANALISIS KUANTITATIF DETEKSI POLA BREAKOUT SAHAM MENGGUNAKAN PENDEKATAN HYBRID RULE-BASED DAN MACHINE LEARNING

**Mata Kuliah:** Sistem Cerdas  
**Kelompok Peneliti (Breakout 1):**
* Edisyah Putra Waruwu
* Marviel David
* Andri Simbolon

---

## BAB I: PENDAHULUAN

### 1.1 Latar Belakang
Pasar modal merupakan salah satu instrumen investasi yang memiliki tingkat kompleksitas dan volatilitas tinggi. Pergerakan harga saham dipengaruhi oleh berbagai faktor fundamental, sentimen pasar, makroekonomi, hingga faktor psikologis pelaku pasar. Dalam analisis teknikal, para pelaku pasar sering kali mencari pola-pola pergerakan harga tertentu untuk mengidentifikasi arah pergerakan harga di masa depan. Salah satu pola yang paling populer dan memiliki potensi keuntungan besar adalah pola **Breakout**.

*Breakout* terjadi ketika harga saham menembus area resistensi (batas atas psikologis) yang biasanya disertai dengan peningkatan volume transaksi yang signifikan. Kejadian ini menandakan adanya pergeseran kekuatan antara penjual (*sellers*) dan pembeli (*buyers*), di mana pembeli mendominasi pasar dan mendorong harga ke tingkat yang lebih tinggi. Secara historis, momentum breakout ini sering kali memicu tren naik (*uptrend*) baru yang kuat.

Namun, kendala utama yang dihadapi oleh para pelaku pasar adalah fenomena **False Breakout** (atau *Bull Trap*). *False breakout* terjadi ketika harga saham menembus tingkat resistensi untuk waktu yang singkat, namun segera berbalik arah dan turun kembali di bawah resistensi tersebut. Kerugian finansial yang signifikan sering kali dialami oleh trader yang melakukan pembelian pada saat terjadi *false breakout*. Statistik menunjukkan bahwa sebagian besar kandidat breakout di pasar saham merupakan *false breakout*.

Secara tradisional, deteksi breakout dilakukan menggunakan pendekatan berbasis aturan (*Rule-Based*), seperti menetapkan harga tertinggi dalam $N$ hari terakhir sebagai resistensi. Meskipun pendekatan ini mudah diimplementasikan, ia tidak memiliki kemampuan untuk membedakan antara breakout yang valid dan yang palsu karena mengabaikan dinamika pasar yang lebih luas seperti momentum, tingkat kejenuhan pembelian (RSI), volatilitas volatilitas harga (ATR), dan kekuatan tren (MACD).

Untuk mengatasi kelemahan tersebut, penelitian ini mengusulkan **Pendekatan Hybrid**. Pendekatan ini menggabungkan keunggulan sistem berbasis aturan sebagai penyaring awal (*first-level filter*) untuk mendeteksi kandidat breakout, diikuti dengan penerapan algoritma Machine Learning (**Random Forest Classifier**) sebagai pengambil keputusan tingkat lanjut (*second-level classifier*). Dengan melatih model Machine Learning khusus pada data kandidat breakout, model diharapkan mampu mengenali pola tersembunyi dari indikator teknikal untuk memprediksi probabilitas keberhasilan breakout tersebut (*True vs False Breakout*).

### 1.2 Rumusan Masalah
Berdasarkan latar belakang di atas, rumusan masalah dalam penelitian ini adalah:
1. Bagaimana merancang aturan (*Rule-Based*) yang efektif untuk mendeteksi kandidat breakout awal pada data historis saham?
2. Bagaimana efektivitas algoritma Random Forest Classifier dalam mengklasifikasikan kandidat breakout menjadi *True Breakout* dan *False Breakout*?
3. Indikator teknikal apa yang memiliki pengaruh paling signifikan (*feature importance*) dalam menentukan keberhasilan suatu breakout saham?

### 1.3 Tujuan Penelitian
Tujuan dari penelitian ini adalah:
1. Membangun program Python hybrid yang mampu memuat data CSV saham dan mendeteksi kandidat breakout secara otomatis.
2. Melatih dan menguji model Random Forest Classifier untuk memprediksi keabsahan sinyal breakout secara kuantitatif.
3. Menganalisis metrik evaluasi model (Akurasi, Presisi, Recall, dan F1-Score) guna memastikan keandalan sinyal transaksi sebelum diimplementasikan pada perdagangan riil.

### 1.4 Manfaat Penelitian
Penelitian ini diharapkan memberikan manfaat sebagai berikut:
1. **Manfaat Akademis:** Memberikan kontribusi ilmiah dalam penerapan kecerdasan buatan (*Artificial Intelligence*) dan analisis kuantitatif pada bidang komputasi keuangan (*computational finance*).
2. **Manfaat Praktis:** Menjadi alat bantu bagi trader dan investor untuk menyaring sinyal trading breakout secara objektif, sistematis, dan meminimalkan risiko kerugian akibat *false breakout*.

---

## BAB II: TINJAUAN PUSTAKA

### 2.1 Analisis Teknikal dan Pola Breakout
Analisis teknikal adalah metode evaluasi instrumen keuangan dengan menganalisis statistik yang dihasilkan oleh aktivitas pasar, seperti harga historis dan volume. Pola breakout didasarkan pada konsep **Support dan Resistance**:
* **Support** adalah tingkat harga di mana minat beli cukup kuat untuk mengatasi tekanan jual, sehingga harga cenderung memantul ke atas.
* **Resistance** adalah tingkat harga di mana tekanan jual cukup kuat untuk mengatasi minat beli, sehingga mencegah harga naik lebih tinggi.

Ketika harga ditutup di atas tingkat resistensi, hal ini mengindikasikan bahwa batas penawaran telah ditembus. Konfirmasi dari sisi volume perdagangan (*volume confirmation*) sangat krusial; volume yang tinggi menunjukkan partisipasi institusional atau akumulasi besar-besaran, meningkatkan probabilitas kelanjutan tren naik.

### 2.2 Fenomena False Breakout (Bull Trap)
*False breakout* terjadi akibat manipulasi pasar jangka pendek atau kurangnya likuiditas lanjutan. Sering kali, harga didorong menembus resistensi untuk memicu perintah *buy stop* dari para trader ritel dan memicu *short covering*. Setelah likuiditas beli terkumpul, pelaku pasar besar (*market makers*) melakukan aksi ambil untung, menyebabkan harga jatuh kembali dengan cepat. Pendekatan berbasis statistik diperlukan untuk mendeteksi anomali ini.

### 2.3 Penerapan Machine Learning dalam Prediksi Pasar Keuangan
Penelitian terdahulu banyak berfokus pada prediksi harga saham harian menggunakan model regresi atau klasifikasi biner langsung pada seluruh data. Namun, model tersebut sering mengalami kegagalan karena fenomena *random walk* dari harga saham. 

Pendekatan hybrid yang membatasi pembelajaran mesin hanya pada kondisi-kondisi ekstrem (seperti saat harga mendekati resistensi) terbukti memberikan hasil yang lebih stabil. **Random Forest Classifier** merupakan algoritma *ensemble learning* berbasis pohon keputusan (*decision trees*) yang bekerja dengan cara membangun banyak pohon keputusan saat masa pelatihan dan mengeluarkan kelas rata-rata (klasifikasi). Algoritma ini sangat andal dalam menangani data keuangan yang *noisy* karena sifatnya yang kuat terhadap *outliers* dan kemampuan mendeteksi interaksi non-linear antar variabel input.

---

## BAB III: METODOLOGI PENELITIAN

### 3.1 Desain Sistem Hybrid
Penelitian ini menggunakan desain sistem dua tingkat (*two-stage system*):
1. **Stage 1 (Rule-Based Filter):** Menyeleksi data historis saham untuk menemukan hari-hari di mana harga memenuhi kriteria teknikal breakout awal.
2. **Stage 2 (Machine Learning Classifier):** Mengambil fitur-fitur teknikal pada hari kandidat tersebut dan memprediksi probabilitas keberhasilan transaksi menggunakan model Random Forest.

### 3.2 Data dan Variabel Penelitian
Data yang digunakan dalam penelitian ini adalah data sekunder berupa harga saham historis harian yang diperoleh dari Yahoo Finance. Variabel yang digunakan meliputi:
* **Variabel Input Dasar:** Date, Open, High, Low, Close, Volume.
* **Fitur Machine Learning (Feature Engineering):**
  1. *Relative Strength Index* (RSI-14): Mengukur kondisi jenuh beli atau jenuh jual.
  2. *MACD, MACD Signal, & MACD Histogram*: Mengukur momentum dan perubahan tren.
  3. *SMA Ratios* (Close/SMA20, Close/SMA50): Mengukur jarak harga terhadap rata-rata tren jangka pendek dan menengah.
  4. *Volume Ratio*: Mengukur kekuatan akumulasi dibanding hari biasa.
  5. *Normalized ATR* (ATR/Close): Mengukur tingkat volatilitas harga saat ini.
  6. *Price Volatility*: Standar deviasi pergerakan harga 20 hari terakhir.
  7. *Momentum (3d & 5d)*: Mengukur kecepatan perubahan harga jangka pendek.

### 3.3 Logika Deteksi dan Formulasi Matematika

#### 1. Aturan Deteksi Kandidat
Tingkat resistensi dinamis ditentukan menggunakan nilai tertinggi dari harga High selama $W$ hari perdagangan sebelumnya ($W=20$):
$$R_t = \max(High_{t-1}, High_{t-2}, \dots, High_{t-W})$$

Rata-rata volume perdagangan historis dihitung sebagai:
$$V\_Avg_t = \frac{1}{W} \sum_{k=1}^{W} Volume_{t-k}$$

Kondisi kandidat breakout didefinisikan sebagai:
$$Is\_Candidate_t = \begin{cases} 
1, & \text{jika } Close_t > R_t \quad \land \quad Volume_t > \alpha \times V\_Avg_t \\ 
0, & \text{lainnya} 
\end{cases}$$
Dengan $\alpha$ ditetapkan sebesar 1.5.

#### 2. Metode Pelabelan Sasaran (Target Labeling)
Menggunakan **Double-Barrier Method** dengan horizon waktu ke depan $H=5$ hari. Batas atas keuntungan ($UP$) ditetapkan sebesar 3% dan batas bawah risiko (*stop loss* / $DOWN$) sebesar 2%:
$$UP_t = Close_t \times (1 + 0.03)$$
$$DOWN_t = Close_t \times (1 - 0.02)$$

Label target untuk model pembelajaran mesin ditentukan oleh kondisi:
$$Y_t = \begin{cases} 
1, & \text{jika } \exists j \in [1, H] \text{ s.t. } High_{t+j} \ge UP_t \text{ sebelum } Low_{t+j} \le DOWN_t \\ 
0, & \text{jika } \exists j \in [1, H] \text{ s.t. } Low_{t+j} \le DOWN_t \text{ sebelum } High_{t+j} \ge UP_t \\
0, & \text{jika } \text{hingga hari ke-} H \text{ tidak ada batas yang terpenuhi}
\end{cases}$$

### 3.4 Metode Pelatihan dan Pembagian Data
Untuk menghindari kebocoran informasi masa depan (*look-ahead bias*), pembagian dataset menjadi data latih (*training*) dan data uji (*testing*) dilakukan secara kronologis berdasarkan indeks waktu:
* **Data Latih (80% pertama):** Digunakan untuk melatih pohon-pohon keputusan dalam Random Forest.
* **Data Uji (20% terakhir):** Digunakan untuk menguji keandalan prediksi model pada periode waktu baru (*out-of-sample testing*).

### 3.5 Metrik Evaluasi Model
Model dievaluasi menggunakan matriks evaluasi klasifikasi standar:
* **Accuracy:** Mengukur persentase kebenaran prediksi total.
* **Precision:** Mengukur ketepatan prediksi sinyal sukses (sangat krusial untuk menghindari kerugian dari *false breakout*).
* **Recall (Sensitivity):** Mengukur kemampuan model menangkap peluang breakout sukses yang ada di pasar.
* **F1-Score:** Rata-rata harmonis yang menyeimbangkan Precision dan Recall.
* **Confusion Matrix:** Tabel kontingensi $2 \times 2$ yang memetakan True Positives (TP), False Positives (FP), True Negatives (TN), dan False Negatives (FN).
