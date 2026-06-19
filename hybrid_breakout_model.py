import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class StockBreakoutPredictor:
    """
    Predictor Breakout Hybrid Saham
    Menggabungkan pendekatan Rule-Based untuk deteksi awal kandidat breakout
    dan Machine Learning (Random Forest) untuk memprediksi validitas breakout (True vs False Breakout).
    
    Tim Peneliti UTS:
    - Edisyah Putra Waruwu
    - Marviel David
    - Andri Simbolon
    """
    
    def __init__(self, csv_path, window=20, volume_factor=1.5, forward_window=5, tp_pct=0.03, sl_pct=0.02):
        self.csv_path = csv_path
        self.window = window
        self.volume_factor = volume_factor
        self.forward_window = forward_window
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        
        self.df = None
        self.candidate_df = None
        self.model = None
        self.feature_cols = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
            'SMA_20_Ratio', 'SMA_50_Ratio', 'Vol_Ratio', 
            'ATR_Norm', 'Momentum_3d', 'Momentum_5d', 'Price_Vol'
        ]
        
    def generate_synthetic_data(self, days=700):
        """
        Menghasilkan data historis saham buatan untuk pengujian program jika file CSV tidak ditemukan.
        Menggunakan model Random Walk dengan drift positif dan menyuntikkan pola breakout yang realistis.
        """
        print("[INFO] Membuat data simulasi saham...")
        np.random.seed(42)
        start_date = datetime.date(2023, 1, 1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
        
        price = 100.0
        prices = []
        volumes = []
        
        # Simulasi harga dan volume dasar
        for i in range(days):
            drift = 0.04  # Tren naik jangka panjang
            noise = np.random.normal(0, 1.2)
            price = max(15.0, price + drift + noise)
            prices.append(price)
            
            base_vol = 150000
            vol_noise = np.random.normal(0, 30000)
            vol = max(20000, base_vol + vol_noise)
            volumes.append(vol)
            
        df = pd.DataFrame(index=dates)
        df.index.name = 'Date'
        df['Close'] = prices
        df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.004, days))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.006, days)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.006, days)))
        df['Volume'] = volumes
        
        # Suntikkan beberapa pola breakout buatan secara acak
        for i in range(50, days - 15):
            recent_high = max(df['High'].iloc[max(0, i-self.window):i])
            # Jika harga mendekati resistensi dan terpilih secara acak
            if df['Close'].iloc[i] > recent_high * 0.97 and np.random.rand() < 0.08:
                # 1. Suntikkan Breakout Candle (Kenaikan tajam + Volume tinggi)
                df.loc[df.index[i], 'Close'] = recent_high * 1.035
                df.loc[df.index[i], 'High'] = df['Close'].iloc[i] * 1.008
                df.loc[df.index[i], 'Volume'] = df['Volume'].iloc[i] * 2.2
                
                # Tentukan secara acak apakah breakout ini akan berhasil (True) atau gagal (False)
                is_true_breakout = np.random.rand() < 0.6  # 60% probabilitas True Breakout
                
                # Suntikkan pergerakan harga ke depan
                for j in range(1, self.forward_window + 2):
                    if i + j >= days:
                        break
                    if is_true_breakout:
                        # Harga terus menguat ke atas
                        df.loc[df.index[i+j], 'Close'] = df['Close'].iloc[i] * (1 + 0.012 * j + np.random.normal(0, 0.004))
                    else:
                        # Bull trap: Harga berbalik turun ke bawah stop-loss
                        df.loc[df.index[i+j], 'Close'] = df['Close'].iloc[i] * (1 - 0.010 * j + np.random.normal(0, 0.004))
                    
                    df.loc[df.index[i+j], 'Open'] = df['Close'].iloc[i+j] * (1 + np.random.normal(0, 0.002))
                    df.loc[df.index[i+j], 'High'] = df[['Open', 'Close']].iloc[i+j].max() * (1 + np.abs(np.random.normal(0, 0.004)))
                    df.loc[df.index[i+j], 'Low'] = df[['Open', 'Close']].iloc[i+j].min() * (1 - np.abs(np.random.normal(0, 0.004)))
                    df.loc[df.index[i+j], 'Volume'] = df['Volume'].iloc[i+j] * (1.3 if is_true_breakout else 0.7)
                    
        df.to_csv(self.csv_path)
        print(f"[INFO] Data simulasi berhasil dibuat dan disimpan ke: {self.csv_path}")

    def load_and_prepare_data(self):
        """
        Memuat data saham dari CSV, melakukan parsing tanggal, dan melakukan validasi kolom.
        """
        if not os.path.exists(self.csv_path):
            print(f"[WARNING] File {self.csv_path} tidak ditemukan.")
            self.generate_synthetic_data()
            
        try:
            self.df = pd.read_csv(self.csv_path)
            # Standarisasi kolom
            self.df.columns = [col.strip().capitalize() for col in self.df.columns]
            
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'])
                self.df.set_index('Date', inplace=True)
            elif self.df.index.name != 'Date':
                # Jika indeks tidak bernama Date, jadikan kolom pertama sebagai index
                self.df.index = pd.to_datetime(self.df.index)
                self.df.index.name = 'Date'
                
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Kolom CSV tidak lengkap. Kolom berikut hilang: {missing_cols}")
                
            # Urutkan berdasarkan tanggal menaik
            self.df.sort_index(inplace=True)
            print(f"[INFO] Data saham berhasil dimuat. Total baris: {len(self.df)}")
            
        except Exception as e:
            print(f"[ERROR] Gagal memuat data: {e}")
            raise e

    def calculate_technical_indicators(self):
        """
        Menghitung indikator teknikal sebagai fitur (features) untuk Machine Learning.
        Dihitung secara manual menggunakan pandas untuk memastikan kejelasan logika matematika.
        """
        print("[INFO] Menghitung indikator teknikal...")
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        volume = self.df['Volume']
        
        # 1. RSI (Relative Strength Index) - Wilder's smoothing
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. MACD (Moving Average Convergence Divergence)
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']
        
        # 3. SMA Ratios (Rasio harga terhadap rata-rata pergerakan sederhana)
        self.df['SMA_20'] = close.rolling(window=20).mean()
        self.df['SMA_50'] = close.rolling(window=50).mean()
        self.df['SMA_20_Ratio'] = close / self.df['SMA_20']
        self.df['SMA_50_Ratio'] = close / self.df['SMA_50']
        
        # 4. Volume Ratio (Volume hari ini dibanding rata-rata 20 hari sebelumnya)
        vol_avg = volume.shift(1).rolling(window=20).mean()
        self.df['Vol_Ratio'] = volume / (vol_avg + 1e-9)
        
        # 5. ATR (Average True Range) Dinormalisasi
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        self.df['ATR_Norm'] = atr / close  # Skala persentase volatilitas terhadap harga
        
        # 6. Momentum Harga (3 Hari & 5 Hari)
        self.df['Momentum_3d'] = close.pct_change(periods=3)
        self.df['Momentum_5d'] = close.pct_change(periods=5)
        
        # 7. Volatilitas Harga (Standard Deviasi dari daily return 20 hari)
        self.df['Price_Vol'] = close.pct_change().rolling(window=20).std()
        
        # Isi nilai NaN awal dengan backward fill/forward fill agar tidak membuang terlalu banyak data
        self.df = self.df.bfill()

    def detect_and_label_candidates(self):
        """
        Langkah 1: Rule-Based Detection untuk mencari kandidat breakout.
        Langkah 2: Melabeli kandidat tersebut menjadi True Breakout (1) atau False Breakout (0).
        """
        print("[INFO] Melakukan Rule-Based Detection dan pelabelan data...")
        
        # Deteksi Resistensi Historis (Nilai tertinggi dalam 'window' hari terakhir, digeser 1 hari)
        self.df['Resistance'] = self.df['High'].shift(1).rolling(window=self.window).max()
        self.df['Vol_Avg_Rule'] = self.df['Volume'].shift(1).rolling(window=self.window).mean()
        
        # Aturan Deteksi Kandidat Breakout: Close menembus resistance DAN Volume di atas rata-rata * volume_factor
        self.df['Is_Candidate'] = (self.df['Close'] > self.df['Resistance']) & \
                                  (self.df['Volume'] > self.volume_factor * self.df['Vol_Avg_Rule'])
        
        # Mengisi nilai awal resistensi yang NaN
        self.df['Resistance'] = self.df['Resistance'].bfill()
        self.df['Vol_Avg_Rule'] = self.df['Vol_Avg_Rule'].bfill()
        self.df['Is_Candidate'] = self.df['Is_Candidate'].fillna(False)
        
        # Pelabelan Forward-Looking (True/False Breakout)
        labels = []
        for i in range(len(self.df)):
            if not self.df['Is_Candidate'].iloc[i]:
                labels.append(np.nan)
                continue
                
            close_ref = self.df['Close'].iloc[i]
            sl_price = close_ref * (1 - self.sl_pct)
            tp_price = close_ref * (1 + self.tp_pct)
            
            is_true = 0  # default: False Breakout
            
            # Periksa pergerakan harga selama forward_window hari ke depan
            for j in range(1, self.forward_window + 1):
                if i + j >= len(self.df):
                    break
                
                high_future = self.df['High'].iloc[i+j]
                low_future = self.df['Low'].iloc[i+j]
                
                # Skenario 1: Menyentuh Stop Loss terlebih dahulu (Gagal)
                if low_future <= sl_price:
                    is_true = 0
                    break
                # Skenario 2: Menyentuh Take Profit terlebih dahulu (Berhasil)
                elif high_future >= tp_price:
                    is_true = 1
                    break
                    
            labels.append(is_true)
            
        self.df['True_Breakout'] = labels
        self.candidate_df = self.df[self.df['Is_Candidate'] == True].copy()
        print(f"[INFO] Deteksi selesai. Jumlah kandidat breakout terdeteksi: {len(self.candidate_df)}")

    def train_ml_model(self):
        """
        Langkah 3: Melatih model Machine Learning pada dataset kandidat breakout.
        Menggunakan Chronological Split untuk memisahkan data latih (80%) dan uji (20%).
        """
        print("[INFO] Melatih model Machine Learning (Random Forest)...")
        # Bersihkan data kandidat dari baris yang mengandung NaN
        clean_candidates = self.candidate_df.dropna(subset=self.feature_cols + ['True_Breakout'])
        
        if len(clean_candidates) < 15:
            print(f"[WARNING] Jumlah data kandidat terlalu sedikit ({len(clean_candidates)}). Model ML membutuhkan data lebih banyak.")
            print("[INFO] Mengubah pembagian data latih/uji menjadi 100% latih untuk demonstrasi.")
            X_train = clean_candidates[self.feature_cols]
            y_train = clean_candidates['True_Breakout'].astype(int)
            X_test, y_test = X_train, y_train
        else:
            X = clean_candidates[self.feature_cols]
            y = clean_candidates['True_Breakout'].astype(int)
            
            # Split secara kronologis (Time Series Split sederhana)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
        # Gunakan Random Forest Classifier dengan penyeimbang bobot kelas
        self.model = RandomForestClassifier(
            n_estimators=150, 
            max_depth=5, 
            random_state=42, 
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Prediksi pada Data Uji
        y_pred = self.model.predict(X_test)
        # Tangani skenario di mana data latihan hanya memiliki 1 kelas target (misal: semuanya 0 atau semuanya 1)
        if len(self.model.classes_) > 1:
            y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        else:
            single_class = self.model.classes_[0]
            y_pred_prob = np.ones(len(X_test)) * float(single_class)
        
        # Evaluasi Metrik
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        
        # Feature Importance
        importances = self.model.feature_importances_
        feat_importance = pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)
        
        # Simpan prediksi test ke DataFrame untuk visualisasi
        test_results = pd.DataFrame(index=X_test.index)
        test_results['Actual'] = y_test
        test_results['Predicted'] = y_pred
        test_results['Probability'] = y_pred_prob
        
        return metrics, feat_importance, test_results

    def generate_visualizations(self, test_results, output_dir="."):
        """
        Menghasilkan visualisasi grafik deteksi breakout dan performa prediksi.
        """
        print("[INFO] Membuat grafik visualisasi...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Grafik Garis Close dengan Deteksi Kandidat Aktual (Keseluruhan Data)
        plt.figure(figsize=(15, 7))
        plt.plot(self.df.index, self.df['Close'], label='Harga Close', color='#1f77b4', alpha=0.8, linewidth=1.5)
        plt.plot(self.df.index, self.df['Resistance'], label=f'Resistensi ({self.window}-day High)', color='#ff7f0e', linestyle='--', alpha=0.7)
        
        # Ambil titik breakout kandidat
        candidates = self.df[self.df['Is_Candidate'] == True]
        true_actual = candidates[candidates['True_Breakout'] == 1]
        false_actual = candidates[candidates['True_Breakout'] == 0]
        
        plt.scatter(true_actual.index, true_actual['Close'], marker='^', color='#2ca02c', s=120, edgecolors='black', label='Aktual: True Breakout (Berhasil)', zorder=5)
        plt.scatter(false_actual.index, false_actual['Close'], marker='v', color='#d62728', s=120, edgecolors='black', label='Aktual: False Breakout (Gagal)', zorder=5)
        
        plt.title('Deteksi Kandidat Breakout (Rule-Based) & Hasil Aktual Historis', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Tanggal', fontsize=12)
        plt.ylabel('Harga (IDR)', fontsize=12)
        plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        path_all = os.path.join(output_dir, 'breakout_detection_all.png')
        plt.savefig(path_all, dpi=300)
        plt.close()
        
        # Plot 2: Performa Model ML pada Data Uji (Out-of-Sample)
        if len(test_results) > 0:
            plt.figure(figsize=(15, 7))
            test_start = test_results.index.min()
            test_end = test_results.index.max()
            
            # Filter chart hanya pada rentang waktu data uji
            df_test_period = self.df.loc[test_start:test_end]
            
            plt.plot(df_test_period.index, df_test_period['Close'], label='Harga Close (Periode Uji)', color='#7f7f7f', alpha=0.8)
            plt.plot(df_test_period.index, df_test_period['Resistance'], label='Resistensi', color='#ff7f0e', linestyle='--', alpha=0.5)
            
            # Kelompokkan hasil uji berdasarkan hasil klasifikasi ML
            tp = test_results[(test_results['Actual'] == 1) & (test_results['Predicted'] == 1)]
            fp = test_results[(test_results['Actual'] == 0) & (test_results['Predicted'] == 1)]
            tn = test_results[(test_results['Actual'] == 0) & (test_results['Predicted'] == 0)]
            fn = test_results[(test_results['Actual'] == 1) & (test_results['Predicted'] == 0)]
            
            # Map ke harga close agar titik terplot tepat pada garis harga
            plt.scatter(tp.index, self.df.loc[tp.index, 'Close'], marker='^', color='#2ca02c', s=140, edgecolors='black', label='True Positive (Model memprediksi True & Aktual Naik)', zorder=5)
            plt.scatter(fp.index, self.df.loc[fp.index, 'Close'], marker='^', color='#d62728', s=140, edgecolors='black', label='False Positive (Model memprediksi True & Aktual Turun)', zorder=5)
            plt.scatter(tn.index, self.df.loc[tn.index, 'Close'], marker='o', color='#1f77b4', s=100, edgecolors='black', label='True Negative (Model memprediksi False & Aktual Turun)', zorder=4)
            plt.scatter(fn.index, self.df.loc[fn.index, 'Close'], marker='x', color='#9467bd', s=120, label='False Negative (Model memprediksi False & Aktual Naik)', zorder=4)
            
            plt.title('Evaluasi Prediksi Model Machine Learning (Random Forest) pada Periode Uji', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Tanggal', fontsize=12)
            plt.ylabel('Harga (IDR)', fontsize=12)
            plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            
            path_test = os.path.join(output_dir, 'breakout_predictions_test.png')
            plt.savefig(path_test, dpi=300)
            plt.close()
            
            print(f"[INFO] Grafik visualisasi berhasil disimpan di folder: {output_dir}")
            return path_all, path_test
            
        return path_all, None

    def run_pipeline(self, output_dir="."):
        """
        Menjalankan seluruh pipeline proses: Load, Feature Engineering, Deteksi, Training, Evaluasi, Visualisasi.
        """
        print("="*60)
        print("STARTING HYBRID STOCK BREAKOUT PREDICTION PIPELINE")
        print("="*60)
        
        self.load_and_prepare_data()
        self.calculate_technical_indicators()
        self.detect_and_label_candidates()
        
        metrics, feat_imp, test_results = self.train_ml_model()
        path_all, path_test = self.generate_visualizations(test_results, output_dir)
        
        print("\n" + "="*20 + " METRIK EVALUASI MODEL ML " + "="*20)
        print(f"Akurasi (Accuracy)   : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Presisi (Precision)  : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Sensitivitas (Recall): {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score             : {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print("\n" + "="*20 + " FITUR PALING BERPENGARUH (FEATURE IMPORTANCE) " + "="*20)
        for idx, (feat, val) in enumerate(feat_imp.items()):
            print(f"{idx+1:2d}. {feat:<15} : {val:.4f} ({val*100:.2f}%)")
            
        print("="*60)
        print("PIPELINE EXECUTED SUCCESSFULLY")
        print("="*60)
        return metrics, feat_imp, test_results


if __name__ == '__main__':
    # Tentukan path file data
    csv_file_path = "saham_data.csv"
    output_directory = "outputs"
    
    # Inisialisasi dan jalankan model predictor
    # window=20 (Resistensi dihitung dari 20 hari tertinggi)
    # volume_factor=1.5 (Volume tembus harus > 1.5 kali rata-rata 20 hari)
    # forward_window=5 (Mengevaluasi hasil dalam 5 hari bursa ke depan)
    # tp_pct=0.03 (Target keuntungan 3% untuk dilabeli sebagai True Breakout)
    # sl_pct=0.02 (Stop-loss 2% di mana jika tertembus dianggap False Breakout)
    predictor = StockBreakoutPredictor(
        csv_path=csv_file_path,
        window=20,
        volume_factor=1.5,
        forward_window=5,
        tp_pct=0.03,
        sl_pct=0.02
    )
    
    # Jalankan pipeline
    predictor.run_pipeline(output_dir=output_directory)
