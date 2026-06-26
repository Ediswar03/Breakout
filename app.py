import os
import io
import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Coba import library interaktif web
try:
    import streamlit as st
except ImportError:
    st = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class StockBreakoutPredictor:
    def __init__(self, df, window=20, volume_factor=1.5, forward_window=5, tp_pct=0.03, sl_pct=0.02, ml_algo="Random Forest"):
        self.df = df.copy()
        self.window = window
        self.volume_factor = volume_factor
        self.forward_window = forward_window
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.ml_algo = ml_algo
        
        self.candidate_df = None
        self.model = None
        self.feature_cols = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
            'SMA_20_Ratio', 'SMA_50_Ratio', 'Vol_Ratio', 
            'ATR_Norm', 'Momentum_3d', 'Momentum_5d', 'Price_Vol'
        ]
        
    def calculate_technical_indicators(self):
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']
        volume = self.df['Volume']
        
        # 1. RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. MACD
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema_fast - ema_slow
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']
        
        # 3. SMA Ratios
        self.df['SMA_20'] = close.rolling(window=20).mean()
        self.df['SMA_50'] = close.rolling(window=50).mean()
        self.df['SMA_20_Ratio'] = close / self.df['SMA_20']
        self.df['SMA_50_Ratio'] = close / self.df['SMA_50']
        
        # 4. Volume Ratio
        vol_avg = volume.shift(1).rolling(window=20).mean()
        self.df['Vol_Ratio'] = volume / (vol_avg + 1e-9)
        
        # 5. ATR Dinormalisasi
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        self.df['ATR_Norm'] = atr / close
        
        # 6. Momentum
        self.df['Momentum_3d'] = close.pct_change(periods=3)
        self.df['Momentum_5d'] = close.pct_change(periods=5)
        
        # 7. Volatilitas
        self.df['Price_Vol'] = close.pct_change().rolling(window=20).std()
        
        self.df = self.df.bfill()

    def detect_and_label_candidates(self):
        self.df['Resistance'] = self.df['High'].shift(1).rolling(window=self.window).max()
        self.df['Vol_Avg_Rule'] = self.df['Volume'].shift(1).rolling(window=self.window).mean()
        
        self.df['Is_Candidate'] = (self.df['Close'] > self.df['Resistance']) & \
                                  (self.df['Volume'] > self.volume_factor * self.df['Vol_Avg_Rule'])
        
        self.df['Resistance'] = self.df['Resistance'].bfill()
        self.df['Vol_Avg_Rule'] = self.df['Vol_Avg_Rule'].bfill()
        self.df['Is_Candidate'] = self.df['Is_Candidate'].fillna(False)
        
        labels = []
        for i in range(len(self.df)):
            if not self.df['Is_Candidate'].iloc[i]:
                labels.append(np.nan)
                continue
                
            close_ref = self.df['Close'].iloc[i]
            sl_price = close_ref * (1 - self.sl_pct)
            tp_price = close_ref * (1 + self.tp_pct)
            is_true = 0
            
            for j in range(1, self.forward_window + 1):
                if i + j >= len(self.df):
                    break
                high_future = self.df['High'].iloc[i+j]
                low_future = self.df['Low'].iloc[i+j]
                
                if low_future <= sl_price:
                    is_true = 0
                    break
                elif high_future >= tp_price:
                    is_true = 1
                    break
            labels.append(is_true)
            
        self.df['True_Breakout'] = labels
        self.candidate_df = self.df[self.df['Is_Candidate'] == True].copy()

    def train_ml_model(self):
        clean_candidates = self.candidate_df.dropna(subset=self.feature_cols + ['True_Breakout'])
        
        if len(clean_candidates) < 8:
            return None, None, None
            
        X = clean_candidates[self.feature_cols]
        y = clean_candidates['True_Breakout'].astype(int)
        
        # Pemisahan kronologis 80/20 — SELALU gunakan data berbeda untuk uji
        split_idx = int(len(X) * 0.8)
        # Pastikan minimal 2 data di set uji
        split_idx = min(split_idx, len(X) - 2)
        split_idx = max(split_idx, 4)  # minimal 4 data latih
        
        if split_idx >= len(X):
            # Data terlalu sedikit untuk dipisah secara bermakna
            return None, None, None
            
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
        if self.ml_algo == "Random Forest":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        elif self.ml_algo == "Gradient Boosting":
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        elif self.ml_algo == "SVM":
            self.model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        elif self.ml_algo == "Logistic Regression":
            self.model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        elif self.ml_algo == "Naive Bayes":
            self.model = GaussianNB()
        elif self.ml_algo == "XGBoost" and HAS_XGBOOST:
            self.model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
            
        self.model.fit(X_train, y_train)
        
        self.candidate_df['ML_Prediction'] = np.nan
        self.candidate_df['Tipe_Data'] = 'Tidak Digunakan'
        if hasattr(self.model, "predict"):
            self.candidate_df.loc[clean_candidates.index, 'ML_Prediction'] = self.model.predict(X)
            self.candidate_df.loc[X_train.index, 'Tipe_Data'] = 'Data Latih (Train)'
            self.candidate_df.loc[X_test.index, 'Tipe_Data'] = 'Data Uji (Test)'
        
        y_pred = self.model.predict(X_test)
        
        if len(self.model.classes_) > 1:
            y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        else:
            single_class = self.model.classes_[0]
            y_pred_prob = np.ones(len(X_test)) * float(single_class)
            
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.zeros(len(self.feature_cols))
            
        feat_importance = pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)
        
        test_results = pd.DataFrame(index=X_test.index)
        test_results['Actual'] = y_test
        test_results['Predicted'] = y_pred
        test_results['Probability'] = y_pred_prob
        
        return metrics, feat_importance, test_results


@st.cache_data(ttl=1800)
def load_yfinance_data(ticker, start, end):
    try:
        raw_df = yf.download(ticker, start=start, end=end)
        if len(raw_df) > 10:
            if isinstance(raw_df.columns, pd.MultiIndex):
                # Try level 0 first
                cols_l0 = raw_df.columns.get_level_values(0)
                if 'Close' in cols_l0 or 'close' in [c.lower() for c in cols_l0]:
                    raw_df.columns = cols_l0
                else:
                    # Try level 1 if Close is not in level 0
                    cols_l1 = raw_df.columns.get_level_values(1)
                    raw_df.columns = cols_l1
            
            # Standardize column names (strip whitespace and Capitalize)
            raw_df.columns = [str(c).strip().capitalize() for c in raw_df.columns]
            return raw_df
        return None
    except Exception:
        return None


def generate_synthetic_data(days=600):
    np.random.seed(42)
    start_date = datetime.date(2022, 1, 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    
    price = 100.0
    prices, volumes = [], []
    
    for i in range(days):
        drift = 0.05
        noise = np.random.normal(0, 1.2)
        price = max(15.0, price + drift + noise)
        prices.append(price)
        
        base_vol = 150000
        vol_noise = np.random.normal(0, 25000)
        vol = max(10000, base_vol + vol_noise)
        volumes.append(vol)
        
    df = pd.DataFrame(index=dates)
    df.index.name = 'Date'
    df['Close'] = prices
    df['Open'] = df['Close'] * (1 + np.random.normal(0, 0.003, days))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, days)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, days)))
    df['Volume'] = volumes
    
    for i in range(40, days - 10):
        recent_high = max(df['High'].iloc[max(0, i-20):i])
        if df['Close'].iloc[i] > recent_high * 0.98 and np.random.rand() < 0.1:
            df.loc[df.index[i], 'Close'] = recent_high * 1.03
            df.loc[df.index[i], 'High'] = df['Close'].iloc[i] * 1.005
            df.loc[df.index[i], 'Volume'] = df['Volume'].iloc[i] * 2.3
            
            is_true = np.random.rand() < 0.6
            for j in range(1, 6):
                if i+j >= days: break
                if is_true:
                    df.loc[df.index[i+j], 'Close'] = df['Close'].iloc[i] * (1 + 0.01 * j + np.random.normal(0, 0.003))
                else:
                    df.loc[df.index[i+j], 'Close'] = df['Close'].iloc[i] * (1 - 0.01 * j + np.random.normal(0, 0.003))
                df.loc[df.index[i+j], 'Open'] = df['Close'].iloc[i+j] * (1 + np.random.normal(0, 0.002))
                df.loc[df.index[i+j], 'High'] = df[['Open', 'Close']].iloc[i+j].max() * (1 + np.abs(np.random.normal(0, 0.003)))
                df.loc[df.index[i+j], 'Low'] = df[['Open', 'Close']].iloc[i+j].min() * (1 - np.abs(np.random.normal(0, 0.003)))
                df.loc[df.index[i+j], 'Volume'] = df['Volume'].iloc[i+j] * (1.2 if is_true else 0.8)
                
    return df


def generate_html_report(stock_name, ticker_code, metrics, show_df, true_b, false_b):
    html_content = f"""
    <html>
    <head>
        <title>Laporan Deteksi Breakout Saham - {stock_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; color: #333; }}
            h1 {{ color: #111; border-bottom: 2px solid #333; padding-bottom: 10px; }}
            .info {{ margin-bottom: 20px; }}
            .info td {{ padding: 5px 15px 5px 0; }}
            .metrics-table, .signals-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            .metrics-table th, .metrics-table td, .signals-table th, .signals-table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
            .metrics-table th, .signals-table th {{ background-color: #f2f2f2; }}
            .badge-true {{ color: green; font-weight: bold; }}
            .badge-false {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body onload="window.print()">
        <h1>Laporan Deteksi & Prediksi Breakout Saham (Hybrid Model)</h1>
        <div class="info">
            <table>
                <tr><td><b>Kode Saham:</b></td><td>{ticker_code}</td></tr>
                <tr><td><b>Nama Saham:</b></td><td>{stock_name}</td></tr>
                <tr><td><b>Tanggal Cetak:</b></td><td>{datetime.datetime.now().strftime('%d-%m-%Y')}</td></tr>
            </table>
        </div>
        
        <h2>Evaluasi Performa Model ML</h2>
        <table class="metrics-table">
            <tr><th>Metrik</th><th>Nilai</th></tr>
            <tr><td>Akurasi (Accuracy)</td><td>{f"{metrics['accuracy']*100:.2f}%" if metrics else 'N/A'}</td></tr>
            <tr><td>Presisi (Precision)</td><td>{f"{metrics['precision']*100:.2f}%" if metrics else 'N/A'}</td></tr>
            <tr><td>Sensitivitas (Recall)</td><td>{f"{metrics['recall']*100:.2f}%" if metrics else 'N/A'}</td></tr>
            <tr><td>F1-Score</td><td>{f"{metrics['f1']*100:.2f}%" if metrics else 'N/A'}</td></tr>
            <tr><td>Total Sinyal True Breakout</td><td>{true_b} kali</td></tr>
            <tr><td>Total Sinyal False Breakout</td><td>{false_b} kali</td></tr>
        </table>
        
        <h2>Daftar Sinyal Breakout</h2>
        <table class="signals-table">
            <thead>
                <tr>
                    <th>No</th>
                    <th>Tanggal</th>
                    <th>Harga Penutupan (Close)</th>
                    <th>Tipe Data</th>
                    <th>Sinyal Aktual</th>
                    <th>Prediksi ML</th>
                    <th>Keterangan</th>
                </tr>
            </thead>
            <tbody>
    """
    for idx, row in show_df.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['No']}</td>
                    <td>{row['Tanggal']}</td>
                    <td>{row['Harga']}</td>
                    <td>{row.get('Tipe Data', '')}</td>
                    <td>{row.get('Sinyal Aktual', '')}</td>
                    <td>{row.get('Prediksi ML', '')}</td>
                    <td>{row['Keterangan']}</td>
                </tr>
        """
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    return html_content


def parse_uploaded_csv(uploaded_file, selected_kode=None):
    # Detect delimiter (comma vs semicolon)
    first_line = uploaded_file.readline().decode('utf-8', errors='ignore')
    uploaded_file.seek(0)
    sep = ';' if ';' in first_line else ','
    
    df = pd.read_csv(uploaded_file, sep=sep)
    
    # Normalize column names to lowercase and strip whitespace
    df.columns = [c.strip().lower() for c in df.columns]
    
    # If it is a multi-stock CSV (contains 'kode') and a specific stock is selected
    if 'kode' in df.columns and selected_kode is not None:
        # Convert selected_kode to lowercase for case-insensitive match
        df = df[df['kode'].astype(str).str.lower() == str(selected_kode).lower()].copy()
        
    # Column mapping dictionary
    mapping = {
        'tanggal': 'Date',
        'date': 'Date',
        'open_price': 'Open',
        'open': 'Open',
        'high_price': 'High',
        'high': 'High',
        'low_price': 'Low',
        'low': 'Low',
        'close_price': 'Close',
        'close': 'Close',
        'volume': 'Volume'
    }
    
    # Rename columns if found in mapping
    rename_dict = {}
    for col in df.columns:
        if col in mapping:
            rename_dict[col] = mapping[col]
            
    df.rename(columns=rename_dict, inplace=True)
    
    # Ensure column capitalization for any other standard columns
    remaining_cols = {col: col.capitalize() for col in df.columns if col not in rename_dict.values()}
    df.rename(columns=remaining_cols, inplace=True)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # Sort by Date ascending
        df.sort_index(inplace=True)
        
    return df


def main():
    if st is None:
        print("[ERROR] Streamlit tidak terinstall.")
        return

    st.set_page_config(page_title="Deteksi Pola Breakout Saham Hybrid", layout="wide", page_icon="📈")
    
    # Custom CSS untuk layout premium
    st.markdown("""
        <style>
        .sidebar-section {
            font-weight: bold;
            font-size: 14px;
            margin-top: 18px;
            margin-bottom: 6px;
            color: #1565c0;
            border-bottom: 1.5px solid #90caf9;
            padding-bottom: 3px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .legend-box {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 13px;
            line-height: 1.9;
        }
        .success-footer {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
            padding: 10px 15px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        /* Tombol Proses biru */
        div[data-testid="stSidebar"] div.stButton > button {
            background-color: #1976d2;
            color: white;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.2s;
        }
        div[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #1565c0;
            color: white;
        }
        /* Tombol Simpan Hasil hijau */
        .btn-simpan > button {
            background-color: #2e7d32 !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            border-radius: 5px !important;
        }
        /* Tombol Keluar abu-abu */
        .btn-keluar > button {
            background-color: #546e7a !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
            border-radius: 5px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar: PENGATURAN
    st.sidebar.markdown("<div class='sidebar-section'>PENGATURAN</div>", unsafe_allow_html=True)
    
    # Mode Sumber Data
    source_type = st.sidebar.selectbox("Pilih Sumber Data", ["Yahoo Finance (Unduh)", "Unggah Berkas CSV", "Simulasi Data Buatan"])
    
    STOCK_OPTIONS = {
        "BBCA - Bank Central Asia Tbk": "BBCA.JK",
        "BBRI - Bank Rakyat Indonesia Tbk": "BBRI.JK",
        "BMRI - Bank Mandiri Tbk": "BMRI.JK",
        "BBNI - Bank Negara Indonesia Tbk": "BBNI.JK",
        "TLKM - Telkom Indonesia Tbk": "TLKM.JK",
        "ASII - Astra International Tbk": "ASII.JK",
        "UNVR - Unilever Indonesia Tbk": "UNVR.JK",
        "GOTO - GoTo Gojek Tokopedia Tbk": "GOTO.JK",
        "ADRO - Adaro Energy Indonesia Tbk": "ADRO.JK",
        "PGAS - Perusahaan Gas Negara Tbk": "PGAS.JK",
        "Kustom Ticker (Ketik Manual)": "CUSTOM"
    }
    
    ticker_code = "SIMULATION"
    stock_name = "Simulasi Saham Buatan"
    stock_sector = "Teknologi"
    
    # Pilihan parameter input
    uploaded_files = None
    if source_type == "Yahoo Finance (Unduh)":
        selected_stock_label = st.sidebar.selectbox("Pilih Saham", list(STOCK_OPTIONS.keys()))
        ticker_code = STOCK_OPTIONS[selected_stock_label]
        if ticker_code == "CUSTOM":
            ticker_code = st.sidebar.text_input("Ketik Ticker Saham (Yahoo Finance)", value="AAPL")
            stock_name = ticker_code
            stock_sector = "Kustom"
        else:
            stock_name = selected_stock_label.split(" - ")[-1]
            stock_sector = "Keuangan" if "Bank" in selected_stock_label or "BBNI" in selected_stock_label else "Komunikasi / Energi"
    elif source_type == "Unggah Berkas CSV":
        uploaded_files = st.sidebar.file_uploader("Unggah File CSV Saham", type=["csv"], accept_multiple_files=True)
        stock_name = "Berkas Unggahan CSV"
        stock_sector = "Statistik / Kustom"
        ticker_code = "CSV_UPLOAD"
        selected_kode = None
        
        csv_min_date = datetime.date(2022, 1, 1)
        csv_max_date = datetime.date.today()
        
        if uploaded_files:
            try:
                all_unique_codes = set()
                all_date_series = []
                
                # Scan files for codes
                for f in uploaded_files:
                    first_line = f.readline().decode('utf-8', errors='ignore')
                    f.seek(0)
                    sep = ';' if ';' in first_line else ','
                    
                    header_df = pd.read_csv(f, sep=sep, nrows=1)
                    f.seek(0)
                    header_cols = [c.strip().lower() for c in header_df.columns]
                    
                    if 'kode' in header_cols:
                        kode_col_orig = header_df.columns[header_cols.index('kode')]
                        full_df_kode = pd.read_csv(f, sep=sep, usecols=[kode_col_orig])
                        f.seek(0)
                        all_unique_codes.update(full_df_kode[kode_col_orig].dropna().unique())
                
                if all_unique_codes:
                    unique_codes = sorted(list(all_unique_codes))
                    selected_kode = st.sidebar.selectbox("Pilih Kode Saham", unique_codes)
                    
                    ticker_code = selected_kode.upper()
                    stock_name = f"Saham {ticker_code} (CSV)"
                    stock_sector = "Unggahan CSV"
                
                # Scan files for dates of selected stock
                for f in uploaded_files:
                    first_line = f.readline().decode('utf-8', errors='ignore')
                    f.seek(0)
                    sep = ';' if ';' in first_line else ','
                    header_df = pd.read_csv(f, sep=sep, nrows=1)
                    f.seek(0)
                    header_cols = [c.strip().lower() for c in header_df.columns]
                    
                    date_col_name = None
                    for col in header_df.columns:
                        if col.strip().lower() in ['tanggal', 'date']:
                            date_col_name = col
                            break
                            
                    if date_col_name is not None:
                        use_cols = [date_col_name]
                        if 'kode' in header_cols:
                            use_cols.append(header_df.columns[header_cols.index('kode')])
                            
                        date_df = pd.read_csv(f, sep=sep, usecols=use_cols)
                        f.seek(0)
                        
                        if 'kode' in header_cols and selected_kode is not None:
                            date_df = date_df[date_df[header_df.columns[header_cols.index('kode')]].astype(str).str.lower() == str(selected_kode).lower()]
                            
                        date_series = pd.to_datetime(date_df[date_col_name], errors='coerce').dropna()
                        if not date_series.empty:
                            all_date_series.append(date_series)
                            
                if all_date_series:
                    combined_dates = pd.concat(all_date_series)
                    csv_min_date = combined_dates.min().date()
                    csv_max_date = combined_dates.max().date()
            except Exception as e:
                st.sidebar.error(f"Gagal mendeteksi kolom CSV: {e}")
        
    filenames_str = "_".join([f.name for f in uploaded_files]) if (source_type == "Unggah Berkas CSV" and uploaded_files) else "empty"
    start_key = f"start_{source_type}_{filenames_str}"
    end_key = f"end_{source_type}_{filenames_str}"

    col_d1, col_d2 = st.sidebar.columns([0.4, 0.6])
    with col_d1:
        st.markdown("<div style='margin-top: 8px; font-size: 14px; font-weight: 500;'>Tanggal Mulai</div>", unsafe_allow_html=True)
    with col_d2:
        default_start = csv_min_date if source_type == "Unggah Berkas CSV" and uploaded_files else datetime.date(2022, 1, 1)
        start_date = st.date_input("Tanggal Mulai", default_start, label_visibility="collapsed", key=start_key)
        
    col_d3, col_d4 = st.sidebar.columns([0.4, 0.6])
    with col_d3:
        st.markdown("<div style='margin-top: 8px; font-size: 14px; font-weight: 500;'>Tanggal Akhir</div>", unsafe_allow_html=True)
    with col_d4:
        default_end = csv_max_date if source_type == "Unggah Berkas CSV" and uploaded_files else datetime.date.today()
        end_date = st.date_input("Tanggal Akhir", default_end, label_visibility="collapsed", key=end_key)
    
    # Tombol Proses untuk memicu reload
    proses_clicked = st.sidebar.button("Proses", use_container_width=True)
    
    # Inisialisasi session state untuk data
    if 'data_df' not in st.session_state or proses_clicked:
        if source_type == "Yahoo Finance (Unduh)":
            raw_data = load_yfinance_data(ticker_code, start_date, end_date)
            st.session_state['data_df'] = raw_data
        elif source_type == "Unggah Berkas CSV" and uploaded_files:
            try:
                temp_dfs = []
                for f in uploaded_files:
                    try:
                        temp_df = parse_uploaded_csv(f, selected_kode=selected_kode)
                        if temp_df is not None and not temp_df.empty:
                            temp_dfs.append(temp_df)
                    except Exception as parse_err:
                        st.sidebar.warning(f"File {f.name} dilewati: {parse_err}")
                
                if temp_dfs:
                    combined_df = pd.concat(temp_dfs)
                    # Deduplicate indices
                    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                    # Sort chronologically
                    combined_df = combined_df.sort_index()
                    st.session_state['data_df'] = combined_df
                else:
                    st.session_state['data_df'] = None
                    st.sidebar.error("Tidak ada data valid yang dapat digabungkan dari berkas CSV.")
            except Exception as e:
                st.session_state['data_df'] = None
                st.sidebar.error(f"Gagal memproses berkas CSV: {e}")
        elif source_type == "Simulasi Data Buatan":
            # Generate total days
            days = (end_date - start_date).days
            days = max(100, min(1000, days))
            st.session_state['data_df'] = generate_synthetic_data(days=days)
        else:
            if 'data_df' not in st.session_state:
                # Fallback awal agar aplikasi tidak kosong
                st.session_state['data_df'] = generate_synthetic_data()

    # Load data dari session state
    df = st.session_state.get('data_df', None)
    
    if df is None:
        st.warning("⚠️ Silakan lengkapi pengaturan sumber data dan klik tombol 'Proses' di sidebar kiri.")
        return
        
    # Filter data berdasarkan rentang tanggal yang dipilih di sidebar
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
    except Exception as e:
        st.sidebar.warning(f"Gagal memfilter rentang tanggal: {e}")
        
    if len(df) < 10:
        st.warning("⚠️ Data pada rentang tanggal yang dipilih terlalu sedikit (minimal 10 baris). Silakan sesuaikan kembali rentang tanggal di sidebar kiri.")
        return

    # Parameter Algoritma di Sidebar
    st.sidebar.markdown("<div class='sidebar-section'>PARAMETER ALGORITMA</div>", unsafe_allow_html=True)
    window = st.sidebar.slider("Rolling Window Resistensi", 5, 40, 20)
    vol_factor = st.sidebar.slider("Faktor Pengali Volume", 1.0, 3.0, 1.5, step=0.1)
    
    st.sidebar.markdown("<div class='sidebar-section'>PARAMETER EVALUASI ML</div>", unsafe_allow_html=True)
    forward_window = st.sidebar.slider("Horizon Evaluasi (Hari)", 3, 10, 5)
    tp_pct = st.sidebar.slider("Take Profit (%)", 1.0, 10.0, 3.0, step=0.5) / 100.0
    sl_pct = st.sidebar.slider("Stop Loss (%)", 1.0, 5.0, 2.0, step=0.5) / 100.0
    algo_list = ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression", "Naive Bayes"]
    if HAS_XGBOOST:
        algo_list.append("XGBoost")
    ml_algo = st.sidebar.selectbox("Pilih Algoritma ML", algo_list)

    # Jalankan analisis
    predictor = StockBreakoutPredictor(
        df=df,
        window=window,
        volume_factor=vol_factor,
        forward_window=forward_window,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        ml_algo=ml_algo
    )
    predictor.calculate_technical_indicators()
    predictor.detect_and_label_candidates()
    
    # DEBUG LOGGING TO FILE
    try:
        with open("debug_log.txt", "w") as f:
            f.write(f"DF shape: {df.shape}\n")
            f.write(f"DF columns: {list(df.columns)}\n")
            f.write(f"DF index: {df.index.name}, type: {type(df.index)}\n")
            f.write(f"Candidate DF shape: {predictor.candidate_df.shape}\n")
            f.write(f"Is_Candidate count: {predictor.df['Is_Candidate'].sum()}\n")
            f.write("\nFirst 20 rows of df:\n")
            f.write(predictor.df[['Close', 'High', 'Volume', 'Resistance', 'Vol_Avg_Rule', 'Is_Candidate']].head(20).to_string())
            f.write("\n\nRows where Close > Resistance:\n")
            c_high = predictor.df[predictor.df['Close'] > predictor.df['Resistance']]
            f.write(f"Count: {len(c_high)}\n")
            f.write(c_high[['Close', 'Resistance', 'Volume', 'Vol_Avg_Rule']].head(10).to_string())
            f.write("\n\nRows where Volume > Factor * Vol_Avg:\n")
            c_vol = predictor.df[predictor.df['Volume'] > predictor.df['Vol_Avg_Rule'] * vol_factor]
            f.write(f"Count: {len(c_vol)}\n")
            f.write(c_vol[['Close', 'Resistance', 'Volume', 'Vol_Avg_Rule']].head(10).to_string())
    except Exception as ex:
        pass

    metrics, feat_imp, test_results = predictor.train_ml_model()
    
    # Stats Saham Sidebar
    latest_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
    change = latest_close - prev_close
    pct_change = (change / prev_close) * 100
    latest_date_str = df.index[-1].strftime('%d %B %Y')
    
    st.sidebar.markdown("<div class='sidebar-section'>INFORMASI SAHAM</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Kode Saham** : `{ticker_code}`")
    st.sidebar.markdown(f"**Nama** : {stock_name}")
    st.sidebar.markdown(f"**Sektor** : {stock_sector}")
    st.sidebar.markdown(f"**Harga Terakhir** : **{latest_close:,.2f}**")
    change_color = "green" if change >= 0 else "red"
    change_sign = "+" if change >= 0 else ""
    st.sidebar.markdown(f"**Perubahan** : <span style='color:{change_color}; font-weight:bold;'>{change_sign}{change:,.2f} ({change_sign}{pct_change:.2f}%)</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Tanggal** : {latest_date_str}")
    
    # Hitung True vs False Breakout
    true_b = len(predictor.candidate_df[predictor.candidate_df['True_Breakout'] == 1])
    false_b = len(predictor.candidate_df[predictor.candidate_df['True_Breakout'] == 0])
    acc_text = f"{metrics['accuracy']*100:.2f}%" if metrics is not None else "N/A"
    
    st.sidebar.markdown("<div class='sidebar-section'>HASIL DETEKSI</div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"🟢 **True Breakout** : <span style='color:green; font-weight:bold;'>{true_b} kali</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"🔴 **False Breakout** : <span style='color:red; font-weight:bold;'>{false_b} kali</span>", unsafe_allow_html=True)
    st.sidebar.markdown(f"⚫ **Akurasi Sinyal** : <span style='color:#1f77b4; font-weight:bold;'>{acc_text}</span>", unsafe_allow_html=True)
    
    # Keterangan
    st.sidebar.markdown("<div class='sidebar-section'>KETERANGAN</div>", unsafe_allow_html=True)
    st.sidebar.markdown("""
        <div class='legend-box'>
        <span style='color:#1f77b4; font-weight:bold;'>━</span> Harga Close<br>
        <span style='color:#ff7f0e; font-weight:bold;'>---</span> Resistensi (Rolling High)<br>
        <span style='color:#2ca02c; font-weight:bold;'>▲</span> True Breakout (Beli)<br>
        <span style='color:#d62728; font-weight:bold;'>▼</span> False Breakout (Abaikan)
        </div>
    """, unsafe_allow_html=True)

    # LAYOUT UTAMA
    st.markdown(f"### Grafik Deteksi Breakout Saham - {stock_name}")
    
    tab_grafik, tab_ml = st.tabs(["📈 Grafik & Sinyal", f"🧠 Performa {ml_algo}"])
    
    # Sinyal Table Data Preparation
    show_df = pd.DataFrame()
    if len(predictor.candidate_df) > 0:
        show_df['No'] = list(range(1, len(predictor.candidate_df) + 1))
        show_df['Tanggal'] = pd.to_datetime(predictor.candidate_df.index).strftime('%d %b %Y')
        show_df['Sinyal Aktual'] = predictor.candidate_df['True_Breakout'].map({1.0: 'True Breakout', 0.0: 'False Breakout'}).values
        if 'Tipe_Data' in predictor.candidate_df.columns:
            show_df['Tipe Data'] = predictor.candidate_df['Tipe_Data'].values
        else:
            show_df['Tipe Data'] = 'Tidak Digunakan'
            
        if 'ML_Prediction' in predictor.candidate_df.columns:
            show_df['Prediksi ML'] = predictor.candidate_df['ML_Prediction'].map({1.0: 'Beli (Naik)', 0.0: 'Abaikan (Jebakan)', np.nan: 'Belum Diprediksi'}).values
        else:
            show_df['Prediksi ML'] = 'Belum Diprediksi'
        show_df['Harga'] = [f"Rp {x:,.2f}" for x in predictor.candidate_df['Close'].values]
        # Tambahkan nilai Feature Engineering: RSI dan SMA Ratio
        show_df['RSI'] = [f"{x:.1f}" for x in predictor.candidate_df['RSI'].values]
        show_df['Jarak thdp MA20'] = [f"{(x - 1) * 100:+.1f}%" for x in predictor.candidate_df['SMA_20_Ratio'].values]
        
        # Tambahkan Keterangan detail
        show_df['Keterangan'] = [
            f"Breakout & Vol {row['Vol_Ratio']:.1f}x"
            for _, row in predictor.candidate_df.iterrows()
        ]
        
    with tab_grafik:
        # Plot dengan Plotly
        if HAS_PLOTLY:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.08, row_heights=[0.75, 0.25])
            
            # Subplot 1: Price Line
            fig.add_trace(go.Scatter(x=predictor.df.index, y=predictor.df['Close'], 
                                     name='Harga Close', line=dict(color='#1f77b4', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=predictor.df.index, y=predictor.df['Resistance'], 
                                     name='Resistensi', line=dict(color='#ff7f0e', dash='dash', width=1.5)), row=1, col=1)
            
            # Plot Sinyal
            if len(predictor.candidate_df) > 0:
                c_true = predictor.df[(predictor.df['Is_Candidate'] == True) & (predictor.df['True_Breakout'] == 1)]
                c_false = predictor.df[(predictor.df['Is_Candidate'] == True) & (predictor.df['True_Breakout'] == 0)]
                
                fig.add_trace(go.Scatter(x=c_true.index, y=c_true['Close'], mode='markers',
                                         marker=dict(symbol='triangle-up', color='#2ca02c', size=13, line=dict(width=1, color='black')),
                                         name='True Breakout'), row=1, col=1)
                fig.add_trace(go.Scatter(x=c_false.index, y=c_false['Close'], mode='markers',
                                         marker=dict(symbol='triangle-down', color='#d62728', size=13, line=dict(width=1, color='black')),
                                         name='False Breakout (Trap)'), row=1, col=1)
            
            # Subplot 2: Volume
            fig.add_trace(go.Bar(x=predictor.df.index, y=predictor.df['Volume'], 
                                 name='Volume', marker=dict(color='#94a3b8', opacity=0.8)), row=2, col=1)
            
            # Tambahkan zoom range selector buttons sesuai screenshot
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            )
            
            fig.update_layout(height=480, hovermode='x unified', margin=dict(t=5, b=5),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Plotly tidak tersedia. Tampilan grafik tidak dapat ditampilkan secara interaktif.")
            
        # Layout Dua Panel (Daftar Sinyal & Evaluasi Akurasi)
        col_list, col_eval = st.columns([0.65, 0.35])
        
        with col_list:
            st.markdown("<div class='sidebar-section' style='font-size:15px; margin-top:5px;'>DAFTAR SINYAL</div>", unsafe_allow_html=True)
            if len(show_df) > 0:
                # Terapkan gaya warna teks pada kolom Sinyal
                def style_signal_col(val):
                    if val in ['True Breakout', 'Beli (Naik)']:
                        return 'color: green; font-weight: bold;'
                    elif val in ['False Breakout', 'Abaikan (Jebakan)']:
                        return 'color: red; font-weight: bold;'
                    return ''
                
                styled_show_df = show_df.style.map(style_signal_col, subset=['Sinyal Aktual', 'Prediksi ML'])
                
                st.dataframe(
                    styled_show_df,
                    column_config={
                        "No": st.column_config.NumberColumn("No", width="small"),
                        "Tanggal": st.column_config.TextColumn("Tanggal", width="medium"),
                        "Tipe Data": st.column_config.TextColumn("Tipe Data", width="small"),
                        "Sinyal Aktual": st.column_config.TextColumn("Sinyal Aktual", width="medium"),
                        "Prediksi ML": st.column_config.TextColumn("Prediksi ML", width="medium"),
                        "Harga": st.column_config.TextColumn("Harga", width="small"),
                        "RSI": st.column_config.TextColumn("RSI", width="small"),
                        "Jarak thdp MA20": st.column_config.TextColumn("Jarak MA20", width="small"),
                        "Keterangan": st.column_config.TextColumn("Info", width="medium")
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("Tidak ada sinyal breakout yang terdeteksi.")
                
        with col_eval:
            st.markdown("<div class='sidebar-section' style='font-size:15px; margin-top:5px;'>EVALUASI AKURASI</div>", unsafe_allow_html=True)
            # Akurasi dari data Uji saja (menghindari bias 100% pada data Latih)
            if 'ML_Prediction' in predictor.candidate_df.columns and 'Tipe_Data' in predictor.candidate_df.columns:
                valid_preds = predictor.candidate_df[(predictor.candidate_df['Tipe_Data'] == 'Data Uji (Test)')].dropna(subset=['True_Breakout', 'ML_Prediction'])
                tot_sig = len(valid_preds)
                if tot_sig > 0:
                    correct_sig = int((valid_preds['True_Breakout'] == valid_preds['ML_Prediction']).sum())
                    wrong_sig = tot_sig - correct_sig
                    acc_val = correct_sig / tot_sig
                else:
                    correct_sig, wrong_sig, acc_val = 0, 0, 0.0
            elif metrics is not None and test_results is not None and len(test_results) > 0:
                tot_sig = len(test_results)
                correct_sig = int((test_results['Actual'] == test_results['Predicted']).sum())
                wrong_sig = tot_sig - correct_sig
                acc_val = correct_sig / tot_sig if tot_sig > 0 else 0.0
            else:
                # Fallback: gunakan label aktual dari seluruh kandidat
                tot_sig = len(predictor.candidate_df)
                correct_sig = int(true_b)
                wrong_sig = int(false_b)
                acc_val = (correct_sig / tot_sig) if tot_sig > 0 else 0.0
                
            eval_rows = {
                "Metrik": ["Total Sinyal Dievaluasi", "Prediksi Tepat", "Prediksi Keliru", "Akurasi Keseluruhan"],
                "Nilai": [str(tot_sig), str(correct_sig), str(wrong_sig), f"{acc_val*100:.2f}%"]
            }
            eval_df = pd.DataFrame(eval_rows)
            
            # Terapkan gaya warna teks hijau tebal khusus untuk nilai Akurasi
            def style_eval_accuracy(row):
                styles = [''] * len(row)
                if row['Metrik'] == 'Akurasi Keseluruhan':
                    styles[1] = 'color: green; font-weight: bold;'
                return styles
            
            styled_eval_df = eval_df.style.apply(style_eval_accuracy, axis=1)
            
            st.dataframe(styled_eval_df, hide_index=True, use_container_width=True)
            
            # Pie Chart Akurasi
            if tot_sig > 0:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['Prediksi Tepat', 'Prediksi Keliru'],
                    values=[correct_sig, wrong_sig],
                    hole=.3,
                    marker=dict(colors=['#2ca02c', '#d62728']),
                    textinfo='percent'
                )])
                fig_pie.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Tidak ada data sinyal untuk pie chart.")
                
    with tab_ml:
        st.subheader(f"🧠 Performa Klasifikasi {ml_algo} (Out-of-Sample)")
        if metrics is None:
            st.warning("⚠️ Data latihan terlalu sedikit untuk menguji performa model Machine Learning.")
        else:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Akurasi (Accuracy)", f"{metrics['accuracy']*100:.2f}%")
            col_m2.metric("Presisi (Precision)", f"{metrics['precision']*100:.2f}%")
            col_m3.metric("Sensitivitas (Recall)", f"{metrics['recall']*100:.2f}%")
            col_m4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
            
            sc1, sc2 = st.columns([0.6, 0.4])
            with sc1:
                st.markdown("**Performa Prediksi Model pada Garis Harga Uji**")
                # Grafik visualisasi data uji
                test_start = test_results.index.min()
                test_end = test_results.index.max()
                df_test = predictor.df.loc[test_start:test_end]
                
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=df_test.index, y=df_test['Close'], name='Harga Close (Periode Uji)', line=dict(color='gray', width=1.5)))
                
                tp = test_results[(test_results['Actual'] == 1) & (test_results['Predicted'] == 1)]
                fp = test_results[(test_results['Actual'] == 0) & (test_results['Predicted'] == 1)]
                tn = test_results[(test_results['Actual'] == 0) & (test_results['Predicted'] == 0)]
                fn = test_results[(test_results['Actual'] == 1) & (test_results['Predicted'] == 0)]
                
                fig_t.add_trace(go.Scatter(x=tp.index, y=predictor.df.loc[tp.index, 'Close'], mode='markers',
                                             marker=dict(symbol='triangle-up', color='green', size=13, line=dict(width=1, color='black')),
                                             name='True Positive (Prediksi Naik & Aktual Naik)'))
                fig_t.add_trace(go.Scatter(x=fp.index, y=predictor.df.loc[fp.index, 'Close'], mode='markers',
                                             marker=dict(symbol='triangle-up', color='red', size=13, line=dict(width=1, color='black')),
                                             name='False Positive (Prediksi Naik & Aktual Turun)'))
                fig_t.add_trace(go.Scatter(x=tn.index, y=predictor.df.loc[tn.index, 'Close'], mode='markers',
                                             marker=dict(symbol='circle', color='blue', size=10, line=dict(width=1, color='black')),
                                             name='True Negative (Prediksi Turun & Aktual Turun)'))
                fig_t.add_trace(go.Scatter(x=fn.index, y=predictor.df.loc[fn.index, 'Close'], mode='markers',
                                             marker=dict(symbol='x', color='purple', size=12),
                                             name='False Negative (Prediksi Turun & Aktual Naik)'))
                
                fig_t.update_layout(height=350, margin=dict(t=5, b=5))
                st.plotly_chart(fig_t, use_container_width=True)
                
            with sc2:
                st.markdown("**Urutan Kepentingan Fitur (Feature Importance)**")
                fig_imp = px.bar(feat_imp.reset_index(), x='index', y=0, 
                                 labels={'index': 'Fitur', '0': 'Skor Bobot'},
                                 color_discrete_sequence=['#3b82f6'])
                fig_imp.update_layout(height=320, xaxis_title="", yaxis_title="", margin=dict(t=5, b=5))
                st.plotly_chart(fig_imp, use_container_width=True)
                
            

    # FOOTER INTERAKTIF — seperti gambar referensi
    st.markdown("---")
    col_foot_l, col_foot_r = st.columns([0.6, 0.4])

    with col_foot_l:
        st.markdown(
            f"<div class='success-footer'>✔️ Proses selesai! Total data yang digunakan: <strong>{len(df)} baris</strong></div>",
            unsafe_allow_html=True
        )

    with col_foot_r:
        # Siapkan data ekspor
        csv_bytes = show_df.to_csv(index=False).encode('utf-8') if len(show_df) > 0 else b''

        buffer_excel = io.BytesIO()
        excel_bytes = b''
        if len(show_df) > 0:
            try:
                with pd.ExcelWriter(buffer_excel, engine='openpyxl') as writer:
                    show_df.to_excel(writer, index=False, sheet_name='Sinyal Breakout')
                excel_bytes = buffer_excel.getvalue()
            except Exception:
                excel_bytes = csv_bytes

        html_report_code = ''
        if len(show_df) > 0:
            html_report_code = generate_html_report(stock_name, ticker_code, metrics, show_df, true_b, false_b)

        # Empat tombol: CSV, Excel, PDF, Keluar — dua baris
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            if len(show_df) > 0:
                st.download_button(
                    label="📄 CSV",
                    data=csv_bytes,
                    file_name=f"sinyal_breakout_{ticker_code}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.button("📄 CSV", disabled=True, use_container_width=True)

        with c2:
            if len(show_df) > 0 and excel_bytes:
                st.download_button(
                    label="📊 Excel",
                    data=excel_bytes,
                    file_name=f"sinyal_breakout_{ticker_code}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.button("📊 Excel", disabled=True, use_container_width=True)

        with c3:
            if len(show_df) > 0:
                st.download_button(
                    label="🖨️ PDF",
                    data=html_report_code,
                    file_name=f"laporan_breakout_{ticker_code}.html",
                    mime="text/html",
                    use_container_width=True,
                    help="Buka file HTML di browser lalu tekan Ctrl+P → Simpan sebagai PDF"
                )
            else:
                st.button("🖨️ PDF", disabled=True, use_container_width=True)

        with c4:
            if st.button("🚪 Keluar", use_container_width=True, type="primary", key="btn_keluar"):
                st.session_state.clear()
                st.rerun()


if __name__ == '__main__':
    main()
