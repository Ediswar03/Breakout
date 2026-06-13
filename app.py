import os
import io
import datetime
import base64
import functools
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class StockBreakoutPredictor:
    def __init__(self, df, window=20, volume_factor=1.5, forward_window=5, tp_pct=0.03, sl_pct=0.02):
        self.df = df.copy()
        self.window = window
        self.volume_factor = volume_factor
        self.forward_window = forward_window
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        
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
        
        # Pemisahan kronologis 80/20
        split_idx = int(len(X) * 0.8)
        split_idx = min(split_idx, len(X) - 2)
        split_idx = max(split_idx, 4)
        
        if split_idx >= len(X):
            return None, None, None
            
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
        self.model.fit(X_train, y_train)
        
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
        
        importances = self.model.feature_importances_
        feat_importance = pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)
        
        test_results = pd.DataFrame(index=X_test.index)
        test_results['Actual'] = y_test
        test_results['Predicted'] = y_pred
        test_results['Probability'] = y_pred_prob
        
        return metrics, feat_importance, test_results


_yf_cache = {}

def load_yfinance_data(ticker, start, end):
    cache_key = (ticker, start, end)
    if cache_key in _yf_cache:
        return _yf_cache[cache_key].copy()
        
    for attempt in range(2):
        try:
            raw_df = yf.download(ticker, start=start, end=end)
            if len(raw_df) > 10:
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)
                _yf_cache[cache_key] = raw_df.copy()
                return raw_df
        except Exception as e:
            print(f"yfinance error untuk {ticker}: {e}")
        time.sleep(1)
        
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
    acc_val = f"{metrics['accuracy']*100:.2f}%" if metrics else 'N/A'
    prec_val = f"{metrics['precision']*100:.2f}%" if metrics else 'N/A'
    rec_val = f"{metrics['recall']*100:.2f}%" if metrics else 'N/A'
    f1_val = f"{metrics['f1']*100:.2f}%" if metrics else 'N/A'
    
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
            <tr><td>Akurasi (Accuracy)</td><td>{acc_val}</td></tr>
            <tr><td>Presisi (Precision)</td><td>{prec_val}</td></tr>
            <tr><td>Sensitivitas (Recall)</td><td>{rec_val}</td></tr>
            <tr><td>F1-Score</td><td>{f1_val}</td></tr>
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


def load_draft():
    try:
        with open("Draft_Bab1_3_UTS.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "File `Draft_Bab1_3_UTS.md` tidak ditemukan."


def parse_contents(contents, filename):
    if contents is None:
        return None
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df.columns = [col.strip().capitalize() for col in df.columns]
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif df.index.name != 'Date':
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                return None
        return df
    except Exception as e:
        print(f"Error parsing uploaded file: {e}")
        return None


# INITIALIZE DASH APP
app = dash.Dash(__name__, title="Deteksi Pola Breakout Saham")

app.layout = html.Div([
    dcc.Store(id='store-processed-data'),
    dcc.Download(id='download-csv'),
    dcc.Download(id='download-excel'),
    dcc.Download(id='download-pdf'),
    
    html.Div([
        html.Div([
            html.Div("PENGATURAN", className="sidebar-title", style={"marginTop":"0"}),
            
            html.Div([
                html.Label("Pilih Saham", className="control-label"),
                dcc.Dropdown(
                    id="stock-dropdown",
                    options=[
                        {"label": "BBCA - Bank Central Asia Tbk", "value": "BBCA.JK"},
                        {"label": "BBRI - Bank Rakyat Indonesia Tbk", "value": "BBRI.JK"},
                        {"label": "BMRI - Bank Mandiri Tbk", "value": "BMRI.JK"},
                        {"label": "BBNI - Bank Negara Indonesia Tbk", "value": "BBNI.JK"},
                        {"label": "TLKM - Telkom Indonesia Tbk", "value": "TLKM.JK"},
                        {"label": "ASII - Astra International Tbk", "value": "ASII.JK"},
                        {"label": "UNVR - Unilever Indonesia Tbk", "value": "UNVR.JK"},
                        {"label": "GOTO - GoTo Gojek Tokopedia Tbk", "value": "GOTO.JK"},
                        {"label": "ADRO - Adaro Energy Indonesia Tbk", "value": "ADRO.JK"},
                        {"label": "PGAS - Perusahaan Gas Negara Tbk", "value": "PGAS.JK"},
                        {"label": "Kustom Ticker (Ketik Manual)", "value": "CUSTOM"}
                    ],
                    value="BBCA.JK",
                    clearable=False,
                    className="dash-dropdown"
                )
            ], id="stock-selector-container", className="control-group"),
            
            html.Div([
                html.Label("Ticker Saham", className="control-label"),
                dcc.Input(id="custom-ticker-input", type="text", value="AAPL",
                          style={"width":"100%","padding":"4px 6px","borderRadius":"3px",
                                 "border":"1px solid #d0d0d0","fontSize":"12px"})
            ], id="custom-ticker-manual-container", className="control-group"),
            
            html.Div([
                html.Label("Unggah File CSV", className="control-label"),
                dcc.Upload(id="upload-csv", children=html.Div(["Klik untuk Unggah"]),
                           className="upload-container", multiple=False),
                html.Div(id="upload-filename-status",
                         style={"fontSize":"10px","color":"#0066cc","marginTop":"2px"})
            ], id="upload-csv-container", className="control-group"),
            
            dcc.Dropdown(id="source-type",
                options=[
                    {"label": "Yahoo Finance (Unduh)", "value": "Yahoo Finance (Unduh)"},
                    {"label": "Unggah Berkas CSV", "value": "Unggah Berkas CSV"},
                    {"label": "Simulasi Data Buatan", "value": "Simulasi Data Buatan"}
                ],
                value="Yahoo Finance (Unduh)", clearable=False,
                style={"display":"none"}),
            
            html.Div([
                html.Label("Tanggal Mulai", className="control-label"),
                dcc.Input(id="date-start", type="text", value="2022-01-01",
                          style={"width":"100%","padding":"3px 6px","borderRadius":"3px",
                                 "border":"1px solid #d0d0d0","fontSize":"11px","fontFamily":"inherit"})
            ], className="control-group"),
            
            html.Div([
                html.Label("Tanggal Akhir", className="control-label"),
                dcc.Input(id="date-end", type="text", value="2024-06-01",
                          style={"width":"100%","padding":"3px 6px","borderRadius":"3px",
                                 "border":"1px solid #d0d0d0","fontSize":"11px","fontFamily":"inherit"})
            ], className="control-group"),
            
            html.Div([
                dcc.Slider(id="window-slider", min=5, max=40, step=1, value=20, marks={}),
                dcc.Slider(id="vol-factor-slider", min=1.0, max=3.0, step=0.1, value=1.5, marks={}),
                dcc.Slider(id="forward-window-slider", min=3, max=10, step=1, value=5, marks={}),
                dcc.Slider(id="tp-pct-slider", min=1.0, max=10.0, step=0.5, value=3.0, marks={}),
                dcc.Slider(id="max-depth-slider", min=3, max=10, step=1, value=5, marks={}),
                dcc.DatePickerRange(id="date-picker", start_date="2022-01-01", end_date="2024-06-01"),
            ], style={"display":"none"}),
            
            html.Button("Proses", id="btn-proses", className="btn-proses", n_clicks=0),
            
            html.Div("INFORMASI SAHAM", className="sidebar-title"),
            html.Div(id="sidebar-stock-info"),
            
            html.Div("HASIL DETEKSI", className="sidebar-title"),
            html.Div(id="sidebar-detection-results"),
            
            html.Div("KETERANGAN", className="sidebar-title"),
            html.Div([
                html.Div([html.Span("\u2014", style={"color":"#1a1a1a","fontWeight":"bold","marginRight":"6px"}), html.Span("Harga Close")], className="legend-item"),
                html.Div([html.Span("\u2014", style={"color":"#f59e0b","fontWeight":"bold","marginRight":"6px"}), html.Span("Resistensi")], className="legend-item"),
                html.Div([html.Span("\u25b2", style={"color":"#059669","fontWeight":"bold","marginRight":"6px"}), html.Span("True Breakout (Beli)")], className="legend-item"),
                html.Div([html.Span("\u25bc", style={"color":"#cc0000","fontWeight":"bold","marginRight":"6px"}), html.Span("False Breakout (Jual)")], className="legend-item"),
            ]),
            
        ], className="sidebar"),
        
        html.Div([
            html.Div([
                html.Div(id="dashboard-content")
            ], className="dashboard-wrapper"),
            
            html.Div([
                html.Div(id="footer-status-text", className="footer-status"),
                html.Div([
                    html.Button("Simpan Hasil", id="btn-download-csv", className="btn-simpan", disabled=True),
                    html.Button("Keluar", id="btn-keluar", className="btn-keluar"),
                    html.Button("", id="btn-download-excel", style={"display":"none"}),
                    html.Button("", id="btn-download-pdf", style={"display":"none"}),
                ], className="footer-actions"),
            ], id="footer-banner-container", className="footer-bar"),
            
        ], className="main-content")
    ], className="app-container")
])



# CALLBACKS
@app.callback(
    [Output('stock-selector-container', 'style'),
     Output('custom-ticker-manual-container', 'style'),
     Output('upload-csv-container', 'style')],
    [Input('source-type', 'value'),
     Input('stock-dropdown', 'value')]
)
def toggle_source_inputs(source_type, stock_dropdown):
    stock_style = {'display': 'none'}
    custom_style = {'display': 'none'}
    upload_style = {'display': 'none'}
    
    if source_type == 'Yahoo Finance (Unduh)':
        stock_style = {'display': 'block'}
        if stock_dropdown == 'CUSTOM':
            custom_style = {'display': 'block'}
    elif source_type == 'Unggah Berkas CSV':
        upload_style = {'display': 'block'}
        
    return stock_style, custom_style, upload_style


@app.callback(
    Output('upload-filename-status', 'children'),
    [Input('upload-csv', 'filename')]
)
def update_upload_filename(filename):
    if filename:
        return f"File terunggah: {filename}"
    return ""


@app.callback(
    Output('store-processed-data', 'data'),
    [Input('btn-proses', 'n_clicks')],
    [State('source-type', 'value'),
     State('stock-dropdown', 'value'),
     State('custom-ticker-input', 'value'),
     State('upload-csv', 'contents'),
     State('upload-csv', 'filename'),
     State('date-start', 'value'),
     State('date-end', 'value'),
     State('window-slider', 'value'),
     State('vol-factor-slider', 'value'),
     State('forward-window-slider', 'value'),
     State('tp-pct-slider', 'value')]
)
def run_model(n_clicks, source_type, stock_dropdown, custom_ticker, upload_contents, upload_filename,
              start_date_str, end_date_str, window, vol_factor, forward_window, tp_pct):
    sl_pct = 2.0
    try:
        start_date = datetime.datetime.strptime(start_date_str.split('T')[0], '%Y-%m-%d').date()
        end_date = datetime.datetime.strptime(end_date_str.split('T')[0], '%Y-%m-%d').date()
    except Exception:
        start_date = datetime.date(2022, 1, 1)
        end_date = datetime.date(2024, 6, 1)
        
    ticker_code = "SIMULATION"
    stock_name = "Simulasi Saham Buatan"
    stock_sector = "Teknologi"
    df = None
    error_msg = None
    
    if source_type == "Yahoo Finance (Unduh)":
        ticker_code = custom_ticker if stock_dropdown == "CUSTOM" else stock_dropdown
        if stock_dropdown == "CUSTOM":
            stock_name = ticker_code
            stock_sector = "Kustom"
        else:
            STOCK_NAMES = {
                "BBCA.JK": ("Bank Central Asia Tbk", "Keuangan"),
                "BBRI.JK": ("Bank Rakyat Indonesia Tbk", "Keuangan"),
                "BMRI.JK": ("Bank Mandiri Tbk", "Keuangan"),
                "BBNI.JK": ("Bank Negara Indonesia Tbk", "Keuangan"),
                "TLKM.JK": ("Telkom Indonesia Tbk", "Komunikasi"),
                "ASII.JK": ("Astra International Tbk", "Keuangan/Konglomerat"),
                "UNVR.JK": ("Unilever Indonesia Tbk", "Barang Konsumsi"),
                "GOTO.JK": ("GoTo Gojek Tokopedia Tbk", "Teknologi"),
                "ADRO.JK": ("Adaro Energy Indonesia Tbk", "Energi"),
                "PGAS.JK": ("Perusahaan Gas Negara Tbk", "Energi")
            }
            if ticker_code in STOCK_NAMES:
                stock_name, stock_sector = STOCK_NAMES[ticker_code]
            else:
                stock_name = ticker_code
                stock_sector = "Kustom"
                
        if HAS_YFINANCE:
            df = load_yfinance_data(ticker_code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if df is None or len(df) < 10:
                error_msg = f"Gagal mengunduh data untuk ticker {ticker_code}. Silakan coba ticker lain atau gunakan sumber data simulasi."
        else:
            error_msg = "Library yfinance tidak terinstall di server."
            
    elif source_type == "Unggah Berkas CSV":
        stock_name = upload_filename if upload_filename else "Berkas Unggahan CSV"
        stock_sector = "Statistik / Kustom"
        ticker_code = "CSV_UPLOAD"
        if upload_contents is not None:
            df = parse_contents(upload_contents, stock_name)
            if df is None:
                error_msg = "Format file CSV tidak valid. Harus mengandung kolom: Date/index, Open, High, Low, Close, Volume."
        else:
            error_msg = "Silakan unggah file CSV saham terlebih dahulu."
            
    else: # Simulasi Data Buatan
        days = (end_date - start_date).days
        days = max(100, min(1000, days))
        df = generate_synthetic_data(days=days)
        
    if error_msg is not None:
        return {
            "error_msg": error_msg,
            "df_json": None,
            "candidate_df_json": None,
            "test_results_json": None,
            "metrics": None,
            "feat_imp": None,
            "stock_info": None,
            "true_b": 0,
            "false_b": 0
        }
        
    if df.index.name != 'Date':
        df.index.name = 'Date'
        
    predictor = StockBreakoutPredictor(
        df=df,
        window=window,
        volume_factor=vol_factor,
        forward_window=forward_window,
        tp_pct=tp_pct / 100.0,
        sl_pct=sl_pct / 100.0
    )
    predictor.calculate_technical_indicators()
    predictor.detect_and_label_candidates()
    metrics, feat_imp, test_results = predictor.train_ml_model()
    
    latest_close = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else latest_close
    change = latest_close - prev_close
    pct_change = (change / prev_close) * 100
    latest_date_str = df.index[-1].strftime('%d %B %Y')
    
    true_b = int(len(predictor.candidate_df[predictor.candidate_df['True_Breakout'] == 1]))
    false_b = int(len(predictor.candidate_df[predictor.candidate_df['True_Breakout'] == 0]))
    
    metrics_serialized = None
    if metrics is not None:
        metrics_serialized = {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "confusion_matrix": metrics["confusion_matrix"].tolist()
        }
        
    feat_imp_dict = {}
    if feat_imp is not None:
        feat_imp_dict = feat_imp.to_dict()
        
    stock_info = {
        "ticker_code": ticker_code,
        "stock_name": stock_name,
        "stock_sector": stock_sector,
        "latest_close": latest_close,
        "prev_close": prev_close,
        "change": change,
        "pct_change": pct_change,
        "latest_date_str": latest_date_str,
        "total_rows": len(df)
    }
    
    return {
        "error_msg": None,
        "df_json": predictor.df.reset_index().to_json(date_format='iso'),
        "candidate_df_json": predictor.candidate_df.reset_index().to_json(date_format='iso'),
        "test_results_json": test_results.reset_index().to_json(date_format='iso') if test_results is not None else None,
        "metrics": metrics_serialized,
        "feat_imp": feat_imp_dict,
        "stock_info": stock_info,
        "true_b": true_b,
        "false_b": false_b
    }


@app.callback(
    [Output('sidebar-stock-info', 'children'),
     Output('sidebar-detection-results', 'children'),
     Output('footer-status-text', 'children'),
     Output('footer-banner-container', 'style'),
     Output('btn-download-csv', 'disabled'),
     Output('btn-download-excel', 'disabled'),
     Output('btn-download-pdf', 'disabled')],
    [Input('store-processed-data', 'data')]
)
def update_sidebar_and_footer(data):
    if data is None or data.get('error_msg') is not None:
        err_msg = data.get('error_msg') if data else "Silakan lengkapi pengaturan dan klik 'Proses'."
        return (
            "Tidak ada data saham.",
            "Tidak ada hasil deteksi.",
            f"⚠️ {err_msg}",
            {"display": "flex", "backgroundColor": "rgba(239, 68, 68, 0.08)", "borderColor": "rgba(239, 68, 68, 0.2)", "color": "var(--accent-red)"},
            True, True, True
        )
        
    stock_info = data['stock_info']
    true_b = data['true_b']
    false_b = data['false_b']
    metrics = data['metrics']
    
    stock_info_html = html.Div([
        html.Div([html.Span("Kode Saham", className="info-item-label"), html.Span(f": {stock_info['ticker_code']}", className="info-item-value")], className="info-item"),
        html.Div([html.Span("Nama", className="info-item-label"), html.Span(f": {stock_info['stock_name']}", className="info-item-value")], className="info-item"),
        html.Div([html.Span("Sektor", className="info-item-label"), html.Span(f": {stock_info['stock_sector']}", className="info-item-value")], className="info-item"),
        html.Div([html.Span("Harga Terakhir", className="info-item-label"), html.Span(f": {stock_info['latest_close']:,.0f}", className="info-item-value")], className="info-item"),
        html.Div([html.Span("Perubahan", className="info-item-label"),
            html.Span(
                f": {'+' if stock_info['change'] >= 0 else ''}{stock_info['change']:,.0f} ({'+' if stock_info['change'] >= 0 else ''}{stock_info['pct_change']:.2f}%)",
                style={"color": "#007a3d" if stock_info['change'] >= 0 else "#cc0000", "fontWeight": "600", "fontSize":"12px"}
            )], className="info-item"),
        html.Div([html.Span("Tanggal", className="info-item-label"), html.Span(f": {stock_info['latest_date_str']}", className="info-item-value")], className="info-item"),
    ])
    
    acc_text = f"{metrics['accuracy']*100:.2f}%" if metrics is not None else "N/A"
    detection_html = html.Div([
        html.Div([html.Span("● ", style={"color":"#059669"}), html.Span(f"True Breakout", style={"color":"#333"}), html.Span(f"  : {true_b} kali", style={"color":"#059669","fontWeight":"600"})], style={"fontSize":"12px","lineHeight":"1.8"}),
        html.Div([html.Span("● ", style={"color":"#cc0000"}), html.Span(f"False Breakout", style={"color":"#333"}), html.Span(f" : {false_b} kali", style={"color":"#cc0000","fontWeight":"600"})], style={"fontSize":"12px","lineHeight":"1.8"}),
        html.Div([html.Span("● ", style={"color":"#555"}), html.Span(f"Akurasi Sinyal", style={"color":"#333"}), html.Span(f" : {acc_text}", style={"color":"#0066cc","fontWeight":"600"})], style={"fontSize":"12px","lineHeight":"1.8"}),
    ])
    
    status_text = [
        html.Span("✓ ", style={"color":"#059669","fontWeight":"bold"}),
        html.Span(f"Proses selesai! Total data yang digunakan: {stock_info['total_rows']} baris")
    ]
    has_signals = (true_b + false_b) > 0
    btn_disabled = not has_signals
    
    return (
        stock_info_html,
        detection_html,
        status_text,
        {"display": "flex"},
        btn_disabled, btn_disabled, btn_disabled
    )


@app.callback(
    Output('dashboard-content', 'children'),
    [Input('store-processed-data', 'data')]
)
def render_dashboard(data):
    if data is None:
        return html.Div("Silakan lengkapi pengaturan dan klik tombol 'Proses'.", style={"padding":"40px", "textAlign":"center", "color":"var(--text-secondary)"})
        
    if data.get('error_msg') is not None:
        return html.Div([
            html.H4("Terjadi Kesalahan", style={"color":"var(--accent-red)", "marginTop":"0"}),
            html.P(data['error_msg'])
        ], className="card", style={"borderColor":"var(--accent-red)", "color":"var(--text-secondary)"})
        
    df = pd.read_json(io.StringIO(data['df_json']))
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
    candidate_df = pd.read_json(io.StringIO(data['candidate_df_json']))
    if len(candidate_df) > 0 and 'Date' in candidate_df.columns:
        candidate_df['Date'] = pd.to_datetime(candidate_df['Date'])
        candidate_df.set_index('Date', inplace=True)
        
    true_b = data['true_b']
    false_b = data['false_b']
    metrics = data['metrics']
    stock_info = data['stock_info']
    
    # Plot
    ticker_display = stock_info['ticker_code'].replace('.JK','') if '.JK' in stock_info['ticker_code'] else stock_info['ticker_code']
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'],
                             name='Harga Close',
                             line=dict(color='#1a1a1a', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Resistance'],
                             name='Resistensi',
                             line=dict(color='#e07b00', width=1.5)))
    
    if len(candidate_df) > 0:
        c_true = candidate_df[candidate_df['True_Breakout'] == 1]
        c_false = candidate_df[candidate_df['True_Breakout'] == 0]
        
        fig.add_trace(go.Scatter(x=c_true.index, y=c_true['Close'], mode='markers+text',
                                 marker=dict(symbol='triangle-up', color='#059669', size=12),
                                 text=[f"True Breakout<br>{dt.strftime('%d %b %Y')}" for dt in c_true.index],
                                 textposition="top center",
                                 textfont=dict(color='#059669', size=9),
                                 name='True Breakout (Beli)'))
        fig.add_trace(go.Scatter(x=c_false.index, y=c_false['Close'], mode='markers+text',
                                 marker=dict(symbol='triangle-down', color='#cc0000', size=12),
                                 text=[f"False Breakout<br>{dt.strftime('%d %b %Y')}" for dt in c_false.index],
                                 textposition="bottom center",
                                 textfont=dict(color='#cc0000', size=9),
                                 name='False Breakout (Jual)'))
    
    fig.update_layout(
        title=dict(text=f"Grafik Deteksi Breakout - {ticker_display}",
                   font=dict(size=16, color='#1a1a1a', family="Inter"), x=0.5),
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font_color='#1a1a1a',
        font=dict(size=12, family="Inter"),
        height=420,
        hovermode='x unified',
        margin=dict(t=45, b=10, l=50, r=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='#f5f5f5',
                activecolor='#d0d0d0',
                font=dict(color='#1a1a1a', size=11)
            ),
            gridcolor='#eeeeee',
            linecolor='#cccccc',
            type='date',
            title=dict(text="")
        ),
        yaxis=dict(gridcolor='#eeeeee', linecolor='#cccccc', title="Harga")
    )
        
    table_rows = []
    if len(candidate_df) > 0:
        for idx, (dt, row) in enumerate(candidate_df.iterrows()):
            sig_text = 'True Breakout' if row['True_Breakout'] == 1.0 else 'False Breakout'
            sig_color = '#059669' if row['True_Breakout'] == 1.0 else '#cc0000'
            desc = f"Close {row['Close']:,.2f} > Resistance {row['Resistance']:,.2f} & Vol {row['Vol_Ratio']:.2f}x"
            table_rows.append(
                html.Tr([
                    html.Td(str(idx + 1)),
                    html.Td(dt.strftime('%d %b %Y')),
                    html.Td(sig_text, style={"color": sig_color, "fontWeight": "600"}),
                    html.Td(f"{row['Close']:,.0f}"),
                    html.Td(desc)
                ])
            )
    else:
        table_rows.append(html.Tr([html.Td("Tidak ada sinyal terdeteksi.", colSpan=5, style={"textAlign":"center"})]))
        
    signal_table = html.Table([
        html.Thead(
            html.Tr([
                html.Th("No"),
                html.Th("Tanggal"),
                html.Th("Sinyal"),
                html.Th("Harga"),
                html.Th("Keterangan")
            ])
        ),
        html.Tbody(table_rows)
    ], className="custom-table")
    
    test_results = pd.read_json(io.StringIO(data['test_results_json'])) if data['test_results_json'] else None
    if len(test_results) > 0 and 'Date' in test_results.columns:
        test_results['Date'] = pd.to_datetime(test_results['Date'])
        test_results.set_index('Date', inplace=True)
        
    if metrics is not None and test_results is not None and len(test_results) > 0:
        tot_sig = len(test_results)
        correct_sig = int((test_results['Actual'] == test_results['Predicted']).sum())
        wrong_sig = tot_sig - correct_sig
        acc_val = correct_sig / tot_sig if tot_sig > 0 else 0.0
    else:
        tot_sig = len(candidate_df)
        correct_sig = int(true_b)
        wrong_sig = int(false_b)
        acc_val = (correct_sig / tot_sig) if tot_sig > 0 else 0.0
        
    eval_table_rows = [
        html.Tr([html.Td("Total Sinyal"), html.Td(str(tot_sig))]),
        html.Tr([html.Td("Sinyal Benar"), html.Td(str(correct_sig))]),
        html.Tr([html.Td("Sinyal Salah"), html.Td(str(wrong_sig))]),
        html.Tr([html.Td("Akurasi", style={"fontWeight":"bold"}), html.Td(f"{acc_val*100:.2f}%", style={"color":"#059669", "fontWeight":"bold"})])
    ]
    
    eval_table = html.Table([
        html.Thead(
            html.Tr([html.Th("Metrik"), html.Th("Nilai", style={"textAlign":"right"})])
        ),
        html.Tbody(eval_table_rows)
    ], className="eval-table")
    
    fig_pie = go.Figure()
    if tot_sig > 0:
        fig_pie.add_trace(go.Pie(
            labels=['Sinyal Benar', 'Sinyal Salah'],
            values=[correct_sig, wrong_sig],
            hole=0,
            marker=dict(colors=['#059669', '#cc0000']),
            textinfo='percent'
        ))
    fig_pie.update_layout(
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font_color='#1a1a1a',
        font=dict(size=11, family="Inter"),
        height=150,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(yanchor="middle", y=0.5, xanchor="left", x=1.0)
    )
    
    return html.Div([
        # Baris Atas: Grafik
        html.Div([
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ], className="card", style={"marginBottom":"10px", "padding": "0"}),
        
        # Baris Bawah: Tabel & Evaluasi
        html.Div([
            # Daftar Sinyal
            html.Div([
                html.Div("DAFTAR SINYAL", className="sidebar-title", style={"marginTop":"0", "color":"#0066cc", "paddingLeft":"15px"}),
                html.Div(signal_table, className="data-table-container", style={"maxHeight":"260px", "overflowY":"auto", "border":"none", "borderTop":"1px solid #d0d0d0", "borderRadius":"0"})
            ], className="card", style={"flex":"0 0 68%", "padding":"10px 0 0 0", "overflow":"hidden"}),
            
            # Evaluasi Akurasi
            html.Div([
                html.Div("EVALUASI AKURASI", className="sidebar-title", style={"marginTop":"0", "color":"#0066cc", "paddingLeft":"15px"}),
                html.Div([
                    eval_table,
                    dcc.Graph(figure=fig_pie, config={'displayModeBar': False})
                ], style={"padding":"0 15px"})
            ], className="card", style={"flex":"0 0 32%", "padding":"10px 0 10px 0", "display":"flex", "flexDirection":"column", "gap":"10px"})
        ], style={"display":"flex", "gap":"10px", "alignItems":"stretch"})
    ])


@app.callback(
    Output('download-csv', 'data'),
    [Input('btn-download-csv', 'n_clicks')],
    [State('store-processed-data', 'data')],
    prevent_initial_call=True
)
def download_csv(n_clicks, data):
    if data is None:
        return None
    candidate_df = pd.read_json(io.StringIO(data['candidate_df_json']))
    if len(candidate_df) > 0:
        if 'Date' in candidate_df.columns:
            candidate_df['Date'] = pd.to_datetime(candidate_df['Date'])
            candidate_df.set_index('Date', inplace=True)
            
        export_df = pd.DataFrame()
        export_df['No'] = list(range(1, len(candidate_df) + 1))
        export_df['Tanggal'] = pd.to_datetime(candidate_df.index).strftime('%d %b %Y')
        export_df['Sinyal'] = candidate_df['True_Breakout'].map({1.0: 'True Breakout', 0.0: 'False Breakout'}).values
        export_df['Harga'] = [f"Rp {x:,.2f}" for x in candidate_df['Close'].values]
        export_df['Keterangan'] = [
            f"Close {row['Close']:,.2f} > Resistance {row['Resistance']:,.2f} & Vol {row['Vol_Ratio']:.2f}x"
            for _, row in candidate_df.iterrows()
        ]
        return dcc.send_data_frame(export_df.to_csv, f"sinyal_breakout_{data['stock_info']['ticker_code']}.csv", index=False)


@app.callback(
    Output('download-excel', 'data'),
    [Input('btn-download-excel', 'n_clicks')],
    [State('store-processed-data', 'data')],
    prevent_initial_call=True
)
def download_excel(n_clicks, data):
    if data is None:
        return None
    candidate_df = pd.read_json(io.StringIO(data['candidate_df_json']))
    if len(candidate_df) > 0:
        if 'Date' in candidate_df.columns:
            candidate_df['Date'] = pd.to_datetime(candidate_df['Date'])
            candidate_df.set_index('Date', inplace=True)
            
        export_df = pd.DataFrame()
        export_df['No'] = list(range(1, len(candidate_df) + 1))
        export_df['Tanggal'] = pd.to_datetime(candidate_df.index).strftime('%d %b %Y')
        export_df['Sinyal'] = candidate_df['True_Breakout'].map({1.0: 'True Breakout', 0.0: 'False Breakout'}).values
        export_df['Harga'] = [f"Rp {x:,.2f}" for x in candidate_df['Close'].values]
        export_df['Keterangan'] = [
            f"Close {row['Close']:,.2f} > Resistance {row['Resistance']:,.2f} & Vol {row['Vol_Ratio']:.2f}x"
            for _, row in candidate_df.iterrows()
        ]
        return dcc.send_data_frame(export_df.to_excel, f"sinyal_breakout_{data['stock_info']['ticker_code']}.xlsx", index=False, sheet_name='Sinyal Breakout')


@app.callback(
    Output('download-pdf', 'data'),
    [Input('btn-download-pdf', 'n_clicks')],
    [State('store-processed-data', 'data')],
    prevent_initial_call=True
)
def download_pdf(n_clicks, data):
    if data is None:
        return None
    
    stock_info = data['stock_info']
    candidate_df = pd.read_json(io.StringIO(data['candidate_df_json']))
    metrics = data['metrics']
    true_b = data['true_b']
    false_b = data['false_b']
    
    if 'Date' in candidate_df.columns:
        candidate_df['Date'] = pd.to_datetime(candidate_df['Date'])
        candidate_df.set_index('Date', inplace=True)
        
    show_df = pd.DataFrame()
    if len(candidate_df) > 0:
        show_df['No'] = list(range(1, len(candidate_df) + 1))
        show_df['Tanggal'] = pd.to_datetime(candidate_df.index).strftime('%d %b %Y')
        show_df['Harga'] = [f"Rp {x:,.2f}" for x in candidate_df['Close'].values]
        show_df['Keterangan'] = [
            f"Close {row['Close']:,.2f} > Resistance {row['Resistance']:,.2f} & Vol {row['Vol_Ratio']:.2f}x"
            for _, row in candidate_df.iterrows()
        ]
    
    html_report = generate_html_report(
        stock_info['stock_name'],
        stock_info['ticker_code'],
        metrics,
        show_df,
        true_b,
        false_b
    )
    
    return dict(content=html_report, filename=f"laporan_breakout_{stock_info['ticker_code']}.html")


@app.callback(
    [Output('store-processed-data', 'data', allow_duplicate=True),
     Output('source-type', 'value'),
     Output('stock-dropdown', 'value'),
     Output('custom-ticker-input', 'value'),
     Output('upload-csv', 'contents'),
     Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date')],
    [Input('btn-keluar', 'n_clicks')],
    prevent_initial_call=True
)
def reset_app(n_clicks):
    return None, "Yahoo Finance (Unduh)", "BBCA.JK", "AAPL", None, "2022-01-01", "2024-06-01"


# RUN APP
if __name__ == '__main__':
    # Use standard port 8050
    app.run(host='127.0.0.1', port=8050, debug=True)
