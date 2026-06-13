with open(r'g:\Tugas\Tugas Sistem Cerdas\pemrograman Python\app.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = '# INITIALIZE DASH APP'
end_marker = '\n\n\n# CALLBACKS'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

new_layout = r'''# INITIALIZE DASH APP
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
                dcc.DatePickerSingle(id="date-start", date="2022-01-01",
                                     display_format="YYYY-MM-DD",
                                     style={"width":"100%"})
            ], className="control-group"),
            
            html.Div([
                html.Label("Tanggal Akhir", className="control-label"),
                dcc.DatePickerSingle(id="date-end", date="2024-06-01",
                                     display_format="YYYY-MM-DD",
                                     style={"width":"100%"})
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
'''

new_content = content[:start_idx] + new_layout + end_marker + content[end_idx + len(end_marker):]

with open(r'g:\Tugas\Tugas Sistem Cerdas\pemrograman Python\app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Layout updated successfully!')
