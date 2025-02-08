import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 1. بارگذاری و پیش‌پردازش داده‌ها
# ============================================================

# بارگذاری داده‌های CSV
data = pd.read_csv('vbemelat.csv')

# اصلاح نام ستون‌ها (حذف < و > و فضاهای اضافی)
data.columns = data.columns.str.replace('<', '').str.replace('>', '').str.strip()

# تبدیل ستون تاریخ به نوع datetime و مرتب‌سازی بر اساس تاریخ
data['DTYYYYMMDD'] = pd.to_datetime(data['DTYYYYMMDD'], format='%Y%m%d')
data = data.sort_values('DTYYYYMMDD')

# افزودن ویژگی‌های فصلی: روز هفته و ماه
data['day_of_week'] = data['DTYYYYMMDD'].dt.dayofweek    # عدد 0 تا 6
data['month'] = data['DTYYYYMMDD'].dt.month               # عدد 1 تا 12

# ============================================================
# 2. محاسبه اندیکاتورهای تکنیکال
# ============================================================

# محاسبه SMA و EMA
data['SMA_10'] = data['CLOSE'].rolling(window=10).mean()
data['EMA_10'] = data['CLOSE'].ewm(span=10, adjust=False).mean()

# تابع محاسبه RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

data['RSI'] = compute_rsi(data['CLOSE'], window=14)

# تابع محاسبه MACD و سیگنال آن
def compute_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['CLOSE'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['CLOSE'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

data['MACD'], data['MACD_signal'] = compute_macd(data)

# تابع محاسبه Bollinger Bands
def compute_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df['CLOSE'].rolling(window=window).mean()
    rolling_std = df['CLOSE'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

data['Bollinger_Upper'], data['Bollinger_Lower'] = compute_bollinger_bands(data)

# تابع محاسبه ATR
def compute_atr(df, window=14):
    high_low = df['HIGH'] - df['LOW']
    high_close = (df['HIGH'] - df['CLOSE'].shift()).abs()
    low_close = (df['LOW'] - df['CLOSE'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

data['ATR'] = compute_atr(data)

# حذف ردیف‌هایی که به دلیل محاسبات اندیکاتورها NaN شده‌اند
data.dropna(inplace=True)

# ============================================================
# 3. آماده‌سازی ویژگی‌ها و ساخت دیتاست سری زمانی
# ============================================================

# انتخاب ویژگی‌ها؛ در اینجا علاوه بر اندیکاتورها، ویژگی‌های فصلی (day_of_week و month) هم اضافه شده‌اند.
features = ['CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOL',
            'SMA_10', 'EMA_10', 'RSI', 'MACD', 'ATR',
            'Bollinger_Upper', 'Bollinger_Lower',
            'day_of_week', 'month']

dataset = data[features].values

# نرمال‌سازی داده‌ها؛ برای مدل‌های عمیق از RobustScaler و برای XGBoost از MinMaxScaler استفاده می‌کنیم.
scaler_dl = RobustScaler()
scaled_data_dl = scaler_dl.fit_transform(dataset)

scaler_xgb = MinMaxScaler(feature_range=(0, 1))
scaled_data_xgb = scaler_xgb.fit_transform(dataset)

# تعریف تابع ساخت دیتاست سری زمانی با پنجره ۹۰ روزه
def create_dataset(data_array, time_step=90):
    X, y = [], []
    for i in range(time_step, len(data_array)):
        X.append(data_array[i - time_step:i])
        y.append(data_array[i, 0])  # هدف: قیمت بسته شدن
    return np.array(X), np.array(y)

time_steps = 90
num_features = len(features)

# داده‌ها برای مدل Deep Learning
X_dl, y_dl = create_dataset(scaled_data_dl, time_step=time_steps)
X_dl = X_dl.reshape(X_dl.shape[0], time_steps, num_features)

# داده‌ها برای XGBoost: تبدیل هر پنجره به یک بردار فلت شده
X_xgb, y_xgb = create_dataset(scaled_data_xgb, time_step=time_steps)
X_xgb = X_xgb.reshape(X_xgb.shape[0], time_steps * num_features)

# تقسیم داده‌ها به آموزش و اعتبارسنجی (بدون shuffle به دلیل داده‌های سری زمانی)
X_dl_train, X_dl_val, y_dl_train, y_dl_val = train_test_split(X_dl, y_dl, test_size=0.1, shuffle=False)
X_xgb_train, X_xgb_val, y_xgb_train, y_xgb_val = train_test_split(X_xgb, y_xgb, test_size=0.1, shuffle=False)

# ============================================================
# 4. ساخت مدل Deep Learning (با بلوک‌های Transformer و Bidirectional GRU)
# ============================================================

# تعریف بلوک Transformer Encoder
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # نرمال‌سازی و MultiHead Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs  # اتصال باقی‌مانده (Residual Connection)
    
    # بخش Feed-Forward
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

# ورودی مدل
input_layer = Input(shape=(time_steps, num_features))
x = input_layer

# اعمال 4 بلوک Transformer Encoder
for _ in range(4):
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

# اضافه کردن لایه‌های Bidirectional GRU جهت استخراج وابستگی‌های زمانی
x = Bidirectional(GRU(128, return_sequences=True))(x)
x = Dropout(0.3)(x)
x = Bidirectional(GRU(64, return_sequences=False))(x)
x = Dropout(0.3)(x)

# لایه‌های Dense نهایی
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(1)(x)

# ساخت مدل Deep Learning
model_dl = Model(inputs=input_layer, outputs=output_layer)
model_dl.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
model_dl.summary()

# تعریف Callback ها برای مدل Deep Learning
callbacks_dl = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    ModelCheckpoint('best_model_dl.h5', monitor='val_loss', save_best_only=True)
]

# آموزش مدل Deep Learning
history_dl = model_dl.fit(X_dl_train, y_dl_train, epochs=150, batch_size=32,
                          validation_data=(X_dl_val, y_dl_val),
                          callbacks=callbacks_dl)

# ============================================================
# 5. آموزش مدل XGBoost (بدون استفاده از early_stopping_rounds)
# ============================================================

# تنظیم مدل XGBoost بدون early_stopping_rounds
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.01, max_depth=6)

# آموزش مدل XGBoost
model_xgb.fit(X_xgb_train, y_xgb_train, eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)

# ============================================================
# 6. پیش‌بینی ۳۰ روز آینده به صورت مرحله به مرحله (iterative forecasting)
# ============================================================

future_days = 30

# ---------------------------
# پیش‌بینی توسط مدل Deep Learning
# ---------------------------
future_preds_dl = []
last_window_dl = scaled_data_dl[-time_steps:].tolist()  # آخرین پنجره ۹۰ روزه داده‌های Deep Learning

for _ in range(future_days):
    inp_dl = np.array(last_window_dl[-time_steps:]).reshape(1, time_steps, num_features)
    pred_dl = model_dl.predict(inp_dl)
    future_preds_dl.append(pred_dl[0, 0])
    # ایجاد ورودی جدید: قیمت بسته شدن پیش‌بینی شده، سایر ویژگی‌ها صفر
    new_entry_dl = np.zeros((num_features,))  
    new_entry_dl[0] = pred_dl[0, 0]  # اضافه کردن قیمت بسته شدن پیش‌بینی شده به ویژگی‌ها
    last_window_dl.append(new_entry_dl)

# معکوس نرمال‌سازی پیش‌بینی‌های Deep Learning
future_preds_dl_full = np.hstack((np.array(future_preds_dl).reshape(-1, 1),
                                  np.zeros((future_days, num_features - 1))))
future_preds_dl_final = scaler_dl.inverse_transform(future_preds_dl_full)[:, 0]

# ---------------------------
# پیش‌بینی توسط مدل XGBoost
# ---------------------------
future_preds_xgb = []
last_window_xgb = scaled_data_xgb[-time_steps:]  # آخرین پنجره ۹۰ روزه برای XGBoost

for _ in range(future_days):
    inp_xgb = last_window_xgb.reshape(1, time_steps * num_features)
    pred_xgb = model_xgb.predict(inp_xgb)
    future_preds_xgb.append(pred_xgb[0])
    # ایجاد ورودی جدید: قیمت بسته شدن پیش‌بینی شده، سایر ویژگی‌ها صفر
    new_entry_xgb = np.zeros((num_features,))  
    new_entry_xgb[0] = pred_xgb[0]  # اضافه کردن قیمت بسته شدن پیش‌بینی شده به ویژگی‌ها
    last_window_xgb = np.vstack([last_window_xgb[1:], new_entry_xgb])

# معکوس نرمال‌سازی پیش‌بینی‌های XGBoost
future_preds_xgb_full = np.hstack((np.array(future_preds_xgb).reshape(-1, 1),
                                   np.zeros((future_days, num_features - 1))))
future_preds_xgb_final = scaler_xgb.inverse_transform(future_preds_xgb_full)[:, 0]

# ---------------------------
# Ensemble نهایی: میانگین دو مدل (می‌توانید وزن‌های دلخواه اختصاص دهید)
# ---------------------------
ensemble_preds = (future_preds_dl_final + future_preds_xgb_final) / 2

# ============================================================
# 7. نمایش نمودار نهایی
# ============================================================

plt.figure(figsize=(12, 6))
# نمایش قیمت‌های واقعی 100 روز آخر
plt.plot(data['DTYYYYMMDD'][-100:], data['CLOSE'].values[-100:], label='قیمت واقعی', color='blue')
# تولید تاریخ‌های آینده (بدون استفاده از پارامتر closed)
future_dates = pd.date_range(start=data['DTYYYYMMDD'].iloc[-1], periods=future_days + 1)[1:]
plt.plot(future_dates, ensemble_preds, label='پیش‌بینی ۳۰ روز آینده (Ensemble)', color='red', linestyle='dashed')

plt.title('پیش‌بینی قیمت ۳۰ روز آینده با مدل Ensemble فوق پیشرفته')
plt.xlabel('تاریخ')
plt.ylabel('قیمت سهام')
plt.legend()
plt.show()
