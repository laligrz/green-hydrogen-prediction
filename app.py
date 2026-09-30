import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# إعدادات الصفحة
st.set_page_config(
    page_title="Hydrogen Production Predictor", page_icon="🔋", layout="centered"
)

st.title(" نموذج تنبؤ إنتاج الهيدروجين الأخضر")


# دالة تحميل البيانات وتدريب النموذج مع التخزين المؤقت لتسريع الأداء
@st.cache_data
def load_and_train_model():
  try:
    df = pd.read_csv("realistic_hydrogen_data.csv")
  except FileNotFoundError:
    return None, None, None

  X = df[["Wind_Speed_m/s", "Electrolyzer_Efficiency_%", "Wind_Power_kW"]]
  y = df["Hydrogen_Production_kg/day"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # بناء نموذج الـ AdaBoost بالمعلمات المثلى
  model = AdaBoostRegressor(
      n_estimators=120, learning_rate=0.08, random_state=42
  )
  model.fit(X_train, y_train)

  # حساب الدقة
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)

  return model, r2, df


model, r2_score_val, df = load_and_train_model()

if model is None:
  st.error(
      "❌ تنبيه: لم يتم العثور على ملف البيانات 'realistic_hydrogen_data.csv'."
      " يرجى تشغيل كود توليد البيانات أولاً!"
  )
else:
  st.success(
      f"✅ تم تحميل البيانات وتدريب النموذج بنجاح! | دقة النموذج ($R^2$):"
      f" {r2_score_val:.4f}"
  )

  # لوحة التحكم الجانبية لإدخال المتغيرات
  st.sidebar.header(" مدخلات المحطة المتجددة")

  min_ws = float(df["Wind_Speed_m/s"].min())
  max_ws = float(df["Wind_Speed_m/s"].max())
  input_wind_speed = st.sidebar.slider(
      "سرعة الرياح (متر/ثانية)", min_ws, max_ws, 10.0
  )

  min_eff = float(df["Electrolyzer_Efficiency_%"].min())
  max_eff = float(df["Electrolyzer_Efficiency_%"].max())
  input_efficiency = st.sidebar.slider(
      "كفاءة المحلل الكهربائي (%)", min_eff, max_eff, 70.0
  )

  # حساب طاقة الرياح فيزيائياً لحظياً بناءً على مكعب السرعة
  air_density = 1.225
  rotor_area = 5000
  calculated_wind_power = (
      0.5 * air_density * rotor_area * (input_wind_speed**3)
  ) / 1000

  st.sidebar.info(
      f" طاقة الرياح المحسوبة فيزيائياً: **{calculated_wind_power:.2f} kW**"
  )

  # زر التنبؤ
  if st.button(" احسب إنتاج الهيدروجين المتوقع"):
    user_input = np.array(
        [[input_wind_speed, input_efficiency, calculated_wind_power]]
    )
    predicted_hydrogen = model.predict(user_input)[0]

    st.metric(
        label="إجمالي إنتاج الهيدروجين المتوقع",
        value=f"{predicted_hydrogen:.2f} kg/day",
    )
