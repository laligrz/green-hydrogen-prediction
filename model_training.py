import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("--- بدء تشغيل نموذج الـ AdaBoost الهجين لإنتاج الهيدروجين ---")

# 1. تحميل قاعدة البيانات الفيزيائية المستندة للثرموديناميكا
try:
  df = pd.read_csv("realistic_hydrogen_data.csv")
  print("تم تحميل ملف البيانات الفيزيائية ('realistic_hydrogen_data.csv') بنجاح!")
except FileNotFoundError:
  print(
      "خطأ: لم يتم العثور على الملف. يرجى التأكد من تشغيل كود توليد البيانات"
      " أولاً."
  )
  exit()

# 2. تحديد مصفوفة المدخلات (X) والمخرج المستهدف (y)
# المدخلات تشمل: سرعة الرياح، كفاءة المحلل الكهربائي، وطاقة الرياح المحسوبة فيزيائياً
X = df[["Wind_Speed_m/s", "Electrolyzer_Efficiency_%", "Wind_Power_kW"]]
y = df["Hydrogen_Production_kg/day"]

# 3. تقسيم البيانات: 80% للتدريب و 20% للاختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"تم تقسيم البيانات بنجاح: {len(X_train)} عينة للتدريب، {len(X_test)} عينة للاختبار.")

# 4. إعداد وتكوين نموذج AdaBoost بالمعلمات المثلى (المستخرجة من تحسين BWOA)
optimized_adaboost = AdaBoostRegressor(
    n_estimators=120,  # عدد المراحل/الأشجار
    learning_rate=0.08,  # معدل التعلم الأمثل
    random_state=42,  # تثبيت النتائج العشوائية
)

# 5. تدريب النموذج الفعلي على بيانات التدريب الفيزيائية
print("جاري تدريب نموذج الـ AdaBoost...")
optimized_adaboost.fit(X_train, y_train)
print("تم الانتهاء من التدريب بنجاح!")

# 6. التنبؤ وتقييم الأداء على بيانات الاختبار
y_pred = optimized_adaboost.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 7. طباعة نتائج التقييم النهائي
print("\n" + "=" * 40)
print("       نتائج تقييم الأداء للنموذج الهجين       ")
print("=" * 40)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score (دقة التفسير): {r2:.4f}")
print("=" * 40)