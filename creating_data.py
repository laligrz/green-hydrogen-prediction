import numpy as np
import pandas as pd

# تحديد عدد العينات (مثلاً 1000 ساعة تشغيل)
n_samples = 1000

# تثبيت العشوائية لضمان تكرار النتائج بثبات
np.random.seed(42)

# 1. توليد مدخلات سرعة الرياح (متر/ثانية) ضمن نطاق واقعي لتوربينات الرياح
wind_speed = np.random.uniform(3.0, 25.0, n_samples)

# 2. توليد كفاءة المحلل الكهربائي (%) ضمن النطاق الصناعي المعتاد (60% إلى 82%)
electrolyzer_efficiency = np.random.uniform(0.60, 0.82, n_samples)

# 3. حساب طاقة الرياح بناءً على الفيزياء (تتناسب مع مكعب السرعة v^3)
air_density = 1.225  # كثافة الهواء عند مستوى سطح البحر (kg/m^3)
rotor_area = 5000     # مساحة مسح شفرات التوربين بالمتر المربع
# معادلة طاقة الرياح بالكيلوواط
wind_power_kw = (0.5 * air_density * rotor_area * (wind_speed ** 3)) / 1000

# 4. حساب إنتاج الهيدروجين ديناميكياً حرارياً
# القيمة الحرارية العليا للهيدروجين (HHV ~ 39.4 kWh/kg)
hhv_hydrogen = 39.4 
daily_hours = 24

# كمية الإنتاج = (الطاقة الكهربائية * الكفاءة * ساعات التشغيل) / القيمة الحرارية
hydrogen_production = (wind_power_kw * electrolyzer_efficiency * daily_hours) / hhv_hydrogen

# إضافة ضوضاء ضئيلة جداً (Gaussian Noise) لتمثيل أخطاء القياس الطبيعية في المحطات الحقيقية
noise = np.random.normal(0, 1.5, n_samples)
hydrogen_production = np.clip(hydrogen_production + noise, 0, None)  # ضمان عدم وجود قيم سالبة

# 5. تجميع البيانات في جدول منظم (DataFrame)
df_physics = pd.DataFrame({
    'Wind_Speed_m/s': np.round(wind_speed, 2),
    'Electrolyzer_Efficiency_%': np.round(electrolyzer_efficiency * 100, 2),
    'Wind_Power_kW': np.round(wind_power_kw, 2),
    'Hydrogen_Production_kg/day': np.round(hydrogen_production, 2)
})

# حفظ الملف بصيغة CSV لاستخدامه في نموذج الذكاء الاصطناعي لاحقاً
df_physics.to_csv('realistic_hydrogen_data.csv', index=False)
print("تم توليد وحفظ قاعدة البيانات الفيزيائية بنجاح باسم 'realistic_hydrogen_data.csv'!")