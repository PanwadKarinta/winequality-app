# app.py (ชื่อไฟล์ที่แนะนำสำหรับ Hugging Face)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
# --- 🔽 ส่วนที่แก้ไข: เปลี่ยนการ Import โมเดล 🔽 ---
from sklearn.tree import DecisionTreeClassifier
# --- 🔼 สิ้นสุดส่วนที่แก้ไข 🔼 ---
from sklearn.metrics import accuracy_score
import gradio as gr

# ----------------------------------------------------
# ส่วนที่ 1: สร้างและฝึกโมเดล (เปลี่ยนเป็น Decision Tree)
# ----------------------------------------------------
model_accuracy = 0.0
best_params = {}

try:
    df = pd.read_csv("winequality-red.csv")

    df['quality_label'] = df['quality'].apply(lambda v: 1 if v >= 7 else 0)

    X = df.drop(['quality', 'quality_label'], axis=1)
    y = df['quality_label']
    FEATURE_COLUMNS = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42, stratify=y
    )

    # --- 🔽 Hyperparameter Tuning: แก้ไขสำหรับ Decision Tree 🔽 ---
    # 1. อัปเดต param_grid ให้เหมาะกับ Decision Tree
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    # 2. เปลี่ยนโมเดลเป็น DecisionTreeClassifier
    dt = DecisionTreeClassifier(random_state=42)
    
    # 3. ใช้ dt เป็น estimator ใน GridSearchCV
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # ใช้โมเดลที่ดีที่สุดจากการค้นหา
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    # --- 🔼 สิ้นสุด Hyperparameter Tuning 🔼 ---

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    model_accuracy = accuracy

    print(f"✅ โมเดล (Decision Tree) อัปเกรดและฝึกเสร็จเรียบร้อย")
    print(f"📊 Best Parameters Found: {best_params}")
    print(f"📊 ความแม่นยำใหม่ของโมเดลบนข้อมูลทดสอบ: {model_accuracy * 100:.2f}%\n")

except FileNotFoundError:
    print("❌ ไม่พบไฟล์ 'winequality-red.csv'")
    # กำหนดค่าพื้นฐานหากไม่มีไฟล์ CSV (เพื่อให้ UI แสดงผลได้)
    FEATURE_COLUMNS = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'ph', 'sulphates', 'alcohol'
    ]
    model = None
    scaler = None

# ----------------------------------------------------
# ส่วนที่ 2: ฟังก์ชันทำนายผล (ไม่ต้องแก้ไข)
# ----------------------------------------------------
def predict_quality(*features):
    if model is None or scaler is None:
        return "เกิดข้อผิดพลาด: ยังไม่ได้ฝึกโมเดล (ไม่พบไฟล์ CSV)"
    try:
        # รับค่า Input ทั้ง 11 ตัว
        features_dict = {col: val for col, val in zip(FEATURE_COLUMNS, features)}
        df_new = pd.DataFrame([features_dict])

        # จัดเรียงคอลัมน์ให้ตรงกับตอนฝึกโมเดล (เผื่อไว้)
        df_new = df_new[FEATURE_COLUMNS]

        new_scaled = scaler.transform(df_new)
        probabilities = model.predict_proba(new_scaled)[0]
        prediction_index = np.argmax(probabilities)
        confidence = probabilities[prediction_index]

        result_text = "🍷 ไวน์คุณภาพดี" if prediction_index == 1 else "🍷 ไวน์คุณภาพไม่ดี"
        return f"{result_text} (ความมั่นใจ {confidence:.2%})"

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"

# ----------------------------------------------------
# ส่วนที่ 3: สร้างหน้าเว็บด้วย Gradio (ไม่ต้องแก้ไข)
# ----------------------------------------------------

feature_translations = {
    "fixed acidity": "ค่าความเป็นกรดคงที่", "volatile acidity": "ค่าความเป็นกรดระเหยง่าย",
    "citric acid": "ค่ากรดซิตริก", "residual sugar": "ค่าน้ำตาลคงเหลือ", "chlorides": "ค่าคลอไรด์",
    "free sulfur dioxide": "ค่าซัลเฟอร์ไดออกไซด์อิสระ", "total sulfur dioxide": "ค่าซัลเฟอร์ไดออกไซด์ทั้งหมด",
    "density": "ค่าความหนาแน่น", "ph": "ค่าพีเอช", "sulphates": "ค่าซัลเฟต", "alcohol": "ค่าแอลกอฮอล์"
}

with gr.Blocks(theme=gr.themes.Soft(primary_hue="rose", secondary_hue="rose")) as demo:
    gr.Markdown(
        """<div style="text-align: center;"><h1>Grandeur Wine Analyzer 🍷</h1><p>ใส่คุณลักษณะต่างๆ ของไวน์แดงเพื่อทำนายคุณภาพ</p></div>"""
    )
    
    gr.Markdown(
        f"""<div style="text-align: center; padding: 10px; border: 1px solid #E5E7EB; border-radius: 0.5rem; background-color: #F9FAFB;"><h3 style="margin:0;">📊 ความแม่นยำโดยรวมของโมเดล (Overall Accuracy): <strong>{model_accuracy:.2%}</strong></h3></div>"""
    )
    
    inputs_list = [gr.Number(label=f"{col.replace('_', ' ').title()} ({feature_translations.get(col, '')})", value=0) for col in FEATURE_COLUMNS]

    gr.Examples(
        examples=[
            [7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0], # Quality 5 -> 0
            [7.8,0.58,0.02,2.0,0.073,9.0,18.0,0.9968,3.36,0.57,9.5], # Quality 7 -> 1
            [6.7,0.58,0.08,1.8,0.097,15.0,65.0,0.9959,3.28,0.54,9.2] # Quality 5 -> 0
        ],
        inputs=inputs_list,
        label="กดเพื่อลองตัวอย่าง (Click to Try an Example)"
    )

    with gr.Row():
        predict_button = gr.Button("วิเคราะห์คุณภาพไวน์ (Predict)", variant="primary")
        clear_button = gr.ClearButton(value="ล้างข้อมูล (Clear)")

    output_text = gr.Textbox(label="ผลการทำนาย (Predicted Quality)", interactive=False, text_align="center", show_copy_button=True)

    with gr.Accordion("คำอธิบายคุณลักษณะ (Feature Descriptions)", open=False):
        gr.Markdown(
            """
            - **Fixed Acidity:** ค่ากรดส่วนใหญ่ในไวน์ที่ไม่ระเหยง่าย
            - **Volatile Acidity:** ปริมาณกรดอะซิติกในไวน์ ซึ่งในระดับสูงจะทำให้มีกลิ่นเหมือนน้ำส้มสายชู
            - **Citric Acid:** พบในปริมาณเล็กน้อย สามารถเพิ่มความสดชื่นให้กับไวน์
            - **Residual Sugar:** ปริมาณน้ำตาลที่เหลืออยู่หลังจากการหมักหยุดลง
            - **Chlorides:** ปริมาณเกลือในไวน์
            - **Free/Total Sulfur Dioxide:** สารกันบูดที่ช่วยป้องกันไวน์จากปฏิกิริยาออกซิเดชันและแบคทีเรีย
            - **Density:** ความหนาแน่นของไวน์ ซึ่งใกล้เคียงกับความหนาแน่นของน้ำ
            - **pH:** อธิบายความเป็นกรด-ด่างของไวน์ (ค่าต่ำ = กรดสูง)
            - **Sulphates:** สารเติมแต่งที่สามารถช่วยป้องกันการเจริญเติบโตของจุลินทรีย์
            - **Alcohol:** เปอร์เซ็นต์แอลกอฮอล์ในไวน์
            """
        )

    gr.Markdown(
        """<hr><p style='text-align: center; font-size: 0.8em; color: grey;'>โมเดลนี้ฝึกจากชุดข้อมูล Red Wine Quality บน Kaggle | สร้างสรรค์ด้วย Gradio</p>"""
    )
    
    predict_button.click(fn=predict_quality, inputs=inputs_list, outputs=output_text)
    clear_button.add(inputs_list + [output_text])

# ----------------------------------------------------
# ส่วนที่ 4: รันโปรแกรม
# ----------------------------------------------------
if __name__ == "__main__":
    demo.launch()