from flask import Flask, render_template, request, send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pickle
import os

app = Flask(__name__)


model_diabetes = pickle.load(open("diabetes_model.pkl", "rb"))
model_heart = pickle.load(open("heart_model.pkl", "rb"))
model_obesity = pickle.load(open("obesity_model.pkl", "rb"))


def generate_pdf_report(diabetes, heart, obesity, suggestions):
    file_path = "report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Health Risk Assessment Report", styles['Title']))
    content.append(Spacer(1, 12))
    
    content.append(Paragraph(f"<b>Diabetes Risk:</b> {round(diabetes*100, 2)}%", styles['Normal']))
    content.append(Paragraph(f"<b>Heart Risk:</b> {round(heart*100, 2)}%", styles['Normal']))
    content.append(Paragraph(f"<b>Obesity Risk:</b> {round(obesity*100, 2)}%", styles['Normal']))
    
    content.append(Spacer(1, 12))
    content.append(Paragraph("Health Suggestions:", styles['Heading2']))
    
    for s in suggestions:
        content.append(Paragraph(f"• {s}", styles['Normal']))
        content.append(Spacer(1, 6))

    doc.build(content)
    return file_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Data lena
    data = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodpressure']),
        float(request.form['skinthickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]

    # 2. Predictions
    diabetes_pred = model_diabetes.predict_proba([data])[0][1]
    heart_pred = model_heart.predict_proba([data])[0][1]
    obesity_pred = model_obesity.predict_proba([data])[0][1]

    # 3. Suggestions Logic
    suggestions = []
    if data[1] > 140: suggestions.append("High glucose level detected.")
    if data[5] > 30: suggestions.append("BMI indicates obesity.")
    if data[2] > 140: suggestions.append("High blood pressure detected.")
    suggestions.append("Maintain a healthy lifestyle with regular activity.")

    # 4. PDF Generate 
    generate_pdf_report(diabetes_pred, heart_pred, obesity_pred, suggestions)

    return render_template(
        'result.html',
        diabetes=diabetes_pred,
        heart=heart_pred,
        obesity=obesity_pred,
        suggestions=suggestions
    )

@app.route('/download')
def download():
    return send_file("report.pdf", as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)