import os
from flask import Flask, render_template, request
import csv




app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        filename = uploaded_file.filename
        segments = filename.split("_")
        name=list(segments)[0]
        data = [
            ["ID","MGMT"],
            [name + "_11"],
            ]

        csv_file = "D:/flask/uploads/my_data.csv"

        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

        print(f"Data saved to {csv_file} successfully.")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)
        return f'File {segments[0]} uploaded successfully.'

    return 'No file uploaded.'


from mymodel import predict    
@app.route('/predict', methods=['POST'])   
def predictions():
    result = predict()
    return result


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
