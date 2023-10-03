import os
from flask import Flask, render_template, request, jsonify
import csv
from mymodel import predict    

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    modality_files = request.files.getlist('files')
    segm_file = request.files['segm']

    if modality_files and segm_file:
        for uploaded_file in modality_files:
            if uploaded_file.filename != '':
                filename = uploaded_file.filename
                segments = filename.split("_")
                name = list(segments)[0]
                data = [
                    ["ID", "MGMT"],
                    [name + "_11"],
                ]

                csv_file = os.path.join(app.config['UPLOAD_FOLDER'], 'my_data.csv')

                with open(csv_file, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(data)

                print(f"Data saved to {csv_file} successfully.")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(file_path)

        # Handle the segmented file (segm_file) here.
        segm_filename = segm_file.filename
        segm_file_path = os.path.join(app.config['UPLOAD_FOLDER'], segm_filename)
        segm_file.save(segm_file_path)

        predictions = predict()

        return jsonify({"message": "Files uploaded successfully.",
                        "pred": predictions})

    return jsonify({"message": "No files uploaded. Coud not make predictions."})


@app.route('/predict', methods=['POST'])   
def predictions():
    result = predict()
    return result


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
