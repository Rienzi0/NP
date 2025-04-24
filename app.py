from flask import Flask, request, render_template_string
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file:
            filepath = os.path.join('/home/NP/nutrinet', 'rgb.png')
            file.save(filepath)

            os.system('python inference.py > output.txt')

            with open('output.txt', 'r') as f:
                output = f.read().splitlines()
            
            calories = round(float(output[0]))
            mass = round(float(output[1]))
            fat = round(float(output[2]))
            carbs = round(float(output[3]))
            protein = round(float(output[4]))

            return render_template_string('''
                <link rel="stylesheet" href="/static/style.css">
                <div class="result-container">
                    <h1>Nutrition Prediction</h1>
                    <div class="result-item">Calories: <span>{{calories}}</span></div>
                    <div class="result-item">Mass: <span>{{mass}}</span> grams</div>
                    <div class="result-item">Fat: <span>{{fat}}</span> grams</div>
                    <div class="result-item">Carbs: <span>{{carbs}}</span> grams</div>
                    <div class="result-item">Protein: <span>{{protein}}</span> grams</div>
                </div>
            ''', calories=calories, mass=mass, fat=fat, carbs=carbs, protein=protein)

    return '''
    <link rel="stylesheet" href="/static/style.css">
    <h1>Upload an Image</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
