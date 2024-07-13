import os
import torch
from flask import Flask, request, render_template, send_from_directory, send_file
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        with torch.autocast("cuda"):
            outputs = pipeline(prompt, height=512, width=512)
        image = outputs.images[0]
        image_path = os.path.join('static', 'generated_image.png')
        image.save(image_path)
        return render_template('index.html', image_path=image_path)
    return render_template('index.html', image_path=None)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/download_image')
def download_image():
    return send_file('static/generated_image.png', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
