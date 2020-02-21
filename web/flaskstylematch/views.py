from flaskstylematch import app
from flask import render_template
from flask import request
from flask import redirect
from functions import *
import os
from werkzeug.utils import secure_filename


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024

def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False

def allowed_image(filename):
    print(filename)

    # We only want files with a . in the filename
    if not "." in filename:
        return False

    # Split the extension from the filename
    ext = filename.rsplit(".", 1)[1]

    # Check if the extension is in ALLOWED_IMAGE_EXTENSIONS
    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route('/')
def index():

    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/images')
    if not os.path.isdir(target):
        os.mkdir(target)

    # Get uploaded image
    image = request.files['image']

    # Verify image
    if image.filename == "":
        print("No filename")
        return redirect(request.url)

    # Verify file has an image type file extension
    if allowed_image(image.filename) == False:
        print("File extension not allowed")
        return redirect(request.url)

    if "filesize" in request.cookies:
        if not allowed_image_filesize(request.cookies["filesize"]):
            print("Filesize exceeded maximum limit")
            return redirect(request.url)

    # Create a secure filename
    filename = secure_filename(image.filename)

    # Save the image
    destination = os.path.join(target, filename)
    image.save(destination)

    # Get predicted class
    predicted_class = predict(image)

    # Get complementary rugs
    first, second, third, first_link, second_link, third_link = get_rugs(predicted_class, filename)

    return render_template('result.html', style_class = predicted_class, filename = filename, first_rug = first, second_rug = second, third_rug = third, first_link = first_link, second_link = second_link, third_link = third_link)
