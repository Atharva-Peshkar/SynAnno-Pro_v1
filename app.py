import synAnno_processing as ip
import os, shutil
from flask import Flask, render_template, session, flash, jsonify, request, send_file, redirect
from flask_session import Session
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import json
import jsonschema
from jsonschema import validate
import sys

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = 'BAD_SECRET_KEY'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

app.config['UPLOAD_FOLDER'] = 'files/'
app.config['UPLOAD_EXTENSIONS'] = ['.json', '.h5']



@app.route('/')
def open_data():
    return render_template("opendata.html", modenext="disabled")


#opening json for now
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    #Check if files folder exists, if not create.
    if os.path.exists('./files'):
        pass
    else:
        os.mkdir('./files')
        
    file_original = request.files['file_original']
    file_gt = request.files['file_gt']
    file_json = request.files['file_json']

    # Check if there is files
    if file_original.filename == '' and file_gt.filename == '' and file_json.filename == '':
        flash("Please upload the original and ground truth .h5 files or a JSON file from a previous annotation session!")

    # Check if the process will start if .h5 files or a json.
    if file_original.filename != '' and file_gt.filename != '':
        original_name = save_file(file_original)
        gt_name = save_file(file_gt)
        if original_name!="error" and gt_name!="error":
            final_json = json.loads(ip.loading_3d_file(os.path.join(app.config['UPLOAD_FOLDER'], file_original.filename), os.path.join(app.config['UPLOAD_FOLDER'], file_gt.filename)))
            #Verify if the json is valid
            if validate_json(final_json):
                filename = "finaljson.json"
                with open(app.config['UPLOAD_FOLDER'] + filename, 'w') as f:
                    json.dump(final_json, f)
                flash("Data ready!")
                return render_template("opendata.html", filename=filename, modecurrent="disabled", modeform="formFileDisabled")
            else:
                print("JSON has not the correct format")
                flash("Something is wrong with the loaded data! Check the data and load again.")
                return render_template("opendata.html", modenext="disabled")
    elif file_json.filename == '':
        flash("Please, upload the original and ground truth .h5 files or a JSON file!")
    # Check if the process will happen with json file
    elif file_json.filename != '':
        filename = save_file(file_json)
        if filename != "error":
            #Verify if the json is valid
            f = open(app.config['UPLOAD_FOLDER'] + filename)
            final_json = json.load(f)
            print("Size da variavel entrada")
            print(sys.getsizeof(final_json))
            if validate_json(final_json):
                flash("Data ready!")
                return render_template("opendata.html", filename=filename, modecurrent="disabled", modeform="formFileDisabled")
            else:
                print("JSON has not the correct format")
                flash("Something is wrong with the loaded data! Check the data and load again.")
                return render_template("opendata.html", modenext="disabled")
    return render_template("opendata.html", modenext="disabled")


@app.route('/set-data/<data_name>')
@app.route('/set-data')
def set_data(data_name='synanno.json'):

    #set the number of cards in one page
    per_page = 30
    session['per_page'] = per_page

    #Open the json data and save it to the session
    f = open(app.config['UPLOAD_FOLDER'] + data_name) #tratar erro
    data = json.load(f)

    if not session.get('data'):
        session['data'] = [data['Data'][i:i+per_page] for i in range(0, len(data['Data']),per_page)]

    session['filename'] = data_name

    #Calculate the number of pages (based on 100 per page) and save it to the session
    number_images = len(data['Data'])
    number_pages = number_images // per_page
    if not (number_images % per_page == 0):
        number_pages = number_pages + 1
    if not session.get('n_pages'):
        session['n_pages'] = number_pages

    return render_template("annotation.html", images=session.get('data')[0], page=0, n_pages=session.get('n_pages'))


@app.route('/annotation')
@app.route('/annotation/<int:page>')
def annotation(page=0):
    return render_template("annotation.html", images=session.get('data')[page], page=page, n_pages=session.get('n_pages'))


@app.route('/finalpage')
def final_page():
    return render_template("exportdata.html")

@app.route('/finalize')
def finalize():
    # Pop all the session content.
    for key in list(session.keys()):
        session.pop(key)
    # Delete all files in files
    if os.path.exists("./"+app.config['UPLOAD_FOLDER']):
        for filename in os.listdir("./"+app.config['UPLOAD_FOLDER']):
            file_path = os.path.join("./"+app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    return render_template("opendata.html", modenext="disabled")


@app.route('/export')
def export_data():
    final_filename = "results-" + session.get('filename')
    # Exporting the final json and pop session
    if session.get('data') and session.get('n_pages') :
        final_file = dict()
        final_file["Data"] = sum(session['data'], [])
        with open(app.config['UPLOAD_FOLDER'] + final_filename, 'w') as f:
            json.dump(final_file, f)
        return send_file(app.config['UPLOAD_FOLDER'] + final_filename, as_attachment=True, attachment_filename=final_filename)
    else:
        return render_template("exportdata.html")
    return render_template("exportdata.html")



@app.route('/update-card', methods=['POST'])
@cross_origin()
def update_card():
    page = int(request.form['page'])
    index = int(request.form['data_id'])-1
    label = request.form['label']

    data = session.get('data')

    if (label == "Incorrect"):
        data[page][index]['Label'] = 'Unsure'
    elif (label == "Unsure"):
        data[page][index]['Label'] = 'Correct'
    elif (label == "Correct"):
        data[page][index]['Label'] = 'Incorrect'

    session['data'] = data
    print(data[page][index]['Label'])

    return jsonify({'result':'success', 'label': data[page][index]['Label']})


@app.route('/get_slice', methods=['POST'])
@cross_origin()
def get_slice():
    page = int(request.form['page'])
    index = int(request.form['data_id']) - 1
    slice = int(request.form['slice'])

    data = session.get('data')
    halfLen = len(data[page][index]['Before'])
    if slice <= 0:
        final_json = jsonify(data=data[page][index]['Before'][slice], halfLen=halfLen)
    if slice > 0:
        final_json = jsonify(data=data[page][index]['After'][slice], halfLen=halfLen)

    return final_json


def save_file(file):
    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1]
    if file_ext not in app.config['UPLOAD_EXTENSIONS']:
        print('incorrect format')
        flash("Incorrect file format! Load again.")
        render_template("opendata.html", modenext="disabled")
        return("error")
    else:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("saved file successfully")
        return(filename)
    return("ok")




def validate_json(json_data):
    f = open("static/json_schema.json")
    json_schema = json.load(f)
    print("Json schema size")
    print(sys.getsizeof(json_schema))
    try:
        validate(instance=json_data, schema=json_schema)
    except jsonschema.exceptions.ValidationError as err:
        return False
    return True


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
