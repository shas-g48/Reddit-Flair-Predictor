from flask import Flask, Blueprint, jsonify, render_template, request, flash, redirect
import nltk
nltk.download('punkt')
import application.utils as utils
import application.model as model
import werkzeug

#UPLOAD_FOLDER = 'application/uploads'
#ALLOWED_EXTENSIONS = {'txt'}

bp = Blueprint('application', __name__)

#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# homepage route
@bp.route('/')
def index():
    return render_template('index.html')

# submit url via text box route
@bp.route('/find', methods=['GET', 'POST'])
def find():
    if request.method == 'POST':
        # check if some input was provided
        if not request.form.get('reddit_url'):
            raise RuntimeError('No url provided')
        else:
            print('#debug url: ', request.form.get('reddit_url'))

            # get a list of dictionaries containing scraped data
            data = utils.get_reddit_data([request.form.get('reddit_url')])

            # writeout data to load in prediction model
            utils.write_reddit_data(data)

            _, pred_object = model.run_model()

            # debugging template
            return render_template('result.html', data=pred_object)
    else:
        return render_template('index.html')



@bp.route('/automated_testing', methods=['POST'])
def automated_testing():
    #if request.method == 'GET':
    #    return render_template('test.html')

    if request.method == 'POST':
        '''print('r1')
        if 'file' not in request.files:
            print('e1')
            flash('No file part')
            return redirect(request.url)
        print('r2')
        file = request.files['file']


        if file.filename == '':
            print('e2')
            flash('No selected file')
            return redirect(request.url)
        #if file and allowed(file.filename):
        #    filename = secure_filename(file.filename)
        #    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #with open(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), 'r') as f:
        #    for line in f:
        #        print(line[:-1])
        print(file)
        print('ow')'''
        if 'file' in request.files.keys():
            file = request.files['file']
        else:
            for keys in request.files.keys():
                print(keys)
                print(type(request.files[keys]))
                print(isinstance(request.files[keys], werkzeug.datastructures.FileStorage))
                if isinstance(request.files[keys], werkzeug.datastructures.FileStorage):
                    file = request.files[keys]

        '''for key in request.files.keys():
            if request.files[key][-4:] == '.txt':
                file = request.files[key]
            else:
                raise RuntimeError('No file provided')'''

        #with open(file, 'r') as f:
        #    for line in f:
        #        print(line)
        file.save('application/uploads/links.txt')

        links = []
        with open('application/uploads/links.txt', 'r') as f:
            for line in f:
                links.append(line[:-1])

        data = utils.get_reddit_data(links)

        # writeout data to load in prediction model
        utils.write_reddit_data(data)

        pred_json, _ = model.run_model()

        # debugging template
        return pred_json



        #return ''
            #return redirect(url_for('uploaded_file',
            #                        filename=filename))



#def allowed_file(filename):
#    return '.' in filename and \
#           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
