from flask import Flask, Blueprint, jsonify, render_template, request, flash, redirect
import werkzeug
import nltk
nltk.download('punkt')

import application.utils as utils
import application.model as model

# flask blueprint
bp = Blueprint('application', __name__)

# homepage route
@bp.route('/')
def index():
    """
    Renders the webapp homepage
    """
    return render_template('index.html')

# submit url via text box route
@bp.route('/find', methods=['GET', 'POST'])
def find():
    """
    Handles requests made using the text box on homepage
    """
    # If request method is post, process the reddit link
    # else render the homepage again
    if request.method == 'POST':

        # check if some input was provided
        if not request.form.get('reddit_url'):
            raise RuntimeError('No url provided')
        else:
            # print the url for debugging purposes
            print('#debug url: ', request.form.get('reddit_url'))

            # get a list of dictionaries containing scraped data
            # in this case, a list with a single element
            data = utils.get_reddit_data([request.form.get('reddit_url')])

            # writeout data to load in prediction model
            utils.write_reddit_data(data)

            # get the dictionary containing predictions
            _, pred_object = model.run_model()

            # render the results
            return render_template('result.html', data=pred_object)
    else:
        # render homepage
        return render_template('index.html')

# submit text file via post route
@bp.route('/automated_testing', methods=['POST'])
def automated_testing():
    # If request method is post, process the text file
    # else render the homepage again
    if request.method == 'POST':

        # check some common keys first to get file
        # else check all keys and get the file storage object
        if 'file' in request.files.keys():
            file = request.files['file']
        elif 'upload_file' in request.files.keys():
            file = request.files['upload_file']
        else:
            # iterate over all keys
            for keys in request.files.keys():
                #print(keys)
                #print(type(request.files[keys]))
                #print(isinstance(request.files[keys], werkzeug.datastructures.FileStorage))

                # if value for current key is a file storage object
                # set file to it
                if isinstance(request.files[keys], werkzeug.datastructures.FileStorage):
                    file = request.files[keys]

        # save the file containing reddit links
        file.save('application/uploads/links.txt')

        # read each link from uploaded file in list
        links = []
        with open('application/uploads/links.txt', 'r') as f:
            for line in f:
                links.append(line[:-1])

        # get a list of dictionaries containing scraped data
        data = utils.get_reddit_data(links)

        # writeout data to load in prediction model
        utils.write_reddit_data(data)

        # get the json object of prediction results
        pred_json, _ = model.run_model()

        # return the json object
        return pred_json

    else:
        # render homepage
        return render_template('index.html')
