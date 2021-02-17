from flask import Flask,render_template, request
import os
from model import pred_regression,pred_classification
import pandas as pd

app = Flask(__name__)


# Upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# Root URL for index.html
@app.route('/')
def index():
     # Set The upload HTML template '\templates\index.html'
    return render_template('index.html')



# Get the uploaded files
@app.route("/", methods=['POST'])
def uploadFiles():
      # get the uploaded file
      choose = request.form['choose']
      uploaded_file = request.files['file']
      filename = uploaded_file.filename

      file_path = os.path.join('static/uploads/', filename)
      uploaded_file.save(file_path)

      filel,file_extension = os.path.splitext(file_path)

      full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)

      if choose == 'CSV' and file_extension == '.csv':

          data = pd.read_csv(full_filename, header=0).head(5)
          data_details = pd.read_csv(full_filename, header=0)
          data_columns = data.columns.values

          row_no = data_details.shape[0]
          col_no = data_details.shape[1]


          return render_template('load_data.html',tables=[data.to_html(classes='data', header="true")],
                             file=full_filename,columns = data_columns,
                             row_no=row_no,col_no=col_no)

      elif choose == 'EXCEL' and file_extension == '.xls':

          data = pd.read_excel(full_filename, header=0).head(5)
          data_details = pd.read_excel(full_filename, header=0)
          data_columns = data.columns.values

          row_no = data_details.shape[0]
          col_no = data_details.shape[1]


          return render_template('load_data.html',tables=[data.to_html(classes='data', header="true")],
                             file=full_filename,columns = data_columns,
                             row_no=row_no,col_no=col_no)


      else:
          a = "File type do not match "
          return render_template('index.html',error = a )



#prediction result
@app.route("/result",methods=['POST'])
def predict():
    name = request.form['name']
    typeml = request.form['typeml']
    drop_data = request.form.getlist('mymultiselect[]')
    target = request.form['target']

    if typeml == "Regression":
        result_reg = pred_regression(name,target,drop_data)
        return render_template('result.html', score_lgbm=result_reg[0],rmse_lgbm=result_reg[1],
                               rmse_rfr=result_reg[2],score_rfr=result_reg[3],ml=typeml)

    elif typeml == "Classification":
        result_classi = pred_classification(name, target, drop_data)
        return render_template('result.html', score_lgbm=result_classi[0],
                               score_rfc=result_classi[1],ml=typeml)



if (__name__ == "__main__"):
     app.run(debug=True)