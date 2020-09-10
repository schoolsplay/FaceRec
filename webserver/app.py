from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# needed so that we can import model from app in views
import model
import views

from Constants import DBASE_URI

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DBASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

app.add_url_rule('/', 'index', views.index)

if __name__ == '__main__':
    app.run(
        debug=True,
        threaded=True,
        host='127.0.0.1'
    )
