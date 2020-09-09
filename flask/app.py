from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from Constants import DBASE_URI

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DBASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

from model import FaceRec, User

@app.route('/')
def hello_world():
    return 'Hello, world!'


