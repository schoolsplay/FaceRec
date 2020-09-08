# this need te be run manually to setup the dbase

from app import db
from Constants import DEBUG

if bool(DEBUG):
    db.drop_all()
db.create_all()

