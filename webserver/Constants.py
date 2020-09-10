
# Various constants for flask
import os
DEBUG = True
DROPDB = False

BASEDIR = os.path.dirname(os.path.abspath(__file__))
DBASE_URI = 'sqlite:///{}'.format(os.path.join(BASEDIR, 'facerec.db'))


