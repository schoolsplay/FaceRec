import secrets

from app import db
from datetime import datetime


class FaceRec(db.Model):
    """Main table for the FaceRec tool"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Unicode, nullable=False)
    known_person = db.Column(db.Boolean, default=False)
    access_date = db.Column(db.DateTime)
    note = db.Column(db.Unicode)
    creation_date = db.Column(db.DateTime, nullable=False)

    def __init__(self, *args, **kwargs):
        """On construction, set date of creation."""
        super().__init__(*args, **kwargs)
        self.creation_date = datetime.now()

class User(db.Model):
    """The User table, not used yet."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.Unicode, nullable=False)
    email = db.Column(db.Unicode, nullable=False)
    password = db.Column(db.Unicode, nullable=False)
    date_joined = db.Column(db.DateTime, nullable=False)
    token = db.Column(db.Unicode, nullable=False)

    def __init__(self, *args, **kwargs):
        """On construction, set date of creation."""
        super().__init__(*args, **kwargs)
        self.date_joined = datetime.now()
        self.token = secrets.token_urlsafe(64)
