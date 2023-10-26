from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    models = db.relationship('MLModel', back_populates='user')
    scalers = db.relationship('Scaler', back_populates='user')
    nn_dropout_models = db.relationship('NNDropoutModel', back_populates='user')
    checkbox_value = db.Column(db.Boolean, default=True)
    default_model = db.Column(db.String(255), nullable=True, default='linear_regression-JDAMTHW')

class MLModel(db.Model):
    __tablename__ = 'ml_models'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    model_name = db.Column(db.String(255), nullable=False)
    model_data = db.Column(db.LargeBinary, nullable=False)
    user = db.relationship('User', back_populates='models')

class Scaler(db.Model):
    __tablename__ = 'scalers'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    scaler_name = db.Column(db.String(255), nullable=False)
    scaler_data = db.Column(db.LargeBinary, nullable=False)
    user = db.relationship('User', back_populates='scalers')

class NNDropoutModel(db.Model):
    __tablename__ = 'nn_dropout_model'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    model_name = db.Column(db.String(255), nullable=False)
    model_path = db.Column(db.String(255), nullable=False)
    user = db.relationship('User', back_populates='nn_dropout_models')
