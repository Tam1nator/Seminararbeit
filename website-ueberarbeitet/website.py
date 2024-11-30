from flask import Flask, render_template, request, jsonify, redirect, flash, url_for
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
#from tensorflow import keras
#import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
#from keras.layers import Layer
#import tensorflow as tf
from sklearn.utils import resample
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
import pickle
from classes import User, MLModel, Scaler, NNDropoutModel, db
import os
from werkzeug.security import generate_password_hash
import math


app = Flask(__name__)
app.secret_key = 'secret_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:Password@localhost:3306/datenbank'
db.init_app(app)


login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Falsche Anmeldedaten', 'login_error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # Überprüfen, ob der Benutzer bereits existiert
        user = User.query.filter_by(email=email).first()
        if user:
            flash("Ein Benutzer mit dieser E-Mail-Adresse existiert bereits", 'register_error')
            return redirect(url_for('login'))

        # Neuen Benutzer erstellen und zur Datenbank hinzufügen
        new_user = User(email=email, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()

        # Den Benutzer nach erfolgreicher Registrierung anmelden
        login_user(new_user)
        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    global model_scalers, available_models, neural_network_dropout, default_model_name
    model_scalers = {
    'linear_regression-JDAMTHW': scaler,
    'decision_tree-JDAMTHW': scaler,
    'random_forest-JDAMTHW': scaler,
    'neural_network-JDAMTHW': scaler,
    'svr_model-JDAMTHW': scaler,
    'xgboost-JDAMTHW': scaler
    }
    available_models = {
    "linear_regression-JDAMTHW": pretrained_linear_regression,
    "decision_tree-JDAMTHW": pretrained_decision_tree,
    "random_forest-JDAMTHW" : pretrained_random_forest,
    "neural_network-JDAMTHW" : None, #pretrained_neural_network
    "svr_model-JDAMTHW" : pretrained_svr_model,
    "xgboost-JDAMTHW" : None #xgb_model
    }
    neural_network_dropout = {}
    default_model_name = 'linear_regression-JDAMTHW'
    return redirect(url_for('index'))

# Dictionary zum Speichern der trainierten Modelle
models = {}

'''class MonteCarloDropout(Layer):
    def __init__(self, rate, **kwargs):
        super(MonteCarloDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)

    def compute_output_shape(self, input_shape):
        return input_shape'''

pretrained_linear_regression = joblib.load('website-ueberarbeitet\\trainierte_modelle\\linear_regression_model.pkl')
pretrained_decision_tree = joblib.load('website-ueberarbeitet\\trainierte_modelle\\decision_tree_model.pkl')
pretrained_random_forest = joblib.load('website-ueberarbeitet\\trainierte_modelle\\random_forest_model.pkl')
#pretrained_neural_network = keras.models.load_model('website-ueberarbeitet\\trainierte_modelle\\neural_network.h5')  
#pretrained_neural_network_dropout = keras.models.load_model('website-ueberarbeitet\\trainierte_modelle\\neural_network_dropout.h5', custom_objects={'MonteCarloDropout': MonteCarloDropout})  
pretrained_svr_model = joblib.load('website-ueberarbeitet\\trainierte_modelle\\svr_model.pkl')

# Scaler laden
scaler = joblib.load('website-ueberarbeitet\\trainierte_modelle\\neural_network_scalar.pkl')

model_scalers = {
    'linear_regression-JDAMTHW': scaler,
    'decision_tree-JDAMTHW': scaler,
    'random_forest-JDAMTHW': scaler,
    'neural_network-JDAMTHW': scaler,
    'svr_model-JDAMTHW': scaler,
    'xgboost-JDAMTHW': scaler
}

#xgb_model = xgb.Booster(model_file='website-ueberarbeitet\\trainierte_modelle\\xgboost_model.json')

available_models = {
    "linear_regression-JDAMTHW": pretrained_linear_regression,
    "decision_tree-JDAMTHW": pretrained_decision_tree,
    "random_forest-JDAMTHW" : pretrained_random_forest,
    "neural_network-JDAMTHW" : None , #pretrained_neural_network
    "svr_model-JDAMTHW" : pretrained_svr_model,
    "xgboost-JDAMTHW" : None #xgb_model
}

training_data = pd.read_csv('website-ueberarbeitet\data\jahreszeit-kodiert.csv')

neural_network_dropout = {}

default_model_name = 'linear_regression-JDAMTHW'

@app.route('/auswertung')
def auswertung():
    try:
        # Pfad zur CSV-Datei
        filepath = 'C:/Users/tamin/OneDrive/Dokumente/Schule/Sek 2/Seminararbeit/website-ueberarbeitet/data/Auswertung.csv'
        
        # Lesen der CSV-Datei
        auswertung_data = pd.read_csv(filepath, sep=';')
        auswertung_data_list = auswertung_data.to_dict(orient='records')

    except Exception as e:
        print(e)  # Fehlerausgabe für das Debugging
        auswertung_data_list = []  # Leere Liste, falls das Lesen der Datei fehlschlägt

    return render_template('auswertung.html', auswertung_data=auswertung_data_list)

@app.route('/mse', methods=['GET', 'POST'])
def mse():
    model_name = request.form.get('model_name') if request.method == 'POST' else default_model_name
    supported_columns = extract_supported_columns(model_name)
    test_data = pd.read_csv('website-ueberarbeitet/data/combined_test_data.csv')
    
    # Überprüfe, ob das Modell und der Scaler verfügbar sind
    if model_name not in available_models or model_name not in model_scalers:
        return "Modell oder Scaler nicht gefunden", 404
    
    model = available_models[model_name]
    scaler = model_scalers[model_name]
    
    # Bereite die Daten vor
    data_to_predict = test_data[supported_columns]
    data_to_predict_scaled = scaler.transform(data_to_predict)
    
    # Berechne die Vorhersagen
    '''if isinstance(model, xgb.Booster):  # check if the model is an instance of XGBoost Booster
        dmatrix_data = xgb.DMatrix(data_to_predict_scaled)  # convert the data to DMatrix format
        predictions = model.predict(dmatrix_data).tolist()  # make predictions
    else:'''
    predictions = model.predict(data_to_predict_scaled).tolist()  # for non-xgboost models

    
    # Überprüfe, ob die Vorhersagen eine verschachtelte Liste sind (z.B. bei einigen Scikit-Learn Modellen)
    if isinstance(predictions[0], list):
        predictions = [item for sublist in predictions for item in sublist]
    
    # Füge die Vorhersagen zu den Testdaten hinzu
    test_data['Vorhersage'] = predictions
    test_data_list = test_data.to_dict(orient="records")
    
    # Berechne den MSE
    actual_values = test_data['Anzahl der Waldbrände']
    mse_value = mean_squared_error(actual_values, predictions)
    rmse = math.sqrt(mse_value)

    # Erweiterter Code zum Speichern der Ergebnisse in der CSV-Datei
    '''
    csv_file_path = 'C:/Users/tamin/OneDrive/Dokumente/Schule/Sek 2/Seminararbeit/Auswertung.csv'  # Geben Sie den Pfad zur CSV-Datei an

    # Überprüfen, ob die Datei bereits existiert und die Spalten enthält
    if not os.path.exists(csv_file_path) or not os.path.isfile(csv_file_path):
        df = pd.DataFrame(columns=['Modell', 'MSE', 'RMSE'])
    else:
        df = pd.read_csv(csv_file_path, sep=';')

    # Überprüfen, ob der Modellname bereits vorhanden ist
    if model_name not in df['Modell'].values:
        # Füge neue Zeile mit den Werten hinzu, wenn Modellname noch nicht vorhanden ist
        new_row_df = pd.DataFrame([{'Modell': model_name, 'MSE': mse_value, 'RMSE': rmse}])
        df = pd.concat([df, new_row_df], ignore_index=True)

    # Speichern der Datei
    df.to_csv(csv_file_path, index=False, sep=';')
    '''
    
    return render_template('mse.html', test_data=test_data_list, model_name=model_name, mse_value=mse_value, rmse=rmse)

def get_user_id():
    if current_user and current_user.is_authenticated:
        return int(current_user.get_id())
    return None

def save_model_to_db(user_id, model_name, model):
    model_data = pickle.dumps(model)
    new_model = MLModel(user_id=user_id, model_name=model_name, model_data=model_data)
    db.session.add(new_model)
    db.session.commit()

def load_model_from_db(user_id):
    global available_models
    models = MLModel.query.filter_by(user_id=user_id).all()
    loaded_models = {model.model_name: pickle.loads(model.model_data) for model in models}
    available_models.update(loaded_models)
    return loaded_models

def save_scaler_to_db(user_id, scaler_name, scaler):
    scaler_data = pickle.dumps(scaler)
    
    # Neuen Skalierer hinzufügen
    new_scaler = Scaler(user_id=user_id, scaler_name=scaler_name, scaler_data=scaler_data)
    db.session.add(new_scaler)
    db.session.commit()

def load_scaler_from_db(user_id):
    global model_scalers
    scalers = Scaler.query.filter_by(user_id=user_id).all()
    loaded_scalers = {scaler.scaler_name: pickle.loads(scaler.scaler_data) for scaler in scalers}
    model_scalers.update(loaded_scalers)
    return loaded_scalers

trained_models_dir = "website-ueberarbeitet\\trainierte_modelle\\"

'''def save_nn_dropout_model_to_db(user_id, model_name, nn_dropout_model):
    if not os.path.exists(trained_models_dir):
        os.makedirs(trained_models_dir)

    model_path = os.path.join(trained_models_dir, f"{model_name}.pkl")
    nn_dropout_model.save(model_path)
    
    new_nn_dropout_model = NNDropoutModel(user_id=user_id, model_name=model_name, model_path=model_path)
    db.session.add(new_nn_dropout_model)
    db.session.commit()


def load_nn_dropout_model(user_id):
    global neural_network_dropout
    nn_dropout_models = db.session.query(NNDropoutModel).filter_by(user_id=user_id).all()
    loaded_nn_dropout_models = {}
    
    for model in nn_dropout_models:
        loaded_model = tf.keras.models.load_model(model.model_path, custom_objects={'MonteCarloDropout': MonteCarloDropout})
        loaded_nn_dropout_models[model.model_name] = loaded_model
    
    neural_network_dropout.update(loaded_nn_dropout_models)
    return loaded_nn_dropout_models
'''
@app.route('/set_defaults', methods=['POST'])
@login_required
def set_defaults():
    interval = 'my_checkbox' in request.form  # Überprüft, ob die Checkbox angekreuzt ist
    default_model = default_model_name
    
    current_user.checkbox_value = interval
    current_user.default_model = default_model
    
    db.session.commit()
    
    return redirect(url_for('index'))


@app.route('/', methods=['GET', 'POST'])
def index(): 
    global default_model_name
    checkbox_default_value = True
    user_id = get_user_id()
    if current_user.is_authenticated:
        load_model_from_db(user_id)
        load_scaler_from_db(user_id)
        #load_nn_dropout_model(user_id)
        default_model_name = current_user.default_model
        checkbox_default_value = current_user.checkbox_value
    # Testdaten einmalig laden
    test_data = pd.read_csv('website-ueberarbeitet\data\combined_test_data.csv')
    test_data_rounded = pd.read_csv('website-ueberarbeitet\data\werte_gerundet_test.csv')
    prediction = None
    interval = None  # Neuer Wert für das Vorhersageintervall
    model_name = request.form.get('model_name') if request.method == 'POST' else default_model_name
    default_model_name = model_name
    interval = (None, None) 
    mse = None
    rmse = None
    true_prediction = None
    dropout_prediction = None

    if model_name in available_models:
        model = available_models.get(model_name)
        supported_columns = extract_supported_columns(model_name)
        selected_model_test_data = test_data
        selected_model_test_data_rounded = test_data_rounded
    else:
        model = None
        supported_columns = []
        selected_model_test_data = test_data
        selected_model_test_data_rounded = test_data_rounded

    if request.method == 'POST' and model:  # Überprüfung, ob ein Modell trainiert wurde
        # Daten für die Vorhersage vorbereiten
        data = {}

        checkbox_default_value = 'my_checkbox' in request.form
        if user_id:
            set_defaults()

        for column in supported_columns:
            value = request.form.get(column)
            if value is not None and value.strip() != '':
                data[column] = [float(value)]
            else:
                data[column] = [None]  # oder setzen Sie einen Standardwert

        data_to_predict = pd.DataFrame(data)

        if model_name in available_models and model_name in model_scalers:
            model = available_models.get(model_name)
            scaler = model_scalers[model_name]
            # Beispiel für neue Daten
            new_data = data_to_predict  
            # Daten skalieren
            new_data_scaled = scaler.transform(new_data)

        # Vorhersage
        if model_name.startswith("random_forest"):
            # Für Random Forest: Alle Vorhersagen der Bäume verwenden
            predictions = [tree.predict(new_data_scaled) for tree in model.estimators_]
            lower_bound = np.percentile(predictions, 2.5)  # 2.5. Perzentil
            upper_bound = np.percentile(predictions, 97.5)  # 97.5. Perzentil
            prediction = np.mean(predictions)
            if checkbox_default_value:
                interval = (lower_bound, upper_bound)

            '''elif model_name.startswith("xgboost"):
                if model_name == 'xgboost-JDAMTHW':
                    dmatrix_data = xgb.DMatrix(new_data_scaled)
                    data = dmatrix_data
                else:
                    data = new_data_scaled
                prediction = model.predict(data)[0]
                if checkbox_default_value:
                    interval = (predict_interval_bootstrap(model, new_data_scaled, model_name))'''

            '''elif model_name == "neural_network-JDAMTHW":
                # Beispiel für neue Daten
                new_data = data_to_predict  # Diese Daten wurden bereits oben erstellt.
                # Daten skalieren
                new_data_scaled = scaler.transform(new_data)
                prediction = model.predict(new_data_scaled)[0][0]

                if checkbox_default_value:
                    T = 100  # Anzahl der Dropout-Vorhersagen
                    predictions = np.array([pretrained_neural_network_dropout.predict(new_data_scaled) for _ in range(T)]).flatten()
                    dropout_prediction = pretrained_neural_network_dropout.predict(new_data_scaled)[0][0]

                    prediction_mean = predictions.mean()
                    prediction_std = predictions.std()
                    lower_bound = prediction_mean - 1.96 * prediction_std
                    upper_bound = prediction_mean + 1.96 * prediction_std

                    interval = (lower_bound, upper_bound)

            elif model_name.startswith("neural_network"):
        
                prediction = model.predict(new_data_scaled)[0][0]
                if checkbox_default_value:
                    T = 100  # Anzahl der Dropout-Vorhersagen
                    predictions = np.array([neural_network_dropout[model_name].predict(new_data_scaled) for _ in range(T)]).flatten()
                    dropout_prediction = neural_network_dropout[model_name].predict(new_data_scaled)[0][0]

                    prediction_mean = predictions.mean()
                    prediction_std = predictions.std()
                    lower_bound = prediction_mean - 1.96 * prediction_std
                    upper_bound = prediction_mean + 1.96 * prediction_std

                    interval = (lower_bound, upper_bound)'''

        elif model_name == "svr_model-JDAMTHW":
            # Beispiel für neue Daten
            new_data = data_to_predict  # Diese Daten wurden bereits oben erstellt.
            # Daten skalieren
            new_data_scaled = scaler.transform(new_data)

            if checkbox_default_value:
                interval = (predict_interval_bootstrap(model, new_data_scaled, model_name))

            prediction = model.predict(new_data_scaled)[0]

        else:
            new_data = data_to_predict  # Diese Daten wurden bereits oben erstellt.
            # Daten skalieren
            new_data_scaled = scaler.transform(new_data)
            prediction = model.predict(new_data_scaled)[0]

            if checkbox_default_value:
                interval = (predict_interval_bootstrap(model, new_data_scaled, model_name))
        
        if request.method == 'POST' and model:
            selected_data_index = request.form.get('selected_row_index')
            if selected_data_index:
                selected_data_index = int(selected_data_index)
                
                # Überprüfen Sie, ob die eingegebenen Daten mit den Testdaten übereinstimmen
                data_matches = all(
                    float(request.form.get(column)) == test_data.loc[selected_data_index, column]
                    for column in supported_columns
                )
                data_matches_rounded = all(
                    float(request.form.get(column)) == test_data_rounded.loc[selected_data_index, column]
                    for column in supported_columns
                )
                
                if data_matches or data_matches_rounded:
                    if data_matches:
                        true_value = test_data.loc[selected_data_index, 'Anzahl der Waldbrände']
                    elif data_matches_rounded:
                        true_value = test_data_rounded.loc[selected_data_index, 'Anzahl der Waldbrände']
                    mse = calculate_mse([true_value], [prediction])
                    true_prediction = request.form.get('true_prediction')
                    rmse = math.sqrt(mse)
                    
                else:
                    mse = None
                    rmse = None

        #für decision_tree sonst js interpretiert als null/undefined
        if prediction == 0.0:
            prediction = "0"
        if mse == 0.0:
            mse = "0"
        if rmse == 0:
            rmse = "0"

    return render_template('index.html', prediction=prediction, rmse=rmse, dropout_prediction=dropout_prediction, interval=interval, mse=mse, true_prediction=true_prediction, test_data=list(enumerate(selected_model_test_data.iterrows())), test_data_columns=selected_model_test_data.columns, test_data_rounded=list(enumerate(selected_model_test_data_rounded.iterrows())), test_data_rounded_columns=selected_model_test_data_rounded.columns, available_models=available_models, supported_columns=supported_columns, default_model=default_model_name, checkbox_value=checkbox_default_value)

def predict_interval_bootstrap(model, new_data_scaled, model_name, n_iterations=10, alpha=0.05):
    """
    Berechnet das zugehörige (1-alpha) * 100 % Vorhersageintervall 
    unter Verwendung des Bootstrapping-Ansatzes.
    """

    column_legend = {
        'J': 'Jahreszeit',
        'D': 'Dauer',
        'A': 'avgtemp_c',
        'M': 'maxwind_kph',
        'T': 'totalprecip_mm',
        'H': 'avghumidity',
        'W': 'Waldfläche in Hektar'
    }
    column_order = {
        'Jahreszeit': 1,
        'Dauer': 2,
        'avgtemp_c': 3,
        'maxwind_kph': 4,
        'totalprecip_mm': 5,
        'avghumidity': 6,
        'Waldfläche in Hektar': 7
    }

    # Modellnamen zerteilen, um das Suffix zu extrahieren
    #suffix = model_name.split("-")[-1]

    # Modellnamen zerteilen, um das Suffix zu extrahieren
    cleaned_model_name = model_name[:-2] if model_name.endswith("-G") else model_name

    # Teile den bereinigten Modellnamen und extrahiere das Suffix
    parts = cleaned_model_name.split("-")
    suffix = "-".join(parts[1:])  # Nimmt alle Teile nach dem ersten "-"

    # Spaltennamen aus dem Suffix extrahieren
    selected_columns = [column_legend[char] for char in suffix]

    selected_columns = sorted(selected_columns, key=lambda x: column_order[x])

    checkbox_round_data = 'checkbox_round_data' in request.form

    if checkbox_round_data:
        training_data = pd.read_csv('website-ueberarbeitet\data\werte_gerundet.csv')
    else:
        training_data = pd.read_csv('website-ueberarbeitet\data\jahreszeit-kodiert.csv')

    X_train = training_data[selected_columns]
    y_train = training_data['Anzahl der Waldbrände']

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.0403, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Bootstrapping-Prozess
    bootstrap_predictions = []
    for _ in range(n_iterations):
        # Stichprobe mit Ersatz aus den Trainingsdaten ziehen
        X_sample, y_sample = resample(X_train_scaled, y_train)
        # Modell mit der Stichprobe trainieren
        '''if model_name.startswith("xgboost"):
            model_clone = xgb.XGBRegressor().fit(X_sample, y_sample)
        else:'''''
        model_clone = model
        model_clone.fit(X_sample, y_sample)
        # Vorhersage für `new_data` machen
        bootstrap_prediction = model_clone.predict(new_data_scaled)[0]
        bootstrap_predictions.append(bootstrap_prediction)
    
    # Vorhersageintervall berechnen
    lower = np.percentile(bootstrap_predictions, 100 * alpha / 2)
    upper = np.percentile(bootstrap_predictions, 100 * (1 - alpha / 2))
    interval = (lower, upper)
    
    return interval

def prediction_interval(prediction, percentage=0.1):

    interval = prediction * percentage

    lower_bound = prediction - interval
    upper_bound = prediction + interval
    
    return lower_bound, upper_bound

def extract_supported_columns(model_name):
    column_legend = {
    'J': 'Jahreszeit',
    'D': 'Dauer',
    'A': 'avgtemp_c',
    'M': 'maxwind_kph',
    'T': 'totalprecip_mm',
    'H': 'avghumidity',
    'W': 'Waldfläche in Hektar'
    }
    if '-' not in model_name:
        return []
    supported_columns = [column_legend[char] for char in model_name.split('-')[1] if char in column_legend]
    return supported_columns

@app.route('/get_supported_columns', methods=['POST'])
def get_supported_columns():
    model_name = request.form.get('model_name')
    supported_columns = extract_supported_columns(model_name)
    return jsonify(supported_columns=supported_columns)

def calculate_mse(true_values, predicted_values):
    return mean_squared_error(true_values, predicted_values)

@app.route('/train_model', methods=['POST'])
def train_model():
    # Die ausgewählten Spalten aus dem Formular abrufen
    selected_columns = request.form.getlist('columns')
    
    # Stellen Sie sicher, dass die Pflichtspalten enthalten sind
    for mandatory_column in ['Dauer', 'avgtemp_c', 'Waldfläche in Hektar']:
        if mandatory_column not in selected_columns:
            selected_columns.append(mandatory_column)

    # Spaltenordnung
        column_order = {
            'Jahreszeit': 1,
            'Dauer': 2,
            'avgtemp_c': 3,
            'maxwind_kph': 4,
            'totalprecip_mm': 5,
            'avghumidity': 6,
            'Waldfläche in Hektar': 7
        }
    selected_columns = sorted(selected_columns, key=lambda x: column_order[x])

    checkbox_round_data = 'checkbox_round_data' in request.form

    if checkbox_round_data:
        training_data = pd.read_csv('website-ueberarbeitet\data\werte_gerundet.csv')
    else:
        training_data = pd.read_csv('website-ueberarbeitet\data\jahreszeit-kodiert.csv')
    # Filtern Sie die Trainingsdaten basierend auf den ausgewählten Spalten
    X = training_data[selected_columns]
    y = training_data['Anzahl der Waldbrände']  # Annahme, dass dies Ihre Zielvariable ist

    # Aufteilen der Daten in Trainings- und Testsätze
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0403, random_state=42)

    # Modell basierend auf dem gewählten Modellnamen trainieren
    model_name = request.form.get('model_name')
    if not model_name:
        return "Fehler: model_name wurde nicht gesendet!", 400


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Bezeichnung für das trainierte Modell basierend auf den ausgewählten Parametern
    param_codes = {
        'Jahreszeit': 'J',
        'Dauer': 'D',
        'avgtemp_c': 'A',
        'maxwind_kph': 'M',
        'totalprecip_mm': 'T',
        'avghumidity': 'H',
        'Waldfläche in Hektar': 'W'
    }
    
    selected_param_string = ''.join([param_codes[param] for param in selected_columns])
    if checkbox_round_data:
        model_display_name = f"{model_name}-{selected_param_string}-G"  # Hier hinzufügen Sie das Suffix
    else:
        model_display_name = f"{model_name}-{selected_param_string}"
    # Updating the default model name when a new model is trained
    global default_model_name
    default_model_name = model_display_name

    model_scalers[model_display_name] = scaler

    # Überprüfen, ob das Modell bereits trainiert wurde
    if model_display_name in available_models:
        flash('Das Modell wurde bereits mit denselben Parametern trainiert.', 'train-model-error')
        return redirect('/')
    
    if model_name == "linear_regression":
        model = LinearRegression().fit(X_train_scaled, y_train)
    elif model_name == "decision_tree":
        model = DecisionTreeRegressor().fit(X_train_scaled, y_train)
    elif model_name == "random_forest":
        model = RandomForestRegressor().fit(X_train_scaled, y_train)
        '''elif model_name == "xgboost":
            model = xgb.XGBRegressor().fit(X_train_scaled, y_train)
        elif model_name == "neural_network":
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_scaled, y_train, epochs=50)

            neural_network_dropout_model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                MonteCarloDropout(0.5),
                keras.layers.Dense(32, activation='relu'),
                MonteCarloDropout(0.5),
                keras.layers.Dense(1)
            ])
            neural_network_dropout_model.compile(optimizer='adam', loss='mse')
            neural_network_dropout_model.fit(X_train_scaled, y_train, epochs=50)'''
    elif model_name == "svr":
        model = SVR().fit(X_train_scaled, y_train)
    else:
        return "Ungültiger Modellname", 400

    # Das trainierte Modell im models-Dictionary speichern
    models[model_name] = model

    user_id = get_user_id()
    if user_id:
        user = User.query.get(user_id)
        user.default_model = model_display_name
        db.session.commit()

    '''if model_name == "neural_network":
        neural_network_dropout[model_display_name] = neural_network_dropout_model
        save_nn_dropout_model_to_db(user_id, model_display_name, neural_network_dropout_model)'''

    available_models[model_display_name] = model
    save_model_to_db(user_id, model_display_name, model)
    save_scaler_to_db(user_id, model_display_name, scaler)

    return redirect('/')
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
