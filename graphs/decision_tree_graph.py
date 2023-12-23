import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Daten laden
data = pd.read_csv('C:\\Users\\tamin\\OneDrive\\Dokumente\\Schule\\Sek 2\\Seminararbeit\\website-ueberarbeitet\\data\\werte_gerundet_test.csv')
X = data[['Jahreszeit', 'Dauer', 'avgtemp_c', 'maxwind_kph', 'totalprecip_mm', 'avghumidity', 'Waldfläche in Hektar']]
y = data['Anzahl der Waldbrände']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0403, random_state=42)

# Daten skalieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modell trainieren
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Baum darstellen und speichern mit hoher DPI-Zahl
plt.figure(figsize=(40, 20))  # Größe des Plots anpassen
plot_tree(dt_model, feature_names=X.columns, filled=True, fontsize=10)
plt.savefig('decision_tree_high_dpi.png', dpi=300)  # Speichert den Baum als Bild mit hoher DPI
plt.tight_layout()
plt.show()  # Zeigt den Baum in einem Diagramm

# Vorhersagen machen
predictions = dt_model.predict(X_test_scaled)

# Ergebnisse tabellarisch darstellen
pd.set_option('display.max_rows', None)
results = X_test.copy()
results['Actual'] = y_test
results['Predicted'] = predictions
print(results)

# Mittleren quadratischen Fehler berechnen und ausgeben
mse = mean_squared_error(y_test, predictions)
print(f"\nMean Squared Error (MSE): {mse}")

# Modell speichern (auskommentiert)
# joblib.dump(dt_model, 'C:\\Users\\tamin\\OneDrive\\Dokumente\\Schule\\Sek 2\\Seminararbeit\\trainierte_modelle\\decision_tree_model.pkl')
