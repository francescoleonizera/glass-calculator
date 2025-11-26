from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from glasspy.predict import GlassNet
import os

# -------------------------
# Configurazione Flask
# -------------------------
app = Flask(__name__)
# Consenti solo richieste dal tuo sito Aruba
CORS(app, origins=["https://www.submerged-combustion.com"])

# -------------------------
# Caricamento modello al boot
# -------------------------
glass_model = GlassNet()  # caricato subito all'avvio

# -------------------------
# Lista di elementi supportati
# -------------------------
supported_oxides = [
    "SiO2","B2O3","Al2O3","Na2O","K2O","CaO","MgO","Li2O",
    "BaO","PbO","TiO2","ZrO2","Fe2O3","MnO","SrO","CeO2",
    "P2O5","ZnO"
]

# -------------------------
# Health check endpoint
# -------------------------
@app.route("/")
def index():
    return "GlassNet API is running!"

# -------------------------
# Endpoint principale
# -------------------------
@app.route("/energy", methods=["POST"])
def calculate_energy():
    data = request.json
    composition = {k: float(v) if v != '' else 0.0 for k, v in data.items() if k in supported_oxides}

    predictions = glass_model.predict(composition)
    pred = predictions.loc[0]

    result = {}

    # Categorie principali
    for cat_name, key in [("Viscosity","Viscosity"), ("Cp","Cp"), ("Density","Density"),
                          ("CTE","CTE"), ("SurfaceTension","SurfaceTension"), ("Resistivity","Resistivity")]:
        cols = [c for c in pred.index if key in c]
        result[cat_name] = {c: pred[c] for c in cols}

    # Cp integrata per energia
    cp_cols = [c for c in pred.index if "Cp" in c][:7]
    temps = np.array([293, 473, 673, 1073, 1273, 1473, 1673])
    cp_values = np.array([pred[c] for c in cp_cols])
    energy_MJ_per_kg = np.trapz(cp_values, temps) / 1e6
    result["SingleValues"] = {"Energy_MJ_per_kg_25-1400C": energy_MJ_per_kg}

    # Scalar properties
    scalar_cols = [
        "AbbeNum","Tg","Tmelt","Tliquidus","TLittletons","TAnnealing","Tstrain","Tsoft",
        "TdilatometricSoftening","RefractiveIndex","RefractiveIndexLow","RefractiveIndexHigh",
        "MeanDispersion","Permittivity","TangentOfLossAngle","TresistivityIs1MOhm.m",
        "YoungModulus","ShearModulus","Microhardness","PoissonRatio","ThermalConductivity",
        "ThermalShockRes","MaxGrowthVelocity","TMaxGrowthVelocity","CrystallizationOnset",
        "CrystallizationPeak","NucleationTemperature","NucleationRate"
    ]
    for col in scalar_cols:
        if col in pred.index:
            result["SingleValues"][col] = pred[col]

    return jsonify(result)

# -------------------------
# Avvio server
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render imposta la porta con PORT
    app.run(host="0.0.0.0", port=port)
