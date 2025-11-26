from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from glasspy.predict import GlassNet
import os

app = Flask(__name__)
CORS(app)

# Lazy load del modello
glass_model = None

# Lista di elementi supportati
supported_oxides = [
    "SiO2","B2O3","Al2O3","Na2O","K2O","CaO","MgO","Li2O",
    "BaO","PbO","TiO2","ZrO2","Fe2O3","MnO","SrO","CeO2",
    "P2O5","ZnO"
]

# Endpoint energia / proprietà vetro
@app.route("/energy", methods=["POST"])
def calculate_energy():
    global glass_model
    if glass_model is None:
        glass_model = GlassNet()  # carica modello al primo uso

    data = request.json
    # Pulizia valori input, vuoti diventano 0
    composition = {k: float(v) if v != '' else 0.0 for k, v in data.items() if k in supported_oxides}
    
    # Predizione con GlassNet
    predictions = glass_model.predict(composition)
    pred = predictions.loc[0]

    result = {}

    # VISCOSITY
    viscosity_cols = [c for c in pred.index if "Viscosity" in c]
    result["Viscosity"] = {c: pred[c] for c in viscosity_cols}

    # HEAT CAPACITY Cp
    cp_cols = [c for c in pred.index if "Cp" in c]
    result["Cp"] = {c: pred[c] for c in cp_cols}

    # DENSITY
    density_cols = [c for c in pred.index if "Density" in c]
    result["Density"] = {c: pred[c] for c in density_cols}

    # CTE
    cte_cols = [c for c in pred.index if "CTE" in c]
    result["CTE"] = {c: pred[c] for c in cte_cols}

    # Surface Tension
    st_cols = [c for c in pred.index if "SurfaceTension" in c]
    result["SurfaceTension"] = {c: pred[c] for c in st_cols}

    # Resistivity
    res_cols = [c for c in pred.index if "Resistivity" in c]
    result["Resistivity"] = {c: pred[c] for c in res_cols}

    # Cp integrata per energia
    temps = np.array([293, 473, 673, 1073, 1273, 1473, 1673])
    cp_values = np.array([pred[c] for c in cp_cols[:7]])
    energy_MJ_per_kg = np.trapz(cp_values, temps) / 1e6
    result["SingleValues"] = {"Energy_MJ_per_kg_25-1400C": energy_MJ_per_kg}

    # PROPRIETÀ SCALARI SINGOLE
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


