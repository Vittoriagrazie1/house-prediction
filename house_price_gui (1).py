# ============================================================
# House Price Prediction — Tkinter GUI
# Zero external libraries — pure Python only
# ML: Linear Regression built from scratch
# ============================================================

import tkinter as tk
from tkinter import ttk
import random
import math


# ─────────────────────────────────────────────────────────────
# 1. DATASET  (pure Python, no numpy/pandas)
# ─────────────────────────────────────────────────────────────

def generate_dataset(n=300, seed=42):
    random.seed(seed)
    nb_bonus = {"A": 30000, "B": 15000, "C": 0}
    data = []
    for _ in range(n):
        area   = random.randint(500, 4000)
        bed    = random.randint(1, 5)
        bath   = random.randint(1, 4)
        age    = random.randint(0, 50)
        dist   = round(random.uniform(1, 30), 1)
        garage = random.randint(0, 1)
        pool   = random.randint(0, 1)
        nb     = random.choice(["A", "B", "C"])
        noise  = random.gauss(0, 12000)

        price = (
            area   * 150
            + bed  * 8000
            + bath * 5000
            - age  * 500
            - dist * 3000
            + garage * 12000
            + pool   * 20000
            + nb_bonus[nb]
            + noise
        )
        price = max(50000, min(1_000_000, price))

        # one-hot neighborhood
        nb_b = 1 if nb == "B" else 0
        nb_c = 1 if nb == "C" else 0

        data.append([area, bed, bath, age, dist, garage, pool, nb_b, nb_c, price])

    return data   # list of rows: [...features..., price]


FEATURE_NAMES = [
    "area_sqft", "bedrooms", "bathrooms", "age_years",
    "distance_km", "has_garage", "has_pool", "nb_B", "nb_C"
]


# ─────────────────────────────────────────────────────────────
# 2. MATH HELPERS
# ─────────────────────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values)

def std(values):
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def transpose(matrix):
    return [[matrix[r][c] for r in range(len(matrix))] for c in range(len(matrix[0]))]

def mat_mul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

def mat_vec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def inverse(M):
    """Gauss-Jordan inverse for small square matrix."""
    n = len(M)
    aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        factor = aug[col][col]
        if abs(factor) < 1e-12:
            raise ValueError("Matrix is singular.")
        aug[col] = [x / factor for x in aug[col]]
        for row in range(n):
            if row != col:
                f = aug[row][col]
                aug[row] = [aug[row][k] - f * aug[col][k] for k in range(2 * n)]
    return [row[n:] for row in aug]


# ─────────────────────────────────────────────────────────────
# 3. STANDARD SCALER (from scratch)
# ─────────────────────────────────────────────────────────────

class Scaler:
    def fit(self, X):
        cols = transpose(X)
        self.means = [mean(c) for c in cols]
        self.stds  = [std(c) or 1.0 for c in cols]

    def transform(self, X):
        return [
            [(X[i][j] - self.means[j]) / self.stds[j] for j in range(len(X[i]))]
            for i in range(len(X))
        ]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ─────────────────────────────────────────────────────────────
# 4. LINEAR REGRESSION (from scratch, OLS normal equation)
# ─────────────────────────────────────────────────────────────

class LinearRegression:
    def fit(self, X, y):
        # Add bias column
        Xb = [[1.0] + row for row in X]
        Xt = transpose(Xb)
        XtX = mat_mul(Xt, Xb)
        Xty = mat_vec(Xt, y)
        self.coef_ = mat_vec(inverse(XtX), Xty)

    def predict(self, X):
        Xb = [[1.0] + row for row in X]
        return [sum(Xb[i][j] * self.coef_[j] for j in range(len(self.coef_)))
                for i in range(len(Xb))]

    def feature_importance(self):
        # Absolute value of coefficients (skip bias at index 0)
        coefs = [abs(c) for c in self.coef_[1:]]
        total = sum(coefs) or 1
        return [c / total for c in coefs]


# ─────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────

def r2_score(y_true, y_pred):
    y_mean = mean(y_true)
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    ss_res = sum((y - p) ** 2 for y, p in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot if ss_tot else 0.0

def mae(y_true, y_pred):
    return mean([abs(y - p) for y, p in zip(y_true, y_pred)])

def rmse(y_true, y_pred):
    return math.sqrt(mean([(y - p) ** 2 for y, p in zip(y_true, y_pred)]))


# ─────────────────────────────────────────────────────────────
# 6. TRAIN
# ─────────────────────────────────────────────────────────────

def train():
    data = generate_dataset()
    random.seed(0)
    random.shuffle(data)

    split = int(0.8 * len(data))
    train_data = data[:split]
    test_data  = data[split:]

    X_train = [row[:-1] for row in train_data]
    y_train = [row[-1]  for row in train_data]
    X_test  = [row[:-1] for row in test_data]
    y_test  = [row[-1]  for row in test_data]

    scaler = Scaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)

    metrics = {
        "R²":   round(r2_score(y_test, preds), 4),
        "MAE":  round(mae(y_test, preds), 0),
        "RMSE": round(rmse(y_test, preds), 0),
    }

    return model, scaler, metrics


# ─────────────────────────────────────────────────────────────
# 7. GUI
# ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self, model, scaler, metrics):
        super().__init__()
        self.model  = model
        self.scaler = scaler
        self.metrics = metrics

        self.title("House Price Predictor")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")
        self._build()

    # ── UI ───────────────────────────────────────────────────

    def _build(self):
        # Title
        hdr = tk.Frame(self, bg="#1a1a2e")
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(hdr, text="  House Price Predictor", bg="#1a1a2e", fg="white",
                 font=("Helvetica", 13, "bold"), pady=10).pack(side="left")
        tk.Label(hdr, text="Linear Regression · Pure Python  ",
                 bg="#1a1a2e", fg="#9999bb", font=("Helvetica", 9)).pack(side="right")

        # Metrics bar
        mbar = tk.Frame(self, bg="#e8eaf6")
        mbar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        for k, v in self.metrics.items():
            val_str = f"${v:,.0f}" if k in ("MAE","RMSE") else str(v)
            cell = tk.Frame(mbar, bg="#e8eaf6")
            cell.pack(side="left", padx=18, pady=6)
            tk.Label(cell, text=k, bg="#e8eaf6", fg="#666688",
                     font=("Helvetica", 8)).pack()
            tk.Label(cell, text=val_str, bg="#e8eaf6", fg="#1a1a2e",
                     font=("Helvetica", 11, "bold")).pack()

        # Input frame
        inp = tk.LabelFrame(self, text="House Features", bg="#f0f0f0",
                            fg="#222222", font=("Helvetica", 10, "bold"),
                            padx=12, pady=8)
        inp.grid(row=2, column=0, columnspan=2, padx=14, pady=(10,4), sticky="ew")

        self.vars = {}
        sliders = [
            ("Area (sqft)",          "area_sqft",  500,  4000, 1800, 50),
            ("Bedrooms",             "bedrooms",   1,    5,    3,    1),
            ("Bathrooms",            "bathrooms",  1,    4,    2,    1),
            ("Age (years)",          "age_years",  0,    50,   10,   1),
            ("Distance to city (km)","distance_km",1,    30,   5,    1),
        ]

        for i, (lbl, key, mn, mx, default, res) in enumerate(sliders):
            tk.Label(inp, text=lbl, bg="#f0f0f0", fg="#444444",
                     font=("Helvetica", 9), width=22, anchor="w"
                     ).grid(row=i, column=0, padx=4, pady=3, sticky="w")
            var = tk.DoubleVar(value=default)
            self.vars[key] = var
            ttk.Scale(inp, from_=mn, to=mx, variable=var, orient="horizontal",
                      length=210, command=lambda v, k=key: self._refresh_label(k)
                      ).grid(row=i, column=1, padx=6, pady=3)
            lbl_w = tk.Label(inp, text=str(default), bg="#f0f0f0", fg="#1a1a2e",
                             font=("Helvetica", 9, "bold"), width=6)
            lbl_w.grid(row=i, column=2, padx=4)
            setattr(self, f"lbl_{key}", lbl_w)

        # Toggles
        self.garage_var = tk.IntVar(value=1)
        self.pool_var   = tk.IntVar(value=0)
        tog = tk.Frame(inp, bg="#f0f0f0")
        tog.grid(row=len(sliders), column=0, columnspan=3, pady=6, sticky="w")
        tk.Checkbutton(tog, text="Has Garage", variable=self.garage_var,
                       bg="#f0f0f0", font=("Helvetica", 9)).pack(side="left", padx=6)
        tk.Checkbutton(tog, text="Has Pool", variable=self.pool_var,
                       bg="#f0f0f0", font=("Helvetica", 9)).pack(side="left", padx=6)

        # Neighborhood
        nb_row = tk.Frame(inp, bg="#f0f0f0")
        nb_row.grid(row=len(sliders)+1, column=0, columnspan=3, pady=(2,6), sticky="w")
        tk.Label(nb_row, text="Neighborhood:", bg="#f0f0f0", fg="#444444",
                 font=("Helvetica", 9)).pack(side="left", padx=4)
        self.nb_var = tk.StringVar(value="A")
        ttk.Combobox(nb_row, textvariable=self.nb_var,
                     values=["A — Premium", "B — Mid-range", "C — Standard"],
                     state="readonly", width=16).pack(side="left")

        # Predict button
        tk.Button(self, text="Predict Price", command=self._predict,
                  bg="#1a1a2e", fg="white", font=("Helvetica", 11, "bold"),
                  relief="flat", padx=20, pady=8, cursor="hand2",
                  activebackground="#2e2e5e", activeforeground="white"
                  ).grid(row=3, column=0, columnspan=2, pady=10)

        # Result
        res_box = tk.Frame(self, bg="#dde3f8", bd=1, relief="solid")
        res_box.grid(row=4, column=0, columnspan=2, padx=14, pady=(0,10), sticky="ew")
        tk.Label(res_box, text="Estimated Price", bg="#dde3f8", fg="#555577",
                 font=("Helvetica", 9)).pack(pady=(8,0))
        self.result_var = tk.StringVar(value="—")
        tk.Label(res_box, textvariable=self.result_var, bg="#dde3f8", fg="#1a1a2e",
                 font=("Helvetica", 22, "bold")).pack(pady=(2,8))

        # Feature importance chart
        fi = tk.LabelFrame(self, text="Feature Importance", bg="#f0f0f0",
                           fg="#222222", font=("Helvetica", 10, "bold"), padx=10, pady=6)
        fi.grid(row=5, column=0, columnspan=2, padx=14, pady=(0,10), sticky="ew")
        self.fi_canvas = tk.Canvas(fi, bg="#f0f0f0", height=175, width=440,
                                   highlightthickness=0)
        self.fi_canvas.pack()
        self._draw_importance()

        # Footer
        tk.Label(self, text="Pure Python · No external libraries",
                 bg="#f0f0f0", fg="#aaaaaa", font=("Helvetica", 8)
                 ).grid(row=6, column=0, columnspan=2, pady=(0,8))

        self.columnconfigure(0, weight=1)

    # ── Helpers ──────────────────────────────────────────────

    def _refresh_label(self, key):
        raw = self.vars[key].get()
        lbl = getattr(self, f"lbl_{key}")
        if key in ("bedrooms", "bathrooms", "age_years"):
            val = int(round(raw))
            self.vars[key].set(val)
            lbl.config(text=str(val))
        elif key == "area_sqft":
            val = int(round(raw / 50) * 50)
            lbl.config(text=str(val))
        else:
            lbl.config(text=f"{raw:.1f}")

    def _predict(self):
        nb = self.nb_var.get()[0]
        row = [
            round(self.vars["area_sqft"].get() / 50) * 50,
            int(round(self.vars["bedrooms"].get())),
            int(round(self.vars["bathrooms"].get())),
            int(round(self.vars["age_years"].get())),
            round(self.vars["distance_km"].get(), 1),
            self.garage_var.get(),
            self.pool_var.get(),
            1 if nb == "B" else 0,
            1 if nb == "C" else 0,
        ]
        scaled = self.scaler.transform([row])
        price  = self.model.predict(scaled)[0]
        price  = max(50000, min(1_000_000, price))
        self.result_var.set(f"${price:,.0f}")

    def _draw_importance(self):
        imps = list(zip(FEATURE_NAMES, self.model.feature_importance()))
        imps.sort(key=lambda x: x[1])

        c = self.fi_canvas
        c.delete("all")
        x0, bar_h, gap = 130, 14, 8
        max_imp = max(i for _, i in imps) or 1
        max_w   = 260

        for idx, (feat, imp) in enumerate(imps):
            y     = 8 + idx * (bar_h + gap)
            bar_w = int((imp / max_imp) * max_w)
            c.create_text(x0 - 6, y + bar_h // 2, text=feat,
                          anchor="e", font=("Helvetica", 8), fill="#555555")
            c.create_rectangle(x0, y, x0 + bar_w, y + bar_h,
                               fill="#4f7cdc", outline="")
            c.create_text(x0 + bar_w + 5, y + bar_h // 2,
                          text=f"{imp:.3f}", anchor="w",
                          font=("Helvetica", 8), fill="#333333")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Training model...")
    model, scaler, metrics = train()
    print(f"R²={metrics['R²']}  MAE=${metrics['MAE']:,.0f}  RMSE=${metrics['RMSE']:,.0f}")
    print("Launching GUI...")
    App(model, scaler, metrics).mainloop()
