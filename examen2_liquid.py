"""
EXAMEN 2 - Modelos Matematicos y Simulacion
YOLO (Ultralytics) para detectar derrames de liquido -> area del derrame -> modelo matematico

Dataset : cls-liquid (1 clase: "liquid", deteccion con bounding boxes)
Variables extraidas por la vision: A(t) = fraccion del cuadro cubierta por liquido, N(t) = numero de manchas
Modelo 1: EDO de crecimiento del derrame   dA/dt = q - k*A       (q: aporte, k: perdida/limpieza)
Modelo 2: Cadena de Markov de seguridad    Seguro -> Precaucion -> Critico (segun A)
Simulacion: Monte Carlo de la cadena bajo escenarios de fuga y de respuesta de limpieza

Uso (PowerShell):
    python examen2_liquid.py prepare --ndjson cls-liquid.ndjson
    python examen2_liquid.py train --data dataset/data.yaml --device 0
    python examen2_liquid.py run --weights resultados/best.pt --fuente video.mp4 --show
"""
import argparse
import json
import math
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("resultados")
OUT.mkdir(exist_ok=True)

# ------------------------- CONFIGURACION -------------------------
MODELO_BASE = "yolo11n.pt"
HIPERPARAMS = dict(epochs=50, imgsz=640, batch=16, lr0=0.01, patience=15, seed=42)
UMBRALES = (0.05, 0.25)          # A < 5% Seguro | 5-25% Precaucion | > 25% Critico
ESTADOS = ["Seguro", "Precaucion", "Critico"]
COLORES = [(0, 200, 0), (0, 200, 255), (0, 0, 255)]
SEED = 42


# ------------------------- FASE I: NDJSON -> FORMATO YOLO -------------------------
def _descargar(t):
    url, dest = t
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception:
            pass
    return False


def preparar(ndjson, destino="dataset"):
    """Descarga las imagenes y crea labels YOLO + data.yaml. El dataset trae solo train/val,
    asi que se separa un 10% del total como TEST (el examen pide Train/Val/Test)."""
    destino = Path(destino)
    rows = [json.loads(l) for l in open(ndjson, encoding="utf-8") if l.strip()]
    meta = rows[0]
    imgs = [r for r in rows if r.get("type") == "image"]
    train = [r for r in imgs if r["split"] == "train"]
    random.Random(SEED).shuffle(train)
    for r in train[: max(1, int(0.10 * len(imgs)))]:
        r["split"] = "test"

    tareas = []
    for r in imgs:
        sp = r["split"]
        (destino / "images" / sp).mkdir(parents=True, exist_ok=True)
        (destino / "labels" / sp).mkdir(parents=True, exist_ok=True)
        tareas.append((r["url"], destino / "images" / sp / r["file"]))
        lineas = [f"{int(b[0])} {b[1]} {b[2]} {b[3]} {b[4]}" for b in r.get("annotations", {}).get("boxes", [])]
        (destino / "labels" / sp / f"{Path(r['file']).stem}.txt").write_text("\n".join(lineas))

    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(_descargar, tareas))
    print(f"Imagenes descargadas: {sum(ok)}/{len(ok)}")
    if not all(ok):
        print("AVISO: algunas URLs fallaron (las URLs firmadas caducan). Vuelve a exportar el ndjson si pasa.")

    nombres = meta.get("class_names", {"0": "liquid"})
    # Sin "path": Ultralytics usa la carpeta del yaml (evita problemas con tildes en la ruta de Windows)
    yaml = f"train: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {k}: {v}\n" for k, v in nombres.items())
    (destino / "data.yaml").write_text(yaml)
    conteo = pd.Series([r["split"] for r in imgs]).value_counts().to_dict()
    print("Split:", conteo, "| data.yaml en", destino / "data.yaml")


# ------------------------- FASE II: ENTRENAMIENTO -------------------------
def entrenar(data_yaml, device=None):
    from ultralytics import YOLO
    model = YOLO(MODELO_BASE)
    model.train(data=data_yaml, device=device, project="runs", name="examen2_liquid",
                exist_ok=True, **HIPERPARAMS)
    best = Path(model.trainer.best)   # la ruta real depende de la version de Ultralytics
    import shutil
    shutil.copy(best, OUT / "best.pt")
    m = YOLO(str(best)).val(data=data_yaml, split="test", device=device)
    metricas = dict(precision=float(m.box.mp), recall=float(m.box.mr),
                    mAP50=float(m.box.map50), mAP50_95=float(m.box.map))
    (OUT / "metricas_test.json").write_text(json.dumps(metricas, indent=2))
    print("Metricas TEST:", metricas, "\nPesos:", best, "(copia en resultados/best.pt)")


# ------------------------- FASE III: VISION -> VARIABLES -------------------------
def area_union(xyxyn, g=200):
    """Fraccion del cuadro cubierta por la union de las cajas (sin doble conteo de solapes)."""
    m = np.zeros((g, g), bool)
    for x1, y1, x2, y2 in xyxyn:
        m[int(y1 * g): int(math.ceil(y2 * g)), int(x1 * g): int(math.ceil(x2 * g))] = True
    return float(m.mean())


def leer_fuente(fuente, fps_img, muestreo):
    """Devuelve (tiempo_s, frame). Acepta video o carpeta de imagenes (secuencia temporal)."""
    import cv2
    p = Path(fuente)
    if not p.exists():
        raise SystemExit(f"No existe '{fuente}'. Revisa el nombre/ruta del video (usa: dir *.mp4).")
    if p.is_dir():
        archivos = sorted(f for f in p.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for i, f in enumerate(archivos):
            yield i / fps_img, cv2.imread(str(f))
    else:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        paso = max(1, int(round(fps / muestreo)))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % paso == 0:
                yield i / fps, fr
            i += 1
        cap.release()


def extraer_variables(weights, fuente, fps_img=1.0, muestreo=2.0, conf=0.25, show=False):
    import cv2
    from ultralytics import YOLO
    model = YOLO(weights)
    filas, writer = [], None
    for t, frame in leer_fuente(fuente, fps_img, muestreo):
        r = model.predict(frame, conf=conf, verbose=False)[0]
        a = area_union(r.boxes.xyxyn.cpu().numpy()) if len(r.boxes) else 0.0
        n = len(r.boxes)
        c = float(r.boxes.conf.mean()) if n else 0.0
        filas.append(dict(t=t, area=a, n=n, conf=c))
        vis = r.plot()
        e = int(np.digitize(a, UMBRALES))
        cv2.putText(vis, f"Area={a*100:.1f}%  N={n}  Estado={ESTADOS[e]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORES[e], 2)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(OUT / "video_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(1.0, muestreo if Path(fuente).is_file() else fps_img), (w, h))
        writer.write(vis)
        if show:
            cv2.imshow("Examen 2 - YOLO + Simulacion", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    return pd.DataFrame(filas)


# ------------------------- MODELO 1: EDO DE CRECIMIENTO DEL DERRAME -------------------------
def edo(t, q, k, a0):
    """Solucion analitica de dA/dt = q - k*A con A(0)=a0."""
    return q / k + (a0 - q / k) * np.exp(-k * t)


def ajustar_edo(t, a):
    from scipy.optimize import curve_fit
    a0 = a[0]
    popt, _ = curve_fit(lambda tt, q, k: edo(tt, q, k, a0), t, a, p0=[0.01, 0.05],
                        bounds=([0, 1e-6], [1, 5]), maxfev=20000)
    return float(popt[0]), float(popt[1]), float(a0)


def suavizar(a, w=5):
    return pd.Series(a).rolling(w, center=True, min_periods=1).mean().to_numpy()


def ajustar_dos_fases(t, a):
    """EDO por fases: derrame (dA/dt = q - k1*A) hasta el pico de area y limpieza/secado
    (dA/dt = -k2*A) despues. El instante de cambio tc es el maximo de la serie suavizada."""
    from scipy.optimize import curve_fit
    ic = int(np.argmax(suavizar(a)))
    ic = max(3, min(ic, len(t) - 4))
    tc, a0 = float(t[ic]), float(a[0])
    p1, _ = curve_fit(lambda x, q, k: edo(x, q, k, a0), t[: ic + 1], a[: ic + 1],
                      p0=[0.01, 0.05], bounds=([0, 1e-6], [1, 5]), maxfev=20000)
    atc = float(edo(tc, p1[0], p1[1], a0))
    p2, _ = curve_fit(lambda x, k: atc * np.exp(-k * (x - tc)), t[ic:], a[ic:],
                      p0=[0.05], bounds=([1e-6], [5]), maxfev=20000)
    return dict(q=float(p1[0]), k1=float(p1[1]), k2=float(p2[0]), tc=tc, a0=a0)


def predecir_dos_fases(t, p):
    atc = edo(p["tc"], p["q"], p["k1"], p["a0"])
    return np.where(t <= p["tc"], edo(t, p["q"], p["k1"], p["a0"]),
                    atc * np.exp(-p["k2"] * (t - p["tc"])))


# ------------------------- MODELO 2: CADENA DE MARKOV -------------------------
def estimar_P(est, k=3, alpha=0.5):
    c = np.full((k, k), alpha)             # suavizado de Laplace
    for a, b in zip(est[:-1], est[1:]):
        c[a, b] += 1
    return c / c.sum(1, keepdims=True)


def estacionaria(P):
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(abs(w - 1))])
    return pi / pi.sum()


def tiempo_a_critico(P, dt):
    """Tiempo medio (s) hasta el estado Critico (absorbente) desde Seguro y Precaucion."""
    q = P[:-1, :-1]
    return np.linalg.inv(np.eye(len(q)) - q).sum(1) * dt


def ajustar_P(P, f_sub, f_baj):
    """Escenario: f_sub escala la prob. de empeorar (fuga mayor); f_baj la de mejorar (limpieza)."""
    P = P.copy()
    for i in range(len(P)):
        for j in range(len(P)):
            if j > i:
                P[i, j] *= f_sub
            elif j < i:
                P[i, j] *= f_baj
        off = P[i].sum() - P[i, i]
        if off > 0.99:
            P[i] *= 0.99 / off
            P[i, i] = 0.0
            off = P[i].sum()
        P[i, i] = 1 - off
    return P


def simular_cadena(P, s0, pasos, n, rng):
    cum = np.cumsum(P, axis=1)
    hits, tiempos = 0, []
    for _ in range(n):
        s = s0
        for t in range(1, pasos + 1):
            s = int(np.searchsorted(cum[s], rng.random()))
            if s >= 2:
                hits += 1
                tiempos.append(t)
                break
    return hits / n, (float(np.mean(tiempos)) if tiempos else float("nan"))


ESCENARIOS = {
    "Base": (1.0, 1.0),
    "Fuga x1.5": (1.5, 1.0),
    "Fuga x2": (2.0, 1.0),
    "Limpieza x2": (1.0, 2.0),
    "Fuga x2 + limpieza x2": (2.0, 2.0),
}


# ------------------------- FASE IV: SENSIBILIDAD Y VALIDACION -------------------------
def sensibilidad_vision(area, dt, recall, precision, n=300, rng=None):
    """Error del detector -> A_real ~ A_obs * precision / recall -> incertidumbre en pi_critico y T_critico."""
    rng = rng or np.random.default_rng(SEED)
    k = 200
    pis, tcs = [], []
    for _ in range(n):
        r = rng.beta(recall * k + 1, (1 - recall) * k + 1)
        p = rng.beta(precision * k + 1, (1 - precision) * k + 1)
        est = np.digitize(np.clip(area * p / r, 0, 1), UMBRALES)
        P = estimar_P(est)
        pis.append(estacionaria(P)[2])
        tcs.append(tiempo_a_critico(P, dt)[0])
    ic = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return dict(pi_critico_media=float(np.mean(pis)), pi_critico_ic95=ic(pis),
                T_critico_media_s=float(np.mean(tcs)), T_critico_ic95_s=ic(tcs))


def validar(t, a, est, dt, frac=0.7):
    """(1) EDO de dos fases: se ajusta con el primer 70% de CADA fase y se predice el 30% restante,
    comparando contra un predictor ingenuo (media del entrenamiento).
    (2) Markov: distribucion estacionaria del primer 70% vs frecuencias observadas en el 30% final."""
    t0 = t - t[0]
    ic = max(5, min(int(np.argmax(suavizar(a))), len(t) - 6))
    tr, te = [], []
    for lo, hi in [(0, ic + 1), (ic + 1, len(t))]:
        c = lo + int((hi - lo) * frac)
        tr += list(range(lo, c))
        te += list(range(c, hi))
    p = ajustar_dos_fases(t0[tr], a[tr])
    pred, real = predecir_dos_fases(t0[te], p), a[te]
    rmse = float(np.sqrt(np.mean((pred - real) ** 2)))
    rmse_base = float(np.sqrt(np.mean((a[tr].mean() - real) ** 2)))
    smape = float(np.mean(2 * abs(pred - real) / (abs(pred) + abs(real) + 1e-9)) * 100)

    c = int(len(t) * frac)
    pi_mod = estacionaria(estimar_P(est[:c]))
    pi_real = np.bincount(est[c:], minlength=3) / len(est[c:])
    tv = float(0.5 * np.abs(pi_mod - pi_real).sum())
    return dict(EDO_RMSE=rmse, EDO_RMSE_ingenuo=rmse_base, EDO_mejora_vs_ingenuo_pct=(1 - rmse / rmse_base) * 100,
                EDO_sMAPE_pct=smape, Markov_distancia_variacion_total=tv,
                pi_modelo=pi_mod.tolist(), pi_observado=pi_real.tolist())


def graficar(t, a, pred, p, esc_df, sens_df, ruta):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(t, a * 100, "o", ms=3, label="Observado (YOLO)")
    ax[0].plot(t, pred * 100, "-", label=f"EDO 2 fases (q={p['q']:.3f}, k1={p['k1']:.2f}, k2={p['k2']:.2f})")
    ax[0].axvline(p["tc"] + t[0], ls="--", c="gray", lw=1)
    ax[0].axhline(UMBRALES[0] * 100, ls=":", c="orange")
    ax[0].axhline(UMBRALES[1] * 100, ls=":", c="red")
    ax[0].set_xlabel("t (s)")
    ax[0].set_ylabel("Area de derrame (%)")
    ax[0].set_title("Modelo 1: EDO por fases (derrame / limpieza)")
    ax[0].legend()
    ax[1].bar(esc_df["escenario"], esc_df["P_critico"])
    ax[1].set_ylabel("P(llegar a Critico)")
    ax[1].set_title("Escenarios (Markov + Monte Carlo)")
    ax[1].tick_params(axis="x", rotation=30)
    ax[2].plot(sens_df["recall"], sens_df["pi_critico"], "o-")
    ax[2].set_xlabel("Recall del detector")
    ax[2].set_ylabel("Tiempo en Critico (estacionaria)")
    ax[2].set_title("Sensibilidad a la vision")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


# ------------------------- EJECUCION COMPLETA (Fases III y IV) -------------------------
def ejecutar(a):
    rng = np.random.default_rng(SEED)
    df = extraer_variables(a.weights, a.fuente, a.fps_img, a.muestreo, show=a.show)
    if len(df) < 15:
        raise SystemExit(f"Solo {len(df)} muestras (<15). Usa un video mas largo, sube --muestreo (p.ej. 5) "
                         "o usa una carpeta de imagenes en secuencia.")
    df.to_csv(OUT / "serie_vision.csv", index=False)
    t, area = df["t"].to_numpy(), df["area"].to_numpy()
    if a.suavizado > 1:   # filtro de media movil: reduce el parpadeo de las detecciones frame a frame
        area = suavizar(area, a.suavizado)
    dt = float(np.median(np.diff(t)))
    est = np.digitize(area, UMBRALES)
    print(f"\nVariables extraidas: {len(df)} muestras | area media={area.mean()*100:.1f}% | N medio={df['n'].mean():.2f}")

    # Modelo 1
    pf = ajustar_dos_fases(t - t[0], area)
    pred = predecir_dos_fases(t - t[0], pf)
    eq = pf["q"] / pf["k1"] * 100
    fase1 = (f"dA/dt = {pf['q']:.4f} - {pf['k1']:.3f}*A (equilibrio {eq:.1f}%)" if eq <= 100
             else f"crecimiento ~lineal dA/dt = {pf['q']:.4f} (k1 no identificable: fase corta)")
    print(f"EDO fase 1 (derrame): {fase1} | cambio de fase en t={pf['tc']:.1f}s "
          f"| fase 2 (limpieza): dA/dt = -{pf['k2']:.3f}*A")

    # Modelo 2
    P = estimar_P(est)
    pi = estacionaria(P)
    tc = tiempo_a_critico(P, dt)
    print("Matriz P:\n", P.round(3), "\nEstacionaria:", dict(zip(ESTADOS, pi.round(3))),
          f"\nTiempo medio a Critico desde Seguro: {tc[0]:.1f}s")

    # Escenarios
    pasos = int(a.horizonte / dt)
    filas = []
    for nombre, (fs, fb) in ESCENARIOS.items():
        pc, tm = simular_cadena(ajustar_P(P, fs, fb), int(est[0]), pasos, 2000, rng)
        filas.append(dict(escenario=nombre, P_critico=pc, tiempo_medio_s=tm * dt))
    esc = pd.DataFrame(filas)
    esc.to_csv(OUT / "escenarios.csv", index=False)
    print(f"\nEscenarios (horizonte {a.horizonte}s):\n", esc.round(3).to_string(index=False))

    # Sensibilidad
    mc = sensibilidad_vision(area, dt, a.recall, a.precision, rng=rng)
    sens = pd.DataFrame([dict(recall=r, pi_critico=estacionaria(
        estimar_P(np.digitize(np.clip(area / r, 0, 1), UMBRALES)))[2]) for r in [0.6, 0.7, 0.8, 0.9, 1.0]])
    print("\nIncertidumbre por vision:", mc)

    # Validacion
    val = validar(t, area, est, dt)
    print("Validacion (70/30):", val)

    graficar(t, area, pred, pf, esc, sens, OUT / "resultados.png")
    (OUT / "resumen.json").write_text(json.dumps(dict(
        edo_dos_fases=pf, P=P.tolist(), estacionaria=pi.tolist(),
        T_critico_desde_seguro_s=float(tc[0]), incertidumbre_vision=mc, validacion=val), indent=2))
    print(f"\nListo. Revisa '{OUT}/' (video_anotado.mp4, serie_vision.csv, escenarios.csv, resultados.png, resumen.json)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("prepare", help="Fase I: ndjson -> carpetas YOLO + data.yaml")
    p0.add_argument("--ndjson", required=True)
    p0.add_argument("--destino", default="dataset")
    p1 = sub.add_parser("train", help="Fase II: entrenar y evaluar")
    p1.add_argument("--data", required=True)
    p1.add_argument("--device", default=None)
    p2 = sub.add_parser("run", help="Fases III y IV")
    p2.add_argument("--weights", required=True)
    p2.add_argument("--fuente", required=True, help="video .mp4 o carpeta con secuencia de imagenes")
    p2.add_argument("--fps-img", type=float, default=1.0, help="imagenes por segundo si la fuente es carpeta")
    p2.add_argument("--muestreo", type=float, default=5.0, help="muestras por segundo si es video")
    p2.add_argument("--horizonte", type=float, default=15.0, help="horizonte de simulacion (s)")
    p2.add_argument("--recall", type=float, default=0.9)
    p2.add_argument("--precision", type=float, default=0.9)
    p2.add_argument("--suavizado", type=int, default=5, help="ventana de media movil sobre A(t) (1 = sin filtro)")
    p2.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.cmd == "prepare":
        preparar(a.ndjson, a.destino)
    elif a.cmd == "train":
        entrenar(a.data, a.device)
    else:
        ejecutar(a)