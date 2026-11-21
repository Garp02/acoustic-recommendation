import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

from pathlib import Path

data = 'data/fma_metadata/'
audio = 'data/fma_small/'
small = 'small'
frecuencia = 22050
duracion = 30


def cargar_tracks(ruta: str = data) -> pd.DataFrame:
    """Lee solo tracks.csv, que es mucho más ligero que features.csv."""
    df = pd.read_csv(ruta + 'tracks.csv', index_col = 0, header = [0, 1])
    
    return df 

def cargar_datos(ruta: str = data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee tracks.csv y features.csv desde la ruta indicada."""
    tracks = cargar_tracks(ruta)
    features = pd.read_csv(ruta + 'features.csv', index_col=0, header=[0, 1, 2])
    
    return tracks, features

def filtrar_subconjunto(tracks: pd.DataFrame, features: pd.DataFrame, subset: str = small,) -> tuple:
    """Separa las características y los géneros de las pistas del subconjunto."""
    mascara = tracks['set']['subset'] == subset
    X = features.loc[mascara]
    y = tracks.loc[mascara, ('track', 'genre_top')]

    return X, y

def resumen(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Imprime dimensiones, nulos y distribución de géneros; devuelve estadísticas descriptivas."""
    print('Dimensiones:', X.shape)
    print('Nulos:', X.isna().sum().sum())
    print(y.value_counts())

    return X.describe().T

def ruta_audio(track_id: int, carpeta: str = audio) -> Path:
    """Devuelve la ruta del MP3 de una pista según la estructura de carpetas de FMA."""
    nombre = f'{track_id:06d}'
    return Path(carpeta) / nombre[:3] / f'{nombre}.mp3'

def muestrear_ids(tracks: pd.DataFrame, n: int = 5, seed: int = 42, subset: str = small) -> list[int]:
    """Elige n identificadores aleatorios del subconjunto con semilla fija."""
    mascara = tracks['set']['subset'] == subset

    return tracks.loc[mascara].sample(n, random_state=seed).index.tolist()

def analizar_audio(ruta: Path, frecuencia: int = frecuencia, duracion: float = duracion) -> dict[str, float]:
    """Calcula estadísticas espectrales, temporales y de ritmo de un archivo de audio."""
    y, sr = librosa.load(ruta, sr = frecuencia, mono = True, duration = duracion)
    centroide = librosa.feature.spectral_centroid(y = y, sr = sr)[0]
    ancho = librosa.feature.spectral_bandwidth(y = y, sr = sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y = y)[0]
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13).mean(axis = 1)
    tempo, _ = librosa.beat.beat_track(y = y, sr = sr)

    resultado = {
        'duracion_s': float(librosa.get_duration(y = y, sr = sr)),
        'pico': float(np.max(np.abs(y))),
        'rms_medio': float(rms.mean()),
        'centroide_medio': float(centroide.mean()),
        'ancho_banda_medio': float(ancho.mean()),
        'rolloff_medio': float(rolloff.mean()),
        'zcr_medio': float(zcr.mean()),
        'tempo_bpm': float(np.atleast_1d(tempo)[0]),
    }

    for i, valor in enumerate(mfcc, start = 1): resultado[f'mfcc_{i}'] = float(valor)
    
    return resultado

def analizar_muestra(ids: list[int], carpeta: str = audio) -> pd.DataFrame:
    """Aplica analizar_audio a varias pistas y omite las que no existen en disco."""
    filas = {}

    for track_id in ids:
    
        ruta = ruta_audio(track_id, carpeta)
    
        if ruta.exists(): filas[track_id] = analizar_audio(ruta)
    
        else: print(f'No encontrado: {ruta}')
    
    return pd.DataFrame.from_dict(filas, orient = 'index')

def graficar_pista(ruta: Path, salida: Path, frecuencia: int = frecuencia, duracion: float = duracion) -> None:
    """Guarda una figura con la forma de onda, la energía RMS y el espectrograma de mel de una pista."""
    y, sr = librosa.load(ruta, sr = frecuencia, mono = True, duration = duracion)
    mel = librosa.feature.melspectrogram(y = y, sr = sr)
    mel_db = librosa.power_to_db(mel, ref = np.max)
    rms = librosa.feature.rms(y = y)[0]
    tiempos = librosa.times_like(rms, sr = sr)

    fig, ejes = plt.subplots(2, 1, figsize = (10, 7), sharex = True)
    fig.suptitle(f'Pista {ruta.stem} | muestreo {sr} Hz | {duracion} s', fontsize = 13)

    librosa.display.waveshow(y, sr = sr, ax = ejes[0], color = 'steelblue', alpha = 0.6)
    ejes[0].plot(tiempos, rms, color = 'crimson', linewidth = 1.5, label = 'Energía RMS')
    ejes[0].set(title = 'Forma de onda y energía', ylabel = 'Amplitud')
    ejes[0].legend(loc = 'upper right')
    ejes[0].grid(alpha = 0.3)

    img = librosa.display.specshow(mel_db, sr = sr, x_axis = 'time', y_axis = 'mel', ax = ejes[1], cmap = 'magma')
    cbar = fig.colorbar(img, ax = ejes[1], format = '%+2.0f dB')
    cbar.set_label('Intensidad (dB respecto al máximo)')
    ejes[1].set(title = 'Espectrograma de mel', xlabel = 'Tiempo (s)', ylabel = 'Frecuencia (Hz, escala mel)')

    fig.tight_layout(rect = [0, 0, 1, 0.96])
    salida.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(salida, dpi = 120)
    plt.close(fig)

def main() -> pd.DataFrame:

    """Analiza una muestra pequeña de pistas y guarda la gráfica de la primera."""
    tracks = cargar_tracks()
    ids = muestrear_ids(tracks, n = 5)
    tabla = analizar_muestra(ids)
    print(tabla.T)
    graficar_pista(ruta_audio(ids[0]), Path('img') / f'{ids[0]:06d}.png')
    return tabla

if __name__ == '__main__':
    
    main()