# Motor de Recomendación Acústica

Autore(s):

- Briones Sánchez Erick Alan
- Esteva Gallegos Carlos Fabián
- Munive Ramírez Ibrahim

## Descripción
 
Recomienda pistas musicales a partir de su firma acústica, sin usar género ni artista como criterio.

Describe cada pista con características espectrales calculadas con librosa. 

Compara clasificadores supervisados y usa una red de Hopfield para recuperar pistas similares.
 
## Dataset
 
Subconjunto fma_small de Free Music Archive: 8\,000 pistas de 30 s en 8 géneros balanceados.
Usa las características precalculadas de features.csv y los metadatos de tracks.csv.

## Modelos
 
Clasificación: perceptrón, k-NN con similitud coseno y regresión logística.

Recuperación: red de Hopfield como memoria asociativa sobre patrones binarizados.
 
## Instalación
 
Instala las dependencias con `pip install -r requirements.txt`.

Descarga fma_metadata.zip y fma_small.zip desde el repositorio de FMA y descomprímelos en una carpeta llamada `data/`.
 
## Ejecución

Ejecuta `python recoa.py` desde la raíz del proyecto.
recoa.py llama en orden a los scripts de cada etapa del pipeline.
 
## Estructura
 
El proyecto se organiza en un script por etapa del pipeline, y `recoa.py` los ejecuta.
 
| Archivo |
| --- |
| `recoa.py` |
| `eda.py` |
| `preprocesamiento.py` |
| `modelos.py` |
| `evaluacion.py` |
| `recomendacion.py` |

$$\tag*{$\blacksquare$}{ }$$
