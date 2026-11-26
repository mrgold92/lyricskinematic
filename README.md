# videolyrics

Genera un videolyrics de **tipografía cinética** (texto animado sobre fondo liso) a partir de una canción y su letra.

Cada palabra entra en el momento en que se canta. Cada verso cambia de composición (párrafo, palabras apiladas, escalera diagonal, una palabra enorme cada vez) y de animación de entrada y salida. Las palabras clave salen más grandes y en color de acento.

## Instalación

Necesitas Python 3.10 o superior. ffmpeg no hace falta instalarlo, viene incluido con `imageio-ffmpeg`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

(En Linux/macOS el intérprete está en `.venv/bin/python`.)

La primera vez que se transcribe, `faster-whisper` descarga el modelo de Whisper desde Hugging Face. `small` son unos 460 MB y se guarda en caché, así que solo se descarga una vez. No hace falta cuenta.

## Uso

```bash
.venv/Scripts/python main.py "mi canción.mp3" --lyrics letra.txt --language es \
    --title "Título" --artist "Tu nombre"
```

- `letra.txt` lleva un verso por línea. Las líneas en blanco se ignoran.
- El vídeo se guarda en `output/<nombre de la canción>.mp4`, o donde indiques con `-o`.

### Flujo recomendado

1. **Genera los tiempos** y revísalos sin renderizar:
   ```bash
   .venv/Scripts/python main.py "mi canción.mp3" --lyrics letra.txt --language es --lyrics-only
   ```
   Los tiempos quedan en `output/<canción>/lyrics.json`, un verso por línea: `["palabra", inicio, fin]`. Si alguna palabra entra desfasada, corrige el tiempo ahí.
2. **Prueba un fragmento** o un frame suelto:
   ```bash
   .venv/Scripts/python main.py "mi canción.mp3" --start 55 --duration 20 --res 720p
   .venv/Scripts/python main.py "mi canción.mp3" --frame 62.5
   ```
3. **Renderiza el vídeo completo.** Con la caché de `lyrics.json`, no repite la transcripción.

### Opciones

| Opción | Qué hace |
|---|---|
| `--lyrics FILE` | Letra real; se alinea con lo que Whisper oye (mucho mejor que sin letra) |
| `--language es` | Idioma del audio; si no lo indicas, se autodetecta |
| `--model small` | Modelo Whisper: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--title`, `--artist` | Tarjeta de título al inicio (por defecto, el nombre del archivo) |
| `--accent "#FFD60A"` | Color de acento: palabras clave y golpe de entrada |
| `--bg`, `--fg` | Colores de fondo y de texto |
| `--reveal auto` | Cómo aparece cada verso (ver más abajo): `auto`, `word`, `chunk` o `line` |
| `--line-wps 2.4` | En `auto`, palabras por segundo a partir de las cuales el verso se muestra entero |
| `--transitions all` | Transiciones entre versos, separadas por comas: `push_l`, `push_u`, `wheel`, `fade`, `flip`, `zoom`, `spin` |
| `--upper` | Todo en mayúsculas |
| `--beat` | El texto pulsa al ritmo (analiza el audio, algo más lento) |
| `--font FILE` | Fuente `.ttf`/`.otf` (por defecto Segoe UI Black si existe) |
| `--seed N` | Cambia el reparto de composiciones y animaciones |
| `--res` | `480p`, `720p`, `1080p` (por defecto) o `1440p` |
| `--fps` | Fotogramas por segundo (30 por defecto) |
| `--start`, `--duration` | Renderiza solo un fragmento |
| `--frame SEG` | Guarda un PNG de ese segundo y termina |
| `--workers N` | Procesos en paralelo (8 por defecto; cada uno usa ~0,8 GB a 1080p) |

### Legibilidad y transiciones

En canciones rápidas, leer palabra a palabra es imposible. Por eso cada verso se revela de una forma según su velocidad (`--reveal auto`):

| Modo | Qué se ve | Se usa cuando |
|---|---|---|
| `line` | El verso entero desde el principio; las palabras sin cantar están tenues y se iluminan al sonar | Rápido (≥ 2,4 palabras/s, ajustable con `--line-wps`) |
| `chunk` | Aparecen grupos de 2–3 palabras un poco antes de cantarse | Intermedio (≥ 1,6 palabras/s) |
| `word` | Cada palabra aparece cuando se canta, con su animación | Lento |

La composición "una palabra enorme cada vez" solo se usa en versos lentos y cortos.

Entre un verso y el siguiente hay una transición en la que uno sale mientras entra el otro. Puedes restringirlas con `--transitions`, por ejemplo `--transitions wheel,push_l,zoom`:

- `push_l`, `push_u`: el verso sale empujado y el siguiente entra por el lado opuesto.
- `wheel`: giran como un carrusel.
- `spin`: el verso gira y se encoge; el siguiente entra girando.
- `zoom`: la cámara atraviesa el verso y el siguiente llega de lejos.
- `flip`: el verso se voltea como una tarjeta. Solo se usa entre versos separados por una pausa.
- `fade`: fundido cruzado.

### Opciones para repetir pasos

| Opción | Qué hace |
|---|---|
| `--realign` | Rehace `lyrics.json` con la transcripción en caché, sin ejecutar Whisper |
| `--fill-gaps` | Con la transcripción en caché, reintenta los tramos de letra que Whisper omitió |
| `--retranscribe` | Vuelve a ejecutar Whisper desde cero |

## Cómo funciona

1. **Transcripción** (`lyrics.py`): `faster-whisper` en CPU da el tiempo de cada palabra.
2. **Alineación**: tu letra se empareja con lo que Whisper oyó. Whisper suele omitir tramos enteros de las canciones; el programa detecta esos huecos y los vuelve a transcribir usando la letra esperada como pista.
3. **Escena** (`scene.py`): cada frame es una función pura del tiempo, dibujada con Skia.
4. **Render** (`render.py`): trozos de 8 s en paralelo, cada uno codificado por su propio ffmpeg. Después se unen sin recodificar y se añade el audio.

## Si algo no sale bien

- **Una parte va desfasada.** Mira la sección de `lyrics.json` y corrige los tiempos a mano. Si Whisper no reconoció ese tramo, el programa avisa con `AVISO: N palabras sin reconocer entre X s y Y s`.
- **Poco reconocimiento.** Prueba `--model medium`. Es más preciso con voz cantada, pero más lento en CPU y pesa unos 1,5 GB.
- **`Invalid argument` al renderizar.** Suele ser falta de memoria. Baja `--workers` o la resolución.
- **Windows bloquea una DLL de scipy** (Control de aplicaciones) al importar librosa. Solo afecta a `--beat`; si pasa, vuelve a ejecutar el comando.
