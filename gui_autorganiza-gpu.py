# ==============================================================================
# AUTORGANIZA IA V3.4 - CÓDIGO COMENTADO
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORTACIONES DE LIBRERÍAS
# ------------------------------------------------------------------------------
# Librerías estándar de Python para interactuar con el sistema operativo y ficheros.
import os
import shutil
import json
import threading
import queue

# Librerías de terceros que deben ser instaladas (pip install ...).
from PIL import Image, UnidentifiedImageError  # Para abrir y procesar imágenes.
import torch                                   # PyTorch, el motor principal de la IA.
from transformers import AutoProcessor, AutoModel # Hugging Face, para descargar y usar los modelos de IA.

# Librería de Python para crear la interfaz gráfica de usuario (GUI).
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext

# ------------------------------------------------------------------------------
# 2. LÓGICA DE PROCESAMIENTO (FUNCIONES INDEPENDIENTES DE LA GUI)
# ------------------------------------------------------------------------------

# Esta función elige un tamaño de lote óptimo basado en la VRAM de la GPU.
# Procesar imágenes en lotes (batching) es mucho más eficiente en una GPU.
def determinar_batch_size_automatico(model_id, log_func):
    """
    Determina un tamaño de lote apropiado basado en la VRAM de la GPU y el modelo.
    Devuelve un número entero que representa cuántas imágenes procesar a la vez.
    """
    # Si no hay GPU disponible (CUDA), usa un valor bajo para no sobrecargar la RAM.
    if not torch.cuda.is_available():
        return 4
    try:
        # Obtiene la memoria de la GPU en Gigabytes.
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log_func(f"VRAM total detectada: {total_vram_gb:.2f} GB")
        
        # Lógica para elegir el tamaño del lote: los modelos más grandes necesitan lotes más pequeños.
        if 'large' in model_id.lower() or 'H-14' in model_id:
            log_func("Modelo grande/enorme detectado. Ajustando batch size.")
            if total_vram_gb < 8: return 4
            if total_vram_gb < 12: return 8
            if total_vram_gb < 24: return 16
            return 32
        else: # Modelos 'base' o más pequeños.
            log_func("Modelo base detectado. Usando batch sizes más grandes.")
            if total_vram_gb < 6: return 8
            if total_vram_gb < 12: return 16
            if total_vram_gb < 24: return 32
            return 64
    except Exception as e:
        # Si falla la detección, usa un valor seguro por defecto.
        log_func(f"No se pudo determinar la VRAM, usando batch size por defecto. Error: {e}")
        return 8

# Esta es la función principal que se ejecuta en un hilo separado para no congelar la GUI.
# Realiza todo el trabajo pesado: cargar el modelo, analizar imágenes y mover ficheros.
def organizar_con_ia_worker(carpeta_principal, modelo_id, log_queue, stop_event, batch_size, config_dict, umbral_confianza):
    """
    Función que realiza la clasificación y movimiento de imágenes.
    Está diseñada para ser ejecutada en un hilo secundario.
    """
    # Función interna para enviar mensajes (logs) a la GUI de forma segura.
    def log(message):
        if not stop_event.is_set(): log_queue.put(message)

    try:
        # --- A. Preparación Inicial ---
        log("Iniciando la organización con IA...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Usando el dispositivo: {device.upper()}")
        if device == "cuda":
            log(f"Batch size: {batch_size}")
        else:
            log("\nADVERTENCIA: No se detecta GPU. El proceso será MUY LENTO.")
            batch_size = 1 # En CPU, procesar de una en una es más seguro.
            log(f"Batch size en CPU forzado a: {batch_size}")

        # --- B. Carga del Modelo de IA ---
        log(f"Cargando el modelo '{modelo_id}'...")
        try:
            # Descarga (si es necesario) y carga el modelo y su procesador en la memoria (RAM o VRAM).
            modelo = AutoModel.from_pretrained(modelo_id).to(device)
            procesador = AutoProcessor.from_pretrained(modelo_id)
            log("Modelo cargado con éxito")
        except Exception as e:
            log(f"ERROR al cargar modelo '{modelo_id}'. {e}")
            log_queue.put("DONE"); return

        # --- C. Preparación de Carpetas y Ficheros ---
        # Lee las descripciones y nombres de carpetas desde el perfil JSON.
        etiquetas_para_ia = list(config_dict.keys())
        carpetas_destino = list(config_dict.values()) + ['revisar']
        # Crea las carpetas de destino si no existen.
        for nombre_carpeta in carpetas_destino:
            ruta_carpeta = os.path.join(carpeta_principal, nombre_carpeta)
            if not os.path.exists(ruta_carpeta): os.makedirs(ruta_carpeta)

        # Prepara un diccionario para contar cuántos archivos se mueven a cada carpeta.
        contador_movimientos = {nombre_carpeta: 0 for nombre_carpeta in carpetas_destino}
        # Encuentra todas las imágenes en la carpeta de origen.
        archivos_imagen = [f for f in os.listdir(carpeta_principal) if os.path.isfile(os.path.join(carpeta_principal, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        total_imagenes = len(archivos_imagen)
        log(f"\nSe encontraron {total_imagenes} imágenes para procesar.")

        # Divide la lista total de imágenes en lotes más pequeños.
        lotes = [archivos_imagen[i:i + batch_size] for i in range(0, len(archivos_imagen), batch_size)]
        
        # --- D. Bucle Principal de Procesamiento ---
        total_procesadas = 0
        for num_lote, lote_archivos in enumerate(lotes):
            # Comprueba si el usuario ha pulsado "Cancelar" antes de cada lote.
            if stop_event.is_set(): break
            log(f"\n--- Procesando Lote {num_lote + 1}/{len(lotes)} ---")
            
            # Carga las imágenes del lote actual, omitiendo las corruptas.
            imagenes_lote, rutas_lote = [], []
            for nombre_archivo in lote_archivos:
                ruta_completa = os.path.join(carpeta_principal, nombre_archivo)
                try:
                    img = Image.open(ruta_completa); img.load() # .load() fuerza la lectura para detectar errores.
                    imagenes_lote.append(img.convert("RGB")); rutas_lote.append(ruta_completa)
                except Exception as e:
                    log(f"  -> ERROR: No se pudo leer '{nombre_archivo}'. Moviendo a 'revisar'. ({e})")
                    shutil.move(ruta_completa, os.path.join(carpeta_principal, 'revisar')); contador_movimientos['revisar'] += 1

            if not imagenes_lote: log("  -> Ninguna imagen válida en este lote."); continue
            
            # Procesa el lote de imágenes con la IA.
            inputs = procesador(text=etiquetas_para_ia, images=imagenes_lote, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad(): # Desactiva el cálculo de gradientes para acelerar.
                outputs = modelo(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1) # Convierte los resultados en probabilidades (0 a 1).
            
            # Itera sobre los resultados y mueve cada fichero.
            for i, ruta_completa_archivo in enumerate(rutas_lote):
                if stop_event.is_set(): break
                nombre_archivo_actual = os.path.basename(ruta_completa_archivo)
                prob_imagen = probs[i]
                mejor_prob, mejor_indice = prob_imagen.max().item(), prob_imagen.argmax().item()

                # La decisión clave: si la confianza de la IA supera nuestro umbral, se mueve.
                if mejor_prob >= umbral_confianza:
                    etiqueta_ganadora = etiquetas_para_ia[mejor_indice]
                    nombre_carpeta_destino = config_dict[etiqueta_ganadora]
                    destino = os.path.join(carpeta_principal, nombre_carpeta_destino, nombre_archivo_actual)
                    try:
                        shutil.move(ruta_completa_archivo, destino)
                        log(f"  Procesado: '{nombre_archivo_actual}' -> '{nombre_carpeta_destino}' ({mejor_prob:.2%})")
                        contador_movimientos[nombre_carpeta_destino] += 1
                    except shutil.Error as e:
                        log(f"  -> ERROR al mover '{nombre_archivo_actual}'. Moviendo a 'revisar'. {e}")
                        shutil.move(ruta_completa_archivo, os.path.join(carpeta_principal, 'revisar', nombre_archivo_actual)); contador_movimientos['revisar'] += 1
                else: # Si la confianza es demasiado baja, se mueve a 'revisar'.
                    log(f"  Procesado: '{nombre_archivo_actual}' -> 'revisar' (Confianza {mejor_prob:.2%} < {umbral_confianza:.2%})")
                    shutil.move(ruta_completa_archivo, os.path.join(carpeta_principal, 'revisar', nombre_archivo_actual)); contador_movimientos['revisar'] += 1

            # Actualiza la barra de progreso en la GUI.
            total_procesadas += len(imagenes_lote)
            progreso = (total_procesadas / total_imagenes) * 100 if total_imagenes > 0 else 0
            log_queue.put(("PROGRESS", progreso))
        
        # --- E. Finalización y Resumen ---
        if stop_event.is_set(): log("\n--- Proceso cancelado por el usuario ---")
        else:
            log("\n--- Resumen de la Organización ---")
            total_movidos = sum(contador_movimientos.values())
            log(f"Total de archivos procesados: {total_movidos}\n")
            for carpeta, cantidad in sorted(contador_movimientos.items()):
                if cantidad > 0: log(f"  - {carpeta}: {cantidad} archivo(s)")
            log("\n¡Organización completada!")
    
    except Exception as e:
        log(f"\n\nERROR CRÍTICO: {e}")
    finally:
        # Envía una señal a la GUI para indicar que el hilo ha terminado.
        log_queue.put("DONE")


# ------------------------------------------------------------------------------
# 3. CLASE DE LA APLICACIÓN GUI (INTERFAZ GRÁFICA)
# ------------------------------------------------------------------------------

class App:
    # El método constructor, se ejecuta al crear la aplicación. Define toda la interfaz.
    def __init__(self, root):
        self.root = root
        self.root.title("Autorganiza IA v3.3 (Umbral Dinámico)")
        self.root.geometry("850x600")
        self.root.minsize(750, 500)

        # --- A. Variables de Estado de la GUI ---
        # Se usan para almacenar las selecciones del usuario (rutas, nombres de modelo, etc.).
        self.config_file_path = tk.StringVar()
        self.folder_path = tk.StringVar()
        self.selected_config_filename = tk.StringVar()
        self.selected_model_name = tk.StringVar()
        self.selected_threshold = tk.StringVar()
        
        # Diccionario que asocia el nombre legible del modelo con su identificador técnico.
        self.model_options = {
            "SigLIP Base (Google, base)": "google/siglip-base-patch16-224",
            "SigLIP Grande (Google, mas preciso)": "google/siglip-large-patch16-384",
            "CLIP Base (OpenAI)": "openai/clip-vit-base-patch32",
            "CLIP Grande (OpenAI)": "openai/clip-vit-large-patch14",
            "OpenCLIP Enorme (LAION, Máx. Precisión)": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        }

        # --- B. Variables para la Comunicación entre Hilos ---
        self.thread = None  # Almacenará el hilo de procesamiento.
        self.stop_event = threading.Event()  # Señal para detener el hilo.
        self.log_queue = queue.Queue()  # Cola para recibir mensajes del hilo.
        self.progress_var = tk.DoubleVar() # Variable para la barra de progreso.

        # --- C. Construcción de la Interfaz Gráfica ---
        # Se usa un layout de rejilla (grid) para que la ventana sea redimensionable.
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        main_frame.columnconfigure(0, weight=1); main_frame.columnconfigure(1, weight=1); main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(3, weight=1) # La fila 3 (log) se expandirá verticalmente.

        # Fila de Configuración Superior (3 menús desplegables)
        config_frame = ttk.LabelFrame(main_frame, text="1. Perfil", padding="10")
        config_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        config_frame.columnconfigure(0, weight=1)
        self.config_combobox = ttk.Combobox(config_frame, textvariable=self.selected_config_filename, state="readonly")
        self.config_combobox.grid(row=0, column=0, sticky="ew")
        self.config_combobox.bind("<<ComboboxSelected>>", self.on_config_selected)

        model_frame = ttk.LabelFrame(main_frame, text="2. Modelo IA", padding="10")
        model_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        model_frame.columnconfigure(0, weight=1)
        self.model_combobox = ttk.Combobox(model_frame, textvariable=self.selected_model_name, state="readonly")
        self.model_combobox['values'] = list(self.model_options.keys())
        self.model_combobox.current(0)
        self.model_combobox.grid(row=0, column=0, sticky="ew")

        threshold_frame = ttk.LabelFrame(main_frame, text="3. Umbral Confianza", padding="10")
        threshold_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0), pady=5)
        threshold_frame.columnconfigure(0, weight=1)
        self.threshold_combobox = ttk.Combobox(threshold_frame, textvariable=self.selected_threshold, state="readonly")
        self.threshold_combobox['values'] = [f"{i/10:.1f}" for i in range(1, 10)]
        self.threshold_combobox.set("0.6")
        self.threshold_combobox.grid(row=0, column=0, sticky="ew")

        # Selección de Carpeta
        folder_frame = ttk.LabelFrame(main_frame, text="4. Selecciona la Carpeta a Organizar", padding="10")
        folder_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=5)
        folder_frame.columnconfigure(0, weight=1)
        self.folder_label = ttk.Label(folder_frame, textvariable=self.folder_path)
        self.folder_label.grid(row=0, column=0, sticky="ew", padx=(0,5))
        self.browse_button = ttk.Button(folder_frame, text="Explorar Carpeta...", command=self.browse_folder)
        self.browse_button.grid(row=0, column=1, sticky="e")

        # Botones de Acción (Iniciar/Cancelar)
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        button_frame.columnconfigure(0, weight=1); button_frame.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(button_frame, text="Iniciar Organización", command=self.start_organization, state="disabled")
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0,5))
        self.cancel_button = ttk.Button(button_frame, text="Cancelar", command=self.cancel_organization, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="ew", padx=(5,0))

        # Área de Registro de Actividad
        log_frame = ttk.LabelFrame(main_frame, text="Registro de Actividad", padding="10")
        log_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=5)
        log_frame.rowconfigure(0, weight=1); log_frame.columnconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state="disabled", bg="black", fg="white")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # Barra de Estado y Progreso
        status_frame = ttk.Frame(main_frame, padding=(0, 5))
        status_frame.grid(row=4, column=0, columnspan=3, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.status_bar = ttk.Label(status_frame, text="Listo. Por favor, elige un perfil y una carpeta.", anchor='w')
        self.status_bar.grid(row=0, column=0, sticky="ew")
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=1, sticky="ew")

        # --- D. Inicialización de la Lógica de la App ---
        self.populate_config_dropdown()
        self.mostrar_instrucciones_iniciales()
    
    # --- E. Métodos de la Aplicación (Manejo de Eventos y Lógica) ---
    
    # Inicia el proceso de organización en un hilo separado.
    def start_organization(self):
        try:
            with open(self.config_file_path.get(), 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        except Exception as e:
            self.log_message(f"ERROR: No se pudo leer el fichero de configuración: {e}")
            return

        self.set_ui_state(is_running=True)
        self.log_text.config(state="normal"); self.log_text.delete(1.0, tk.END); self.log_text.config(state="disabled")
        self.progress_var.set(0)
        while not self.log_queue.empty(): self.log_queue.get()
        self.stop_event.clear()

        # Recoge todas las configuraciones de la GUI.
        selected_model_key = self.selected_model_name.get()
        model_id = self.model_options[selected_model_key]
        umbral_confianza = float(self.selected_threshold.get())
        dynamic_batch_size = determinar_batch_size_automatico(model_id, self.log_message)
        
        # Crea y lanza el hilo secundario, pasándole todos los parámetros.
        self.thread = threading.Thread(
            target=organizar_con_ia_worker,
            args=(self.folder_path.get(), model_id, self.log_queue, self.stop_event, dynamic_batch_size, config_dict, umbral_confianza)
        )
        self.thread.start()
        # Inicia el chequeo de la cola de mensajes.
        self.root.after(100, self.process_log_queue)

    # Activa o desactiva los controles de la GUI.
    def set_ui_state(self, is_running):
        state = "disabled" if is_running else "normal"
        readonly_state = "disabled" if is_running else "readonly"

        self.browse_button.config(state=state)
        self.config_combobox.config(state=readonly_state)
        self.model_combobox.config(state=readonly_state)
        self.threshold_combobox.config(state=readonly_state)
        self.start_button.config(state="disabled") # Siempre se desactiva aquí.
        self.cancel_button.config(state="normal" if is_running else "disabled")
        
        if is_running:
            self.status_bar.config(text="Procesando... por favor, espera.")
        else:
            # Se vuelve a activar el botón de inicio solo si las condiciones se cumplen.
            self.check_and_enable_start()

    # Busca ficheros .json y los añade al menú desplegable.
    def populate_config_dropdown(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_files = [f for f in os.listdir(script_dir) if f.lower().endswith('.json')]
            if json_files:
                self.config_combobox['values'] = json_files
            else:
                self.status_bar.config(text="No se encontraron perfiles .json.")
        except Exception as e:
            self.status_bar.config(text=f"Error al buscar perfiles: {e}")

    # Se ejecuta cuando el usuario selecciona un perfil del menú.
    def on_config_selected(self, event=None):
        filename = self.selected_config_filename.get()
        if filename:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, filename)
            self.config_file_path.set(full_path)
            self.status_bar.config(text=f"Perfil: {filename}")
            self.check_and_enable_start()

    # Muestra las instrucciones iniciales en el área de log.
    def mostrar_instrucciones_iniciales(self):
        instrucciones = """Bienvenido a Autorganiza v3.3
Este programa identifica y mueve las imágenes de la carpeta que indiques
 en subcarpetas que creará en la esa misma.
El listado de subcarpetas se definen en un fichero .json en el que
 está la explicación de la imagen a mover y el nombre de la carpeta, para cada una.
Puedes tener varios ficheros json y utilizar el que necesites cada vez. 

Sigue estos 4 pasos para configurar tu organización:

1. Elige tu Perfil: Define las categorías a buscar en un ficheri json.
2. Selecciona el Modelo IA: Elige el "cerebro" que analizará las imágenes.
3. Ajusta el Umbral: La confianza mínima para clasificar y mover (0.6 es equilibrado).
4. Explora y selecciona tu carpeta de imágenes.

El botón "Iniciar" se activará cuando todo esté listo.
"""
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, instrucciones)
        self.log_text.config(state="disabled")

    # Envía un mensaje a la cola para ser procesado por la GUI.
    def log_message(self, message):
        self.log_queue.put(message)

    # Comprueba si se ha seleccionado un perfil y una carpeta para activar el botón "Iniciar".
    def check_and_enable_start(self):
        if self.config_file_path.get() and self.folder_path.get():
            self.start_button.config(state="normal")
        else:
            self.start_button.config(state="disabled")

    # Abre el diálogo para seleccionar una carpeta.
    def browse_folder(self):
        path = filedialog.askdirectory(title="Selecciona la carpeta con imágenes")
        if path:
            self.folder_path.set(path)
            self.status_bar.config(text=f"Carpeta: {path}")
            self.check_and_enable_start()

    # Envía la señal de cancelación al hilo de procesamiento.
    def cancel_organization(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.cancel_button.config(state="disabled")
            self.status_bar.config(text="Cancelando...")

    # Procesa los mensajes recibidos desde el hilo secundario.
    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                if isinstance(message, tuple) and message[0] == "PROGRESS":
                    self.progress_var.set(message[1])
                elif message == "DONE":
                    status_text = "Proceso cancelado." if self.stop_event.is_set() else "¡Proceso completado!"
                    self.status_bar.config(text=status_text)
                    self.progress_var.set(100 if not self.stop_event.is_set() else self.progress_var.get())
                    self.set_ui_state(is_running=False)
                    self.thread = None; return
                else:
                    self.log_text.config(state="normal")
                    self.log_text.insert(tk.END, str(message) + "\n")
                    self.log_text.see(tk.END) # Auto-scroll
                    self.log_text.config(state="disabled")
        except queue.Empty:
            if self.thread and self.thread.is_alive():
                # Si no hay mensajes pero el hilo sigue vivo, vuelve a comprobar en 100ms.
                self.root.after(100, self.process_log_queue)

# ------------------------------------------------------------------------------
# 4. PUNTO DE ENTRADA DE LA APLICACIÓN
# ------------------------------------------------------------------------------
# Este bloque solo se ejecuta cuando se corre este fichero directamente.
if __name__ == "__main__":
    # Crea la ventana principal de la aplicación.
    root = tk.Tk()
    try:
        # Intenta aplicar un estilo nativo de Windows para un mejor aspecto.
        style = ttk.Style(root); style.theme_use('winnative')
    except tk.TclError:
        # Si no está en Windows, simplemente continúa sin el estilo.
        pass
    
    # Crea una instancia de nuestra clase de aplicación.
    app = App(root)
    
    # Inicia el bucle principal de la GUI, que la mantiene visible y receptiva.
    root.mainloop()
