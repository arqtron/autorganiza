import os
import shutil
from PIL import Image, UnidentifiedImageError # Importamos el error específico
import torch
from transformers import CLIPProcessor, CLIPModel

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import threading
import queue

# --- CONFIGURACIÓN (copia aquí tu diccionario y umbral) ---
# --- CONFIGURACIÓN PRINCIPAL - ¡DICCIONARIO RECOMENDADO EN INGLÉS! ---
ETIQUETAS_Y_CARPETAS = {
    "a screenshot of text, a web page, an application, or a software program": "capturas-de-pantalla",
    "the exterior of a building, a facade, a blueprint, or an architectural structure": "arquitectura",
    "an electronic device, a circuit board, a gadget, a computer, or hardware": "electronica",
    "a screenshot of a registration form, a login page, or a user profile": "registros-en-webs",
    "a meme, a joke, a comic strip, or a funny humorous image": "humor",
    "a painting, a sculpture, a drawing, or a piece of classic or modern art": "arte",
    "a movie still, a scene from a TV series, a movie poster, an actor or actress": "cine-tv",
    "a home automation system, a smart switch, a sensor, or a home control panel, an image with the home assistant logo": "domotica",
    "a presentation slide, a tutorial, a chart, a diagram, or training material": "formacion",
    "the interior of a house, a decorated room, furniture, or interior design": "interiorismo",
    "a car, a motorcycle, an airplane, a boat, or any motor vehicle": "motor",
    "a plate of food, a drink, ingredients, or a cooking recipe": "recetas",
    "an image of an order, a purchase process, an invoice, an item with a price tag": "compras",
    "a graphic design example, a logo, a user interface, an industrial design object, a pattern, or an aesthetic composition": "disseny",
    "a VR headset, a screenshot of a virtual or augmented reality application": "VR",
    "a photo of a person, a group of people, a portrait, or a selfie": "gente"
}
UMBRAL_CONFIANZA = 0.60
BATCH_SIZE = 16

# --- FUNCIÓN WORKER ---
def organizar_con_ia_worker(carpeta_principal, modelo_id, log_queue):
    
    def log(message):
        log_queue.put(message)

    try:
        log("Iniciando la organización con IA (Modo acelerado con GPU nvidia).")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Usando el dispositivo: {device.upper()}")
        if device == "cpu":
            log("\nADVERTENCIA: No se ha detectado GPU. El proceso será MUY LENTO.")

        log(f"Cargando el modelo '{modelo_id}'...")
        modelo = CLIPModel.from_pretrained(modelo_id).to(device)
        procesador = CLIPProcessor.from_pretrained(modelo_id)
        log("Modelo cargado con éxito")

        etiquetas_para_ia = list(ETIQUETAS_Y_CARPETAS.keys())
        carpetas_destino = list(ETIQUETAS_Y_CARPETAS.values()) + ['revisar']
        for nombre_carpeta in carpetas_destino:
            ruta_carpeta = os.path.join(carpeta_principal, nombre_carpeta)
            if not os.path.exists(ruta_carpeta):
                os.makedirs(ruta_carpeta)
        
        contador_movimientos = {nombre_carpeta: 0 for nombre_carpeta in carpetas_destino}

        archivos_imagen = [f for f in os.listdir(carpeta_principal) if os.path.isfile(os.path.join(carpeta_principal, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and f != os.path.basename(__file__)]
        log(f"\nSe encontraron {len(archivos_imagen)} imágenes para procesar.")

        lotes = [archivos_imagen[i:i + BATCH_SIZE] for i in range(0, len(archivos_imagen), BATCH_SIZE)]

        for num_lote, lote_archivos in enumerate(lotes):
            log(f"\n--- Procesando Lote {num_lote + 1}/{len(lotes)} ---")
            
            imagenes_lote, rutas_lote = [], []
            for nombre_archivo in lote_archivos:
                ruta_completa = os.path.join(carpeta_principal, nombre_archivo)
                # --- Bloque try/except para cada imagen individual ---
                try:
                    # Intenta abrir la imagen
                    img = Image.open(ruta_completa)
                    # Forzar la carga de datos para detectar errores de truncamiento
                    img.load() 
                    imagenes_lote.append(img.convert("RGB"))
                    rutas_lote.append(ruta_completa)
                except (UnidentifiedImageError, OSError) as e:
                    # Captura errores de formato o de ficheros truncados
                    log(f"  -> ERROR: No se pudo leer '{nombre_archivo}'. Posiblemente corrupta. ({e})")
                    log("     Moviendo a la carpeta 'revisar'...")
                    shutil.move(ruta_completa, os.path.join(carpeta_principal, 'revisar'))
                    contador_movimientos['revisar'] += 1
                except Exception as e:
                    # Captura cualquier otro error inesperado con una imagen
                    log(f"  -> ERROR INESPERADO con '{nombre_archivo}': {e}.")
                    log("     Moviendo a la carpeta 'revisar'...")
                    shutil.move(ruta_completa, os.path.join(carpeta_principal, 'revisar'))
                    contador_movimientos['revisar'] += 1
            
            if not imagenes_lote:
                log("  -> Ninguna imagen válida en este lote para procesar.")
                continue

            # El resto del procesamiento del lote continúa con las imágenes válidas
            inputs = procesador(text=etiquetas_para_ia, images=imagenes_lote, return_tensors="pt", padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = modelo(**inputs)
            
            probs = outputs.logits_per_image.softmax(dim=1)
            
            for i, ruta_completa_archivo in enumerate(rutas_lote):
                # ... (resto de la lógica de clasificación y movimiento) ...
                nombre_archivo_actual = os.path.basename(ruta_completa_archivo)
                prob_imagen = probs[i]
                mejor_prob, mejor_indice = prob_imagen.max().item(), prob_imagen.argmax().item()
                
                if mejor_prob >= UMBRAL_CONFIANZA:
                    etiqueta_ganadora = etiquetas_para_ia[mejor_indice]
                    nombre_carpeta_destino = ETIQUETAS_Y_CARPETAS[etiqueta_ganadora]
                    log(f"  Procesado: '{nombre_archivo_actual}' -> Moviendo a '{nombre_carpeta_destino}' ({mejor_prob:.2%})")
                    shutil.move(ruta_completa_archivo, os.path.join(carpeta_principal, nombre_carpeta_destino, nombre_archivo_actual))
                    contador_movimientos[nombre_carpeta_destino] += 1
                else:
                    log(f"  Procesado: '{nombre_archivo_actual}' -> Moviendo a 'revisar' ({mejor_prob:.2%})")
                    shutil.move(ruta_completa_archivo, os.path.join(carpeta_principal, 'revisar', nombre_archivo_actual))
                    contador_movimientos['revisar'] += 1
        
        # ... (código del resumen final) ...
        log("\n-----------------------------------------------------")
        log("---             Resumen de la Organización          ---")
        log("-----------------------------------------------------")
        total_movidos = sum(contador_movimientos.values())
        log(f"Total de archivos procesados y movidos: {total_movidos}\n")
        log("Desglose por carpeta:")
        for carpeta, cantidad in sorted(contador_movimientos.items()):
            if cantidad > 0:
                log(f"  - {carpeta}: {cantidad} archivo(s)")
        log("\n¡Organización con IA completada!")

    except Exception as e:
        log(f"\n\nERROR CRÍTICO GENERAL: {e}")
    finally:
        log_queue.put("DONE")

# --- CLASE DE LA APLICACIÓN GUI CON RADIO BUTTONS ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Autorganiza IA")
        self.root.geometry("800x600")

        self.folder_path = tk.StringVar()
        self.model_options = {
            "Modelo Base (OpenAI)": "openai/clip-vit-base-patch32",
            "Modelo Grande (OpenAI)": "openai/clip-vit-large-patch14",
            "Modelo OpenCLIP (Máxima Precisión)": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        }
        # Variable para el radio button, se guarda el ID del modelo
        self.selected_model_id = tk.StringVar(value=list(self.model_options.values())[0])
        
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill="both", expand=True)

        folder_frame = ttk.LabelFrame(frame, text="1. Selecciona la Carpeta a Organizar", padding="10")
        folder_frame.pack(fill="x", pady=5)
        
        self.folder_label = ttk.Label(folder_frame, textvariable=self.folder_path)
        self.folder_label.pack(side="left", fill="x", expand=True, padx=5)
        
        self.browse_button = ttk.Button(folder_frame, text="Explorar...", command=self.browse_folder)
        self.browse_button.pack(side="right")

        # --- Marco para los Radio Buttons ---
        model_frame = ttk.LabelFrame(frame, text="2. Selecciona el Modelo de IA", padding="10")
        model_frame.pack(fill="x", pady=5)
        
        # Crea un radio button por cada opción en el diccionario
        for (text, model_id) in self.model_options.items():
            ttk.Radiobutton(
                model_frame, 
                text=text, 
                variable=self.selected_model_id, 
                value=model_id
            ).pack(anchor="w", padx=10)
        
        self.start_button = ttk.Button(frame, text="Iniciar Organización", command=self.start_organization, state="disabled")
        self.start_button.pack(fill="x", pady=10)

        log_frame = ttk.LabelFrame(frame, text="Registro de Actividad", padding="10")
        log_frame.pack(fill="both", expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state="disabled", bg="black", fg="white")
        self.log_text.pack(fill="both", expand=True)
        
        self.status_bar = ttk.Label(self.root, text="Listo.", padding="5", relief="sunken")
        self.status_bar.pack(side="bottom", fill="x")

    def browse_folder(self):
        # ... (sin cambios) ...
        path = filedialog.askdirectory()
        if path:
            self.folder_path.set(path)
            self.start_button.config(state="normal")
            self.status_bar.config(text=f"Carpeta seleccionada: {path}")

    def start_organization(self):
        self.start_button.config(state="disabled")
        self.browse_button.config(state="disabled")
        # Deshabilita los radio buttons en lugar del menú
        for child in self.root.winfo_children()[0].winfo_children()[1].winfo_children():
            if isinstance(child, ttk.Radiobutton):
                child.config(state="disabled")
        
        self.status_bar.config(text="Procesando... por favor, espera.")

        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        
        self.log_queue = queue.Queue()
        
        self.thread = threading.Thread(
            target=organizar_con_ia_worker,
            # Pasa el ID del modelo directamente desde la variable de los radio buttons
            args=(self.folder_path.get(), self.selected_model_id.get(), self.log_queue)
        )
        self.thread.start()
        
        self.root.after(100, self.process_log_queue)

    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                if message == "DONE":
                    self.status_bar.config(text="¡Proceso completado!")
                    self.start_button.config(state="normal")
                    self.browse_button.config(state="normal")
                    # Habilita los radio buttons de nuevo
                    for child in self.root.winfo_children()[0].winfo_children()[1].winfo_children():
                        if isinstance(child, ttk.Radiobutton):
                            child.config(state="normal")
                    return
                else:
                    self.log_text.config(state="normal")
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state="disabled")
        except queue.Empty:
            if self.thread.is_alive():
                self.root.after(100, self.process_log_queue)

if __name__ == "__main__":
    root = tk.Tk()
    # Para un mejor aspecto en Windows
    style = ttk.Style(root)
    style.theme_use('winnative')
    app = App(root)
    root.mainloop()