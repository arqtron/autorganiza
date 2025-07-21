import os
import shutil
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

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
BATCH_SIZE = 32
# -----------------------------

def organizar_con_ia(carpeta_principal):
    """
    Organiza imágenes usando la GPU, con un menú para elegir entre modelos de alta potencia.
    """
    print("Iniciando la organización con IA (Modo GPU Acelerado).")
    
    # --- Selección de Modelo ---
    modelos = {
        '1': "openai/clip-vit-base-patch32",
        '2': "openai/clip-vit-large-patch14",
        '3': "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" # ¡NUEVO! Modelo OpenCLIP
    }
    print("\n--- Selección del Modelo de IA ---")
    print("1: Modelo Base (OpenAI, Equilibrado)")
    print("2: Modelo Grande (OpenAI, Alta Precisión)")
    print("3: Modelo OpenCLIP (Open Source, Máxima Precisión)")
    
    eleccion = ''
    while eleccion not in modelos:
        eleccion = input(f"Selecciona una opción ({', '.join(modelos.keys())}) y pulsa Enter: ")
    
    modelo_id = modelos[eleccion]
    print(f"\nHas seleccionado el modelo: {modelo_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando el dispositivo: {device.upper()}")
    if device == "cpu":
        print("\nADVERTENCIA: No se ha detectado GPU. El proceso será MUY LENTO.")

    try:
        print(f"Cargando el modelo '{modelo_id}'... (Puede tardar MUCHO la primera vez)")
        # Lógica de carga unificada y simple
        modelo = CLIPModel.from_pretrained(modelo_id).to(device)
        procesador = CLIPProcessor.from_pretrained(modelo_id)
        print("¡Modelo cargado con éxito!")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}.")
        return

    # --- Creación de carpetas y contadores ---
    etiquetas_para_ia = list(ETIQUETAS_Y_CARPETAS.keys())
    carpetas_destino = list(ETIQUETAS_Y_CARPETAS.values()) + ['revisar']
    for nombre_carpeta in carpetas_destino:
        if not os.path.exists(os.path.join(carpeta_principal, nombre_carpeta)):
            os.makedirs(os.path.join(carpeta_principal, nombre_carpeta))
    
    contador_movimientos = {nombre_carpeta: 0 for nombre_carpeta in carpetas_destino}

    nombre_script = os.path.basename(__file__)
    archivos_imagen = [f for f in os.listdir(carpeta_principal) if os.path.isfile(os.path.join(carpeta_principal, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg')) and f != nombre_script]
    print(f"\nSe encontraron {len(archivos_imagen)} imágenes para procesar.")

    lotes = [archivos_imagen[i:i + BATCH_SIZE] for i in range(0, len(archivos_imagen), BATCH_SIZE)]

    for num_lote, lote_archivos in enumerate(lotes):
        print(f"\n--- Procesando Lote {num_lote + 1}/{len(lotes)} ---")
        
        imagenes_lote, rutas_lote = [], []
        
        for nombre_archivo in lote_archivos:
            ruta_completa = os.path.join(carpeta_principal, nombre_archivo)
            try:
                imagenes_lote.append(Image.open(ruta_completa).convert("RGB"))
                rutas_lote.append(ruta_completa)
            except Exception as e:
                print(f"  -> ERROR al abrir '{nombre_archivo}': {e}. Moviendo a 'revisar'.")
                shutil.move(ruta_completa, os.path.join(carpeta_principal, 'revisar'))
                contador_movimientos['revisar'] += 1
        
        if not imagenes_lote: continue

        # Lógica de procesamiento unificada
        inputs = procesador(text=etiquetas_para_ia, images=imagenes_lote, return_tensors="pt", padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = modelo(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=1)
        
        for i, ruta_completa_archivo in enumerate(rutas_lote):
            nombre_archivo_actual = os.path.basename(ruta_completa_archivo)
            prob_imagen = probs[i]
            mejor_prob, mejor_indice = prob_imagen.max().item(), prob_imagen.argmax().item()
            
            if mejor_prob >= UMBRAL_CONFIANZA:
                etiqueta_ganadora = etiquetas_para_ia[mejor_indice]
                nombre_carpeta_destino = ETIQUETAS_Y_CARPETAS[etiqueta_ganadora]
                print(f"  Procesado: '{nombre_archivo_actual}' -> Moviendo a '{nombre_carpeta_destino}' (Confianza: {mejor_prob:.2%})")
                shutil.move(ruta_completa_archivo, os.path.join(carpeta_principal, nombre_carpeta_destino, nombre_archivo_actual))
                contador_movimientos[nombre_carpeta_destino] += 1
            else:
                print(f"  Procesado: '{nombre_archivo_actual}' -> Moviendo a 'revisar' (Confianza baja: {mejor_prob:.2%})")
                shutil.move(ruta_completa_archivo, os.path.join(carpeta_principal, 'revisar', nombre_archivo_actual))
                contador_movimientos['revisar'] += 1

    # --- Resumen Final ---
    print("\n-----------------------------------------------------")
    print("---             Resumen de la Organización          ---")
    print("-----------------------------------------------------")
    print(f"Total de archivos encontrados: {len(archivos_imagen)}")
    total_movidos = sum(contador_movimientos.values())
    print(f"Total de archivos procesados y movidos: {total_movidos}\n")
    print("Desglose por carpeta:")
    for carpeta, cantidad in sorted(contador_movimientos.items()):
        if cantidad > 0:
            print(f"  - {carpeta}: {cantidad} archivo(s)")
    print("\n¡Organización con IA completada!")


if __name__ == "__main__":
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    organizar_con_ia(directorio_script)