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
# -----------------------------

def organizar_con_ia(carpeta_principal):
    """
    Organiza imágenes basándose en su contenido (versión optimizada para CPU).
    """
    print("Iniciando la organización con IA (Modo CPU).")
    
    # --- Selección de Modelo ---
    modelos = {
        '1': "openai/clip-vit-base-patch32",
        '2': "openai/clip-vit-large-patch14",
        '3': "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    }
    print("\n--- Selección del Modelo de IA ---")
    print("1: Modelo Base (OpenAI, Recomendado para CPU)")
    print("2: Modelo Grande (OpenAI, Preciso pero MUY LENTO en CPU)")
    print("3: Modelo OpenCLIP (Máxima Precisión, EXTREMADAMENTE LENTO en CPU)")
    
    eleccion = ''
    while eleccion not in modelos:
        eleccion = input(f"Selecciona una opción ({', '.join(modelos.keys())}) y pulsa Enter: ")
    
    modelo_id = modelos[eleccion]
    print(f"\nHas seleccionado el modelo: {modelo_id}")

    try:
        print(f"Cargando el modelo '{modelo_id}'... (Puede tardar la primera vez)")
        # Carga el modelo directamente a la CPU. No hay lógica de 'device'.
        modelo = CLIPModel.from_pretrained(modelo_id)
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

    # --- Bucle de procesamiento simple, una imagen a la vez ---
    for i, nombre_archivo in enumerate(archivos_imagen):
        ruta_completa_archivo = os.path.join(carpeta_principal, nombre_archivo)
        print(f"\nProcesando [{i+1}/{len(archivos_imagen)}]: {nombre_archivo}")

        try:
            imagen = Image.open(ruta_completa_archivo).convert("RGB")
            inputs = procesador(text=etiquetas_para_ia, images=imagen, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = modelo(**inputs)
            
            probs = outputs.logits_per_image.softmax(dim=1)
            
            mejor_prob = probs.max().item()
            mejor_indice = probs.argmax().item()
            
            if mejor_prob >= UMBRAL_CONFIANZA:
                etiqueta_ganadora = etiquetas_para_ia[mejor_indice]
                nombre_carpeta_destino = ETIQUETAS_Y_CARPETAS[etiqueta_ganadora]
                
                print(f"  -> Coincidencia: '{etiqueta_ganadora.split(',')[0]}' (Confianza: {mejor_prob:.2%})")
                print(f"  -> Moviendo a la carpeta '{nombre_carpeta_destino}'")
                shutil.move(ruta_completa_archivo, os.path.join(carpeta_principal, nombre_carpeta_destino, nombre_archivo))
                contador_movimientos[nombre_carpeta_destino] += 1
            else:
                print(f"  -> Confianza baja ({mejor_prob:.2%}). Moviendo a la carpeta 'revisar'")
                shutil.move(ruta_completa_archivo, os.path.join(carpeta_principal, 'revisar', nombre_archivo))
                contador_movimientos['revisar'] += 1

        except Exception as e:
            print(f"  -> ERROR al procesar el fichero: {e}. Moviendo a 'revisar'.")
            shutil.move(ruta_completa_archivo, os.path.join(carpeta_principal, 'revisar', nombre_archivo))
            contador_movimientos['revisar'] += 1
            continue
    
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