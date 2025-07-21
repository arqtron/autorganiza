_Script en python que organiza las imagenes de una carpeta en subcarpetas según su contenido, con IA_

# Autorganiza (CPU & GPU)


**[English version below]**

### Versión en Castellano

**Autorganiza** es un potente script de Python que utiliza Inteligencia Artificial para analizar y clasificar automáticamente tus imágenes en carpetas temáticas según su contenido real, no solo por su nombre de archivo.

¿Tienes una carpeta con miles de capturas de pantalla, fotos y memes mezclados? Autorganiza la pone en orden por ti.

#### Características Principales
*   **Clasificación por Contenido:** Olvida los nombres de archivo. Autorganiza "mira" tus imágenes y entiende si es un paisaje, una factura, un meme o el código de un programa.
*   **Totalmente Personalizable:** Define tus propias categorías de forma muy sencilla. Solo tienes que editar un diccionario de Python, asociando frases descriptivas (para la IA) con los nombres de carpeta que tú quieras.
*   **Inteligencia Zero-Shot:** Gracias al modelo CLIP de OpenAI, no necesita un tedioso proceso de entrenamiento. Funciona desde el primer momento con las categorías que le indiques.
*   **Control de Calidad:** Establece un umbral de confianza. Las imágenes que la IA no pueda clasificar con seguridad se mueven a una carpeta `revisar` para que tú tomes la decisión final.
*   **Doble Versión (CPU y GPU):**
    *   `autorganiza`: Usa la CPU y es compatible con cualquier ordenador.
    *   `autorganiza-gpu`: Diseñada para una aceleración masiva en ordenadores con tarjetas gráficas NVIDIA (CUDA), procesando miles de imágenes en una fracción del tiempo.

---

### English Version

**Autorganiza** is a powerful Python script that leverages Artificial Intelligence to automatically analyze and sort your images into thematic folders based on their actual content, not just their filenames.

Do you have a folder with thousands of mixed screenshots, photos, and memes? Autorganiza puts it in order for you.

#### Key Features
*   **Content-Based Classification:** Forget filenames. Autorganiza "looks" at your images and understands if it's a landscape, an invoice, a meme, or a code snippet.
*   **Fully Customizable:** Easily define your own categories. Simply edit a Python dictionary, mapping descriptive prompts (for the AI) to your desired folder names.
*   **Zero-Shot Intelligence:** Powered by OpenAI's CLIP model, it requires no tedious training process. It works out-of-the-box with any categories you provide.
*   **Quality Control:** Set a confidence threshold. Images that the AI can't classify with high certainty are moved to a `revisar` (review) folder for your final say.
*   **Dual Version (CPU & GPU):**
    *   `autorganiza`: Uses the CPU and is compatible with any computer.
    *   `autorganiza-gpu`: Designed for massive acceleration on computers with NVIDIA graphics cards (CUDA), processing thousands of images in a fraction of the time.
