```
   [ jpg ]
[ png ] [ img ]
   [ jpg ]
      |
      V
  [ Autorganiza.py ]
      |
+---------+---------+
|         |         |
V         V         V
[arte] [facturas] [humor]
```

# Autorganiza (CPU & GPU)

**[English version below]**

### Versión en Castellano

**Autorganiza** es un potente script de Python que utiliza Inteligencia Artificial para analizar y clasificar automáticamente tus imágenes en carpetas temáticas según su contenido real.

#### Instalación

1.  **Prerrequisito: Python**
    *   Asegúrate de tener Python 3.8 o una versión más reciente instalada desde [python.org](https://www.python.org/downloads/).

2.  **Descargar el Repositorio**
    *   Descarga los archivos `autorganiza` (CPU) y/o `autorganiza-gpu` en tu ordenador.

3.  **Instalar las Librerías**
    *   Abre una terminal (`cmd` o `PowerShell`) y elige **una** de las siguientes opciones:

    **Opción A: Para la versión CPU (`autorganiza`)**
    ```bash
    pip install torch transformers Pillow
    ```

    **Opción B: Para la versión GPU (`autorganiza-gpu`)**
    *Requiere una tarjeta gráfica NVIDIA (CUDA).*
    *   Obtén el comando de instalación de PyTorch con CUDA desde su **[página oficial](https://pytorch.org/get-started/locally/)**. Será algo similar a:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
    *   Instala el resto de librerías:
        ```bash
        pip install transformers Pillow
        ```

> ### ⚠️ Aviso Importante sobre el Idioma del Diccionario
> **Se recomienda encarecidamente escribir las descripciones del diccionario `ETIQUETAS_Y_CARPETAS` en INGLÉS.**
>
> Dado que los modelos de IA ofrecidos (OpenAI y OpenCLIP) fueron entrenados principalmente con datos en inglés, proporcionarles descripciones en este idioma mejorará notablemente la precisión y fiabilidad de la clasificación.

#### Características Principales
*   **Clasificación por Contenido:** "Mira" tus imágenes y entiende su contexto.
*   **Totalmente Personalizable:** Define tus propias categorías editando el diccionario del script.
*   **Selección de IA:** Elige al inicio entre diferentes modelos de IA según tus necesidades de velocidad y precisión.
*   **Control de Calidad:** Las imágenes dudosas se mueven a una carpeta `revisar`.
*   **Resumen Final:** Muestra un informe detallado de cuántos archivos se han movido a cada carpeta.

#### Selección del Modelo (Rendimiento vs. Precisión)
El script te permite elegir al inicio entre tres potentes modelos de IA.

*   **`openai/clip-vit-base-patch32` (Modelo Base - Equilibrado)**
    *   **Descripción:** El estándar de OpenAI. Ofrece un excelente equilibrio entre velocidad y precisión. Ideal para la mayoría de los usuarios.
    *   **Requisitos (Estimados):** 6 GB de VRAM (GPU) / 8-16 GB de RAM (CPU).

*   **`openai/clip-vit-large-patch14` (Modelo Grande - Alta Precisión)**
    *   **Descripción:** La versión superior de OpenAI. Notablemente más preciso, ideal para obtener resultados de alta calidad.
    *   **Requisitos (Estimados):** 8-10 GB de VRAM (GPU) / 16 GB de RAM (CPU).

*   **`laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (Modelo OpenCLIP - Máxima Precisión)**
    *   **Descripción:** Un modelo de código abierto entrenado en el masivo dataset LAION-2B (2.000 millones de imágenes). Ofrece una precisión de vanguardia.
    *   **Requisitos (Estimados):** 12-16 GB de VRAM (GPU). *(Su uso en modo CPU será extremadamente lento)*.

---

### English Version

**Autorganiza** is a powerful Python script that leverages Artificial Intelligence to automatically analyze and sort your images into thematic folders based on their actual content.

#### Installation

1.  **Prerequisite: Python**
    *   Ensure you have Python 3.8 or newer installed from [python.org](https://www.python.org/downloads/).

2.  **Download the Repository**
    *   Download the `autorganiza` (CPU) and/or `autorganiza-gpu` files to your computer.

3.  **Install Libraries**
    *   Open a terminal (`cmd` or `PowerShell`) and choose **one** of the following options:

    **Option A: For the CPU Version (`autorganiza`)**
    ```bash
    pip install torch transformers Pillow
    ```

    **Option B: For the GPU Version (`autorganiza-gpu`)**
    *Requires an NVIDIA graphics card (CUDA).*
    *   Get the PyTorch installation command with CUDA support from the **[official website](https://pytorch.org/get-started/locally/)**. It will look similar to this:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
    *   Install the remaining libraries:
        ```bash
        pip install transformers Pillow
        ```

> ### ⚠️ Important Note on Dictionary Language
> **It is strongly recommended to write the descriptions in the `ETIQUETAS_Y_CARPETAS` dictionary in ENGLISH.**
>
> Since the offered AI models (OpenAI and OpenCLIP) were primarily trained on English data, providing them with English prompts will significantly improve classification accuracy and reliability.

#### Key Features
*   **Content-Based Classification:** It "looks" at your images and understands their context.
*   **Fully Customizable:** Define your own categories by editing the script's dictionary.
*   **AI Selection:** Choose between different AI models at startup based on your speed and accuracy needs.
*   **Quality Control:** Uncertain images are moved to a `revisar` (review) folder.
*   **Final Summary:** Displays a detailed report of how many files were moved to each folder.

#### Model Selection (Performance vs. Accuracy)
The script allows you to choose from three powerful AI models at runtime.

*   **`openai/clip-vit-base-patch32` (Base Model - Balanced)**
    *   **Description:** The OpenAI standard. Offers an excellent balance between speed and accuracy. Ideal for most users.
    *   **Requirements (Est.):** 6 GB of VRAM (GPU) / 8-16 GB of RAM (CPU).

*   **`openai/clip-vit-large-patch14` (Large Model - High Accuracy)**
    *   **Description:** OpenAI's superior version. Noticeably more accurate, ideal for high-quality results.
    *   **Requirements (Est.):** 8-10 GB of VRAM (GPU) / 16 GB of RAM (CPU).

*   **`laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (OpenCLIP Model - Max Accuracy)**
    *   **Description:** An open-source model trained on the massive LAION-2B dataset (2 billion images). It offers state-of-the-art accuracy.
    *   **Requirements (Est.):** 12-16 GB of VRAM (GPU). *(Running this on CPU will be extremely slow)*.
