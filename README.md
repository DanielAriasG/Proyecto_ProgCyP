# Proyecto Final: Análisis de Rendimiento de Dotplot Secuencial vs. Paralelización 
# Daniel Arias Garzón

## Introducción
Este proyecto consiste en generar un dotplot entre dos secuencias de ADN utilizando diferentes enfoques de paralelización: PyCUDA, MPI y procesamiento multithread/multiproceso. Un dotplot es una herramienta visual que permite comparar dos secuencias al representar visualmente sus similitudes. En este programa, se utilizan distintas técnicas de paralelización para mejorar la eficiencia de la tarea de comparación y generación del dotplot.

## Requisitos

### 1. Instalación de CUDA Toolkit
- Descarga e instala el [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) correspondiente a tu sistema operativo.
- Asegúrate de que los controladores de tu tarjeta gráfica NVIDIA sean compatibles con la versión de CUDA que instalaste.

### 2. Instalación de PyCUDA
- Instala PyCUDA usando pip:
  ```bash
  pip install pycuda
  ```

### 3. Instalación de MPI
- Instala MPI en tu sistema:
  - En Linux/MacOS:
    ```bash
    sudo apt install mpich  # o sudo apt install openmpi
    ```
  - En Windows, utiliza [Microsoft MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi).
- Instala `mpi4py`:
  ```bash
  pip install mpi4py
  ```

### 4. Instalación de otras dependencias
- Instala Biopython para manejar archivos FASTA:
  ```bash
  pip install biopython
  ```
- Para la versión multithread y multiproceso, instala `concurrent.futures` (generalmente incluida en Python 3).

## Ejecución
### PyCUDA
Para ejecutar el programa usando PyCUDA, utiliza el siguiente comando desde la línea de comandos:

```bash
python script.py --file1=secuencia1.fasta --file2=secuencia2.fasta --output=dotplot.txt
```

### MPI
Para ejecutar el programa usando MPI, utiliza `mpirun` o `mpiexec`:

```bash
mpirun -n 4 python script.py --file1=secuencia1.fasta --file2=secuencia2.fasta --output=dotplot.txt
```

- `-n 4`: Especifica el número de procesos a utilizar.

### Multithread y Multiproceso
Para ejecutar la versión multithread/multiproceso, ejecuta:

```bash
python script.py --file1=secuencia1.fasta --file2=secuencia2.fasta --output=dotplot.txt --threads=4
```

- `--threads`: Especifica el número de hilos o procesos a utilizar.

## Descripción del Código
- **leer_secuencia**: Lee y devuelve la secuencia de un archivo FASTA.
- **generar_dotplot_gpu**: Genera el dotplot utilizando PyCUDA para realizar la comparación de manera paralela en la GPU.
- **generar_dotplot_mpi**: Genera el dotplot utilizando MPI para dividir la tarea entre procesos.
- **generar_dotplot_multihilo**: Genera el dotplot utilizando procesamiento multithread/multiproceso.
- **guardar_dotplot**: Guarda la matriz del dotplot en un archivo de texto.
- **main**: Controla la ejecución del programa, mide el tiempo de lectura y generación del dotplot, y muestra los resultados.

## Resultados
El programa genera un archivo de salida (`dotplot.txt`) que contiene la matriz del dotplot. Puedes usar herramientas como programas de visualización de matrices o scripts adicionales para representar gráficamente el dotplot.

## Notas
- Asegúrate de que tu sistema cumpla con los requisitos de hardware y software necesarios para ejecutar CUDA y MPI.
- La versión de PyCUDA debe ser compatible con la versión de CUDA instalada.
- Para MPI, verifica que `mpirun` o `mpiexec` esté correctamente configurado en tu sistema.

Para cualquier pregunta o ayuda adicional, no dudes en contactar.

