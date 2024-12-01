import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from scipy.signal import convolve2d
import argparse
import time
from multiprocessing import Pool

# Función para leer secuencias desde archivos FASTA
def leer_secuencia(fasta_file):
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)

# Generar una parte del dotplot usando multiprocessing
def generar_dotplot_parcial(args):
    seq1, seq2, start, end = args
    len_seq2 = len(seq2)
    parcial = np.zeros((end - start, len_seq2))

    for i in range(start, end):
        for j in range(len_seq2):
            if seq1[i] == seq2[j]:
                parcial[i - start, j] = 1

    return parcial

# Generar el dotplot completo con multiprocessing
def generar_dotplot_multiproceso(seq1, seq2, num_processes=4):
    len_seq1 = len(seq1)
    chunk_size = len_seq1 // num_processes
    tasks = []

    for i in range(num_processes):
        start = i * chunk_size
        end = len_seq1 if i == num_processes - 1 else (i + 1) * chunk_size
        tasks.append((seq1, seq2, start, end))

    with Pool(processes=num_processes) as pool:
        resultados = pool.map(generar_dotplot_parcial, tasks)

    return np.vstack(resultados)

# Filtrar la matriz usando un kernel para detectar diagonales
def filtrar_dotplot(matriz):
    kernel = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
    filtrada = convolve2d(matriz, kernel, mode="same", boundary="fill", fillvalue=0)
    return filtrada

# Guardar el dotplot como una imagen
def guardar_dotplot(matriz, output_file):
    plt.imshow(matriz, cmap="Greys", origin="lower")
    plt.title("Dotplot")
    plt.xlabel("Secuencia 2")
    plt.ylabel("Secuencia 1")
    plt.savefig(output_file)
    plt.close()

# Función principal
def main():
    parser = argparse.ArgumentParser(description="Generar y filtrar un dotplot entre dos secuencias usando multiprocessing.")
    parser.add_argument("--file1", type=str, required=True, help="Archivo FASTA de la primera secuencia.")
    parser.add_argument("--file2", type=str, required=True, help="Archivo FASTA de la segunda secuencia.")
    parser.add_argument("--output", type=str, default="dotplot.png", help="Archivo de salida para el dotplot.")
    parser.add_argument("--filtered_output", type=str, default="dotplot_filtrado.png", help="Archivo de salida para el dotplot filtrado.")
    parser.add_argument("--processes", type=int, default=4, help="Número de procesos a utilizar.")

    args = parser.parse_args()

    # Medir el tiempo de lectura de secuencias
    start_time = time.time()
    secuencia1 = leer_secuencia(args.file1)
    secuencia2 = leer_secuencia(args.file2)
    read_time = time.time() - start_time

    # Medir el tiempo de generación del dotplot
    start_time = time.time()
    matriz_dotplot = generar_dotplot_multiproceso(secuencia1, secuencia2, num_processes=args.processes)
    dotplot_time = time.time() - start_time

    # Medir el tiempo de filtrado
    start_time = time.time()
    matriz_filtrada = filtrar_dotplot(matriz_dotplot)
    filter_time = time.time() - start_time

    # Guardar las imágenes
    guardar_dotplot(matriz_dotplot, args.output)
    guardar_dotplot(matriz_filtrada, args.filtered_output)

    total_time = read_time + dotplot_time + filter_time

    print(f"Tiempo de lectura de secuencias: {read_time:.2f} segundos")
    print(f"Tiempo de generación del dotplot: {dotplot_time:.2f} segundos")
    print(f"Tiempo de filtrado: {filter_time:.2f} segundos")
    print(f"Tiempo total de ejecución: {total_time:.2f} segundos")
    print(f"Dotplot generado y guardado en {args.output}")
    print(f"Dotplot filtrado generado y guardado en {args.filtered_output}")

if __name__ == "__main__":
    main()
