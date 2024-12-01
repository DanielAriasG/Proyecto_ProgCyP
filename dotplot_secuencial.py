import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from scipy.signal import convolve2d
import argparse

# Función para leer secuencias desde archivos FASTA
def leer_secuencia(fasta_file):
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)

# Generar el dotplot
def generar_dotplot(seq1, seq2):
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)
    matriz = np.zeros((len_seq1, len_seq2))

    for i in range(len_seq1):
        for j in range(len_seq2):
            if seq1[i] == seq2[j]:
                matriz[i, j] = 1

    return matriz

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
    parser = argparse.ArgumentParser(description="Generar y filtrar un dotplot entre dos secuencias.")
    parser.add_argument("--file1", type=str, required=True, help="Archivo FASTA de la primera secuencia.")
    parser.add_argument("--file2", type=str, required=True, help="Archivo FASTA de la segunda secuencia.")
    parser.add_argument("--output", type=str, default="dotplot.png", help="Archivo de salida para el dotplot.")
    parser.add_argument("--filtered_output", type=str, default="dotplot_filtrado.png", help="Archivo de salida para el dotplot filtrado.")

    args = parser.parse_args()

    # Leer las secuencias
    secuencia1 = leer_secuencia(args.file1)
    secuencia2 = leer_secuencia(args.file2)

    # Generar dotplot
    matriz_dotplot = generar_dotplot(secuencia1, secuencia2)

    # Filtrar el dotplot
    matriz_filtrada = filtrar_dotplot(matriz_dotplot)

    # Guardar las imágenes
    guardar_dotplot(matriz_dotplot, args.output)
    guardar_dotplot(matriz_filtrada, args.filtered_output)

    print(f"Dotplot generado y guardado en {args.output}")
    print(f"Dotplot filtrado generado y guardado en {args.filtered_output}")

if __name__ == "__main__":
    main()
