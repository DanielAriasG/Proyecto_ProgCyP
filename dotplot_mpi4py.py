from mpi4py import MPI
import numpy as np
from Bio import SeqIO
from scipy.signal import convolve2d
import argparse
import time

# Función para leer secuencias desde archivos FASTA
def leer_secuencia(fasta_file):
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)

# Generar una parte del dotplot
def generar_dotplot_parcial(seq1, seq2, start, end):
    len_seq2 = len(seq2)
    parcial = np.zeros((end - start, len_seq2))

    for i in range(start, end):
        for j in range(len_seq2):
            if seq1[i] == seq2[j]:
                parcial[i - start, j] = 1

    return parcial

# Filtrar la matriz usando un kernel para detectar diagonales
def filtrar_dotplot(matriz):
    kernel = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
    filtrada = convolve2d(matriz, kernel, mode="same", boundary="fill", fillvalue=0)
    return filtrada

# Función principal para MPI
def main():
    parser = argparse.ArgumentParser(description="Generar y filtrar un dotplot entre dos secuencias usando MPI.")
    parser.add_argument("--file1", type=str, required=True, help="Archivo FASTA de la primera secuencia.")
    parser.add_argument("--file2", type=str, required=True, help="Archivo FASTA de la segunda secuencia.")
    parser.add_argument("--output", type=str, default="dotplot.png", help="Archivo de salida para el dotplot.")
    parser.add_argument("--filtered_output", type=str, default="dotplot_filtrado.png", help="Archivo de salida para el dotplot filtrado.")

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Leer las secuencias
        start_time = time.time()
        seq1 = leer_secuencia(args.file1)
        seq2 = leer_secuencia(args.file2)
        read_time = time.time() - start_time

        len_seq1 = len(seq1)
        chunk_size = len_seq1 // size

        # Dividir el trabajo
        tasks = [(seq1, seq2, i * chunk_size, len_seq1 if i == size - 1 else (i + 1) * chunk_size) for i in range(size)]
    else:
        tasks = None

    # Distribuir las tareas
    task = comm.scatter(tasks, root=0)

    # Procesar la tarea asignada
    partial_result = generar_dotplot_parcial(*task)

    # Recoger los resultados
    gathered_results = comm.gather(partial_result, root=0)

    if rank == 0:
        # Combinar resultados
        dotplot_matrix = np.vstack(gathered_results)

        # Filtrar la matriz
        start_time = time.time()
        filtered_matrix = filtrar_dotplot(dotplot_matrix)
        filter_time = time.time() - start_time

        # Guardar las matrices
        np.savetxt(args.output, dotplot_matrix, fmt="%d")
        np.savetxt(args.filtered_output, filtered_matrix, fmt="%d")

        print(f"Tiempo de lectura de secuencias: {read_time:.2f} segundos")
        print(f"Tiempo de filtrado: {filter_time:.2f} segundos")
        print(f"Dotplot generado y guardado en {args.output}")
        print(f"Dotplot filtrado generado y guardado en {args.filtered_output}")

if __name__ == "__main__":
    main()
