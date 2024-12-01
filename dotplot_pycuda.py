import numpy as np
from Bio import SeqIO
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import argparse
import time

# Función para leer secuencias desde archivos FASTA
def leer_secuencia(fasta_file):
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)

# Código CUDA para generar el dotplot
cuda_code = """
__global__ void dotplot_kernel(char *seq1, char *seq2, int len1, int len2, int *dotplot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < len1 && j < len2) {
        dotplot[i * len2 + j] = (seq1[i] == seq2[j]) ? 1 : 0;
    }
}
"""

# Generar el dotplot utilizando PyCUDA
def generar_dotplot_gpu(seq1, seq2):
    len1 = len(seq1)
    len2 = len(seq2)

    # Convertir las secuencias a formato de bytes
    seq1_bytes = np.frombuffer(seq1.encode('ascii'), dtype=np.byte)
    seq2_bytes = np.frombuffer(seq2.encode('ascii'), dtype=np.byte)

    # Crear matriz de salida
    dotplot = np.zeros((len1, len2), dtype=np.int32)

    # Transferir datos a la GPU
    seq1_gpu = cuda.mem_alloc(seq1_bytes.nbytes)
    seq2_gpu = cuda.mem_alloc(seq2_bytes.nbytes)
    dotplot_gpu = cuda.mem_alloc(dotplot.nbytes)

    cuda.memcpy_htod(seq1_gpu, seq1_bytes)
    cuda.memcpy_htod(seq2_gpu, seq2_bytes)

    # Compilar y ejecutar el kernel CUDA
    mod = SourceModule(cuda_code)
    kernel = mod.get_function("dotplot_kernel")

    block_size = (16, 16, 1)
    grid_size = ((len1 + block_size[0] - 1) // block_size[0],
                 (len2 + block_size[1] - 1) // block_size[1], 1)

    kernel(seq1_gpu, seq2_gpu, np.int32(len1), np.int32(len2), dotplot_gpu, block=block_size, grid=grid_size)

    # Transferir resultados de vuelta a la CPU
    cuda.memcpy_dtoh(dotplot, dotplot_gpu)

    return dotplot

# Guardar el dotplot como archivo
def guardar_dotplot(matriz, output_file):
    np.savetxt(output_file, matriz, fmt="%d")

# Función principal
def main():
    parser = argparse.ArgumentParser(description="Generar un dotplot entre dos secuencias usando PyCUDA.")
    parser.add_argument("--file1", type=str, required=True, help="Archivo FASTA de la primera secuencia.")
    parser.add_argument("--file2", type=str, required=True, help="Archivo FASTA de la segunda secuencia.")
    parser.add_argument("--output", type=str, default="dotplot.txt", help="Archivo de salida para el dotplot.")

    args = parser.parse_args()

    # Leer las secuencias
    start_time = time.time()
    seq1 = leer_secuencia(args.file1)
    seq2 = leer_secuencia(args.file2)
    read_time = time.time() - start_time

    # Generar el dotplot
    start_time = time.time()
    dotplot = generar_dotplot_gpu(seq1, seq2)
    generate_time = time.time() - start_time

    # Guardar el dotplot
    guardar_dotplot(dotplot, args.output)

    print(f"Tiempo de lectura de secuencias: {read_time:.2f} segundos")
    print(f"Tiempo de generación del dotplot: {generate_time:.2f} segundos")
    print(f"Dotplot generado y guardado en {args.output}")

if __name__ == "__main__":
    main()
