import numpy as np
from Bio import SeqIO
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import argparse
import time

# Function to read sequences from FASTA files
def leer_secuencia(fasta_file):
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)

# CUDA kernel code to generate the dotplot
cuda_code = """
__global__ void dotplot_kernel(char *seq1, char *seq2, int len1, int len2, int *dotplot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < len1 && j < len2) {
        dotplot[i * len2 + j] = (seq1[i] == seq2[j]) ? 1 : 0;
    }
}
"""

# Function to filter the dotplot using a kernel
def filtrar_dotplot(matriz):
    kernel = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
    filtrada = convolve2d(matriz, kernel, mode="same", boundary="fill", fillvalue=0)
    return filtrada

# Function to generate the dotplot using PyCUDA
def generar_dotplot_gpu(seq1, seq2):
    len1 = len(seq1)
    len2 = len(seq2)

    # Convert sequences to byte format
    seq1_bytes = np.frombuffer(seq1.encode('ascii'), dtype=np.byte)
    seq2_bytes = np.frombuffer(seq2.encode('ascii'), dtype=np.byte)

    # Create output matrix
    dotplot = np.zeros((len1, len2), dtype=np.int32)

    # Transfer data to GPU
    seq1_gpu = cuda.mem_alloc(seq1_bytes.nbytes)
    seq2_gpu = cuda.mem_alloc(seq2_bytes.nbytes)
    dotplot_gpu = cuda.mem_alloc(dotplot.nbytes)

    cuda.memcpy_htod(seq1_gpu, seq1_bytes)
    cuda.memcpy_htod(seq2_gpu, seq2_bytes)

    # Compile and execute CUDA kernel
    mod = SourceModule(cuda_code)
    kernel = mod.get_function("dotplot_kernel")

    block_size = (16, 16, 1)
    grid_size = ((len1 + block_size[0] - 1) // block_size[0],
                 (len2 + block_size[1] - 1) // block_size[1], 1)

    kernel(seq1_gpu, seq2_gpu, np.int32(len1), np.int32(len2), dotplot_gpu, block=block_size, grid=grid_size)

    # Transfer results back to CPU
    cuda.memcpy_dtoh(dotplot, dotplot_gpu)

    return dotplot

# Function to save the dotplot to a file
def guardar_dotplot(matriz, output_file):
    np.savetxt(output_file, matriz, fmt="%d")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Generate a dotplot between two sequences using PyCUDA.")
    parser.add_argument("--file1", type=str, required=True, help="Path to the first FASTA file.")
    parser.add_argument("--file2", type=str, required=True, help="Path to the second FASTA file.")
    parser.add_argument("--output", type=str, default="dotplot.txt", help="Output file for the dotplot.")

    args = parser.parse_args()

    # Measure sequence reading time
    start_time = time.time()
    seq1 = leer_secuencia(args.file1)
    seq2 = leer_secuencia(args.file2)
    read_time = time.time() - start_time

    # Measure dotplot generation time
    start_time = time.time()
    dotplot = generar_dotplot_gpu(seq1, seq2)
    generate_time = time.time() - start_time

    # Apply filter to the dotplot
    start_time = time.time()
    filtered_dotplot = filtrar_dotplot(dotplot)
    filter_time = time.time() - start_time

    # Save the dotplots
    guardar_dotplot(dotplot, args.output)
    guardar_dotplot(filtered_dotplot, "filtered_" + args.output)

    print(f"Time to read sequences: {read_time:.2f} seconds")
    print(f"Time to generate dotplot: {generate_time:.2f} seconds")
    print(f"Time to filter dotplot: {filter_time:.2f} seconds")
    print(f"Dotplot saved to {args.output}")
    print(f"Filtered dotplot saved to filtered_{args.output}")

if __name__ == "__main__":
    main()
