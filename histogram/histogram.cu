#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;



/*

Compute histogram of a .pgm image using Cuda. Used to compare two versions.
In the first version, the bins reside in global device memory, and are
incremented using atomic addition.
In the second version, in order to limit contention for the bins in global
memory, a separate subhistogram per thread block is kept in shared memory and
afterwards combined.

*/







/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}

__global__ void histogramKernel(unsigned char* image, long img_size, unsigned int* histogram) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < img_size) {
        atomicAdd(&histogram[image[i]], 1);
    }
}

__global__ void histogramKernelSM(unsigned char* image, long img_size, unsigned int* histogram) {
    __shared__ unsigned int myhist[256];
    if (threadIdx.x < 256) {
        myhist[threadIdx.x] = 0;
    }
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < img_size) {
        atomicAdd(&myhist[image[i]], 1);
    }
    __syncthreads();
    if (threadIdx.x < 256) {
        atomicAdd(&histogram[threadIdx.x], myhist[threadIdx.x]);
    }
}

void histogramCuda(unsigned char* image, long img_size, unsigned int* histogram, int hist_size, int smkernel=0) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    unsigned char* deviceImage = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceImage, img_size * sizeof(unsigned char)));
    if (deviceImage == NULL) {
        cout << "could not allocate memory!" << endl;
        return;
    }
    unsigned int* deviceHisto = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceHisto, hist_size * sizeof(unsigned int)));
    if (deviceHisto == NULL) {
        checkCudaCall(cudaFree(deviceImage));
        cout << "could not allocate memory!" << endl;
        return;
    }

    timer kernelTime1 = timer("kernelTime1");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceImage, image, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    if(smkernel){
        histogramKernelSM<<<(img_size+threadBlockSize-1)/threadBlockSize, threadBlockSize>>>(deviceImage, img_size, deviceHisto);
    }else{
        histogramKernel<<<(img_size+threadBlockSize-1)/threadBlockSize, threadBlockSize>>>(deviceImage, img_size, deviceHisto);
    }
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(histogram, deviceHisto, hist_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceImage));
    checkCudaCall(cudaFree(deviceHisto));

    cout << "histogram (kernel): \t\t" << kernelTime1  << endl;
    cout << "histogram (memory): \t\t" << memoryTime << endl;
}

void histogramSeq(unsigned char* image, long img_size, unsigned int* histogram, int hist_size) {
  int i; 

  timer sequentialTime = timer("Sequential");
  
  for (i=0; i<hist_size; i++) histogram[i]=0;

  sequentialTime.start();
  for (i=0; i<img_size; i++) {
	histogram[image[i]]++;
  }
  sequentialTime.stop();
  
  cout << "histogram (sequential): \t\t" << sequentialTime << endl;

}

void die(const char *msg)
{
    if (errno != 0) 
        perror(msg);
    else
        fprintf(stderr, "error: %s\n", msg);
    exit(1);
}


// Read pgm into unsigned char array, also does alloc for data
void readpgm_char(const char *fname, unsigned char **data,
                         long *size)
{
    char format[3];
    FILE *f;
    unsigned imgw, imgh, maxv, v;
    size_t i;

    printf("Reading PGM data from %s...\n", fname);

    if (!(f = fopen(fname, "r"))) die("fopen");

    fscanf(f, "%2s", format);
    if (format[0] != 'P' || format[1] != '2') die("only ASCII PGM input is supported");
    
    if (fscanf(f, "%u", &imgw) != 1 ||
        fscanf(f, "%u", &imgh) != 1 ||
        fscanf(f, "%u", &maxv) != 1) die("invalid input");

    if (!(*data = (unsigned char *)calloc(imgw * imgh, sizeof(char)))) die("calloc");

    for (i = 0; i < imgw * imgh; ++i)
    {
        if (fscanf(f, "%u", &v) != 1) die("invalid data");
        (*data)[i] = (v / 256);
    }
    *size = imgw * imgh;
    fclose(f);
}

void usage(const char *pname)
{
    printf("Usage: %s [OPTION]...\n"
           "  -f FILE    Read input image from FILE.\n"
           "  -s         Use the shared memory kernel\n"
           ,pname);
    exit(0);
}

int main(int argc, char* argv[]) {
    char* fname = (char*)"../../images/areas_2000x2000.pgm";
    long img_size;
    int hist_size = 256;
    int smkernel = 0;
    
    int ch;
    while ((ch = getopt(argc, argv, "f:s")) != -1)
    {
        switch(ch) {
        case 'f': fname = optarg; break;
        case 's': smkernel = 1; break;
        default: usage(argv[0]);
        }
    }

    // unsigned char *image = (unsigned char *)malloc(img_size * sizeof(unsigned char)); 
    unsigned char *image;
    readpgm_char(fname, &image, &img_size);

    unsigned int *histogramS = (unsigned int *)malloc(hist_size * sizeof(unsigned int));     
    unsigned int *histogram = (unsigned int *)malloc(hist_size * sizeof(unsigned int));

    cout << "Compute the histogram of a gray image with " << img_size << " pixels." << endl;
    if(smkernel) { cout << "Using shared memory" << endl;}

    histogramSeq(image, img_size, histogramS, hist_size);
    histogramCuda(image, img_size, histogram, hist_size, smkernel);
    
    // verify the resuls
    for(int i=0; i<hist_size; i++) {
	  if (histogram[i]!=histogramS[i]) {
            cout << "error in results! Bin " << i << " is "<< histogram[i] << ", but should be " << histogramS[i] << endl; 
            exit(1);
        }
    }
    cout << "results OK!" << endl;
     
    free(image);
    free(histogram);
    free(histogramS);         
    
    return 0;
}
