/* 
* Module: High Performance Computing (6CS005)
* Assessment: Task 4 - Image Blur using CUDA
* Author: Adam Scatchard (2113690)
*
* This program has been designed to accept a PNG image address through the CLA and blur
* it with a 3x3 pixel box-blur, before saving the new image to a new PNG file "output.png".
* This program utilizes CUDA to process every output pixel's component value simultaniously
* without using any locks or semaphores. This program also contains some safety checks with
* responsible memory management in the event of a predicted error.
*
* Task Requirements:
* (5)  Reading in an image file into a 1D or 2D array
* (15) Allocating & freeing correct memory on GPU based on input data
* (30) Applying box filter on image data in kernel function
* (30) Returning blurred image data from GPU to CPU
* (20) Outputting correct image with blur applied to PNG file
*/

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

// Typedefs / Structs
typedef struct CudaProperties{
    char *deviceName;
    int *computeCapability;
    int *maxBlocks;
    int *maxThreads;
    int warpSize;
}CudaProperties;

// Prototypes
void GetCUDAProperties(CudaProperties *properties);
void PrintCUDAProperties(CudaProperties *properties);
__global__ void DEV_Blur(unsigned char* dev_oldImage, unsigned char* dev_newImage);

// Main
int main (int argc, char *argv[]) {
    /* Variable Declarations */
    char *filename;
    unsigned int width, height, error;
    unsigned char *oldImageData, *newImageData;
    CudaProperties *properties;
    int numComponents;
    

    /* Process and validate CLA's */
    {
        if (argc != 2) {
            printf("This program expects 1 argument. It instead received %i.\n", argc);
            printf("Arg1: File Directory/Name (String)\n");
            return 1;
        }
        filename = argv[1];
        printf("Image Address: %s\n", filename);
    }


    /* Validate & Print CUDA compatibility */
    {
        properties = (CudaProperties*)malloc(sizeof(CudaProperties));
        GetCUDAProperties(properties);
        if (properties->computeCapability[0] == 0 && properties->computeCapability[1] == 0) {
            printf("Device does not appear to be CUDA compatible. Terminating program\n");
            return 2;
        }
        PrintCUDAProperties(properties);
    }


    /* Open and load PNG image */
    {
        error = lodepng_decode32_file(&oldImageData, &width, &height, filename);
        if (error) {
            printf("%s\n", lodepng_error_text(error));
            return error;
        }
    }
    printf("png image loaded\n");

    /* Saftey check image size vs CUDA dims */
    int CUDAError = 0;
    if (height > properties->maxBlocks[0]) {
        printf("Image height (%d) is larger than the CUDA device is capable of (%d)\n", height, properties->maxBlocks[0]);
        CUDAError++;
    }
    if (width > properties->maxBlocks[1]){
        printf("Image width (%d) is larger than the CUDA device is capable of (%d)\n", width, properties->maxBlocks[1]);
        CUDAError++;
    }
    if (CUDAError > 0) {
        printf("Aborting program. Attempting to free any allocated memory\n");
        {   // Free cudaProperties struct memory
            free(properties->deviceName);
            free(properties->maxBlocks);
            free(properties->maxThreads);
            free(properties->computeCapability);
            free(properties);
            printf("\tCudaProperties freed\n");
        }
        {   // Free image array memory
            free(oldImageData);
            printf("\toldImageData freed\n");
        }
        printf("All memory freed. Closing\n");
        return 1;
    }

    /* Prepare memory for edited image data */
    numComponents = width * height * 4;
    newImageData = (unsigned char*)malloc(sizeof(unsigned char) * numComponents);
    for (int i = 0; i < width * height * 4; i++) {
        newImageData[i] = 0;
    }

    /* Prepare GPU variables and allocate memory */
    unsigned char* dev_oldImage,* dev_newImage;
    cudaMalloc((void**) &dev_oldImage, (sizeof(unsigned char) * numComponents));
    cudaMalloc((void**) &dev_newImage, (sizeof(unsigned char) * numComponents));
    printf("GPU memory allocated\n");

    cudaMemcpy(dev_oldImage, (void*)oldImageData, (sizeof(unsigned char) * numComponents), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_newImage, (void*)newImageData, (sizeof(unsigned char) * numComponents), cudaMemcpyHostToDevice);
    printf("Memory copied to GPU\n");


    /* Run kernel function */
    printf("Starting Kernal function\n");
    DEV_Blur<<<dim3(height, width, 1), dim3(4, 1, 1)>>>(dev_oldImage, dev_newImage);
    printf("Kernal function closed\n");


    /* Retrieve data from GPU and free GPU memory */
    cudaMemcpy(newImageData, dev_newImage, (sizeof(unsigned char) * numComponents), cudaMemcpyDeviceToHost);
    printf("Memory copied from GPU\n");

    cudaFree(dev_oldImage);
    cudaFree(dev_newImage);
    printf("GPU memory freed\n");


    /* Pass edited image data to lodePNG library to re-encode */
    error = lodepng_encode32_file("output.png", newImageData, width, height);
    if (error) {
        printf("%s\n", lodepng_error_text(error));
        return error;
    }
    printf("Outputting PNG file\n");


    /* Free any remaining CPU memory */
    printf("Freeing remaining memory\n");
    {   // Free cudaProperties struct memory
        free(properties->deviceName);
        free(properties->maxBlocks);
        free(properties->maxThreads);
        free(properties->computeCapability);
        free(properties);
        printf("\tCudaProperties freed\n");
    }
    {   // Free image array memory
        free(oldImageData);
        free(newImageData);
    }
    printf("\tImageData pointer freed\n");

    return 0;
}

// Host functions
void GetCUDAProperties(CudaProperties* properties){
	properties->deviceName = (char*)malloc(sizeof(char) * 256);
	properties->maxBlocks = (int*)malloc(sizeof(int) * 3);
	properties->maxThreads = (int*)malloc(sizeof(int) * 3);
	properties->computeCapability = (int*)malloc(sizeof(int) * 2);
	
	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
	strcpy(properties->deviceName, prop.name);
	properties->maxBlocks[0] = prop.maxGridSize[0];
	properties->maxBlocks[1] = prop.maxGridSize[1];
	properties->maxBlocks[2] = prop.maxGridSize[2];
	properties->maxThreads[0] = prop.maxThreadsDim[0];
	properties->maxThreads[1] = prop.maxThreadsDim[1];
	properties->maxThreads[2] = prop.maxThreadsDim[2];
	properties->warpSize = prop.warpSize;
	properties->computeCapability[0] = prop.major;
	properties->computeCapability[1] = prop.minor;
}

void PrintCUDAProperties(CudaProperties* properties){
	printf("DeviceName: %s\n", properties->deviceName);
    printf("Compute Capability: %d.%d\n", properties->computeCapability[0], properties->computeCapability[1]);
    printf("Max Grid Size: [%d,%d,%d]\n", properties->maxBlocks[0],properties->maxBlocks[1],properties->maxBlocks[2]);
    printf("Max Threads Dim: [%d,%d,%d]\n", properties->maxThreads[0],properties->maxThreads[1],properties->maxThreads[2]);
	printf("Warp Size: %d\n", properties->warpSize);
}

// __Global__ functions
__global__ void DEV_Blur(unsigned char* dev_oldImage, unsigned char* dev_newImage) {
    /* Kernel block/thread structure : Grid(height, width, 1) x Block(4, 1, 1) */

    /* Kernel function design:
    * Image data is provided as a single-dimension array of u-chars.
    * The Block X/Y and Thread X values define the row, col, and component index respectively 
    * The blur mask is defined as a 3x3 pixel grid (+/-1 from the thread's index)
    * Multiple reads can occur at any location on the oldImage without issue
    * Each write of the newImage is always in a thread-index unique location ... a single u-char index
    * Limitations: Image size limited to grid dimension X and Y, dependent on CUDA Compute Capability
    * */

    int height = gridDim.x;
    int width = gridDim.y;
    int row = blockIdx.x;
    int col = blockIdx.y;
    int com = threadIdx.x;
    int index = (row * width * 4) + (col * 4) + com;
    int componentValue = 0, count = 0;

    /* Loop through blur mask using index offsets */
    for (int r = -1; r < 2; r++) {
        for (int c = -1; c < 2; c++){
            int maskRow = row + r, maskCol = col + c;
            if (maskRow < 0 || maskRow >= height) {
                continue;
            }
            if (maskCol < 0 || maskCol >= width) {
                continue;
            }
            componentValue += dev_oldImage[index + (r * width * 4) + (c * 4)];
            count++;
        }
    }

    /* Compute and store component average value in newImage array */
    componentValue = int(componentValue / count);
    dev_newImage[index] = componentValue;
}