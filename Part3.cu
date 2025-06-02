/* 
* Module: High Performance Computing (6CS005)
* Assessment: Task 3 - Password Cracking using CUDA
* Author: Adam Scatchard (2113690)
*
* This program takes in a single string argument in the form of an encrypted
* password (6 lower-case letters, 4 numbers), and uses CUDA to brute-force crack
* the password, returning the unencrypted password to the terminal.
*
* Task Requirements:
* (25) Generate encrypted password in kernel function
* (15) Allocate and free correct memory amount on GPU based on input data
* (40) Program uses >1 Block and Thread
* (20) Decrypted password is returned to CPU and printed
*/

/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Globals */

/* Structs */
typedef struct CudaProperties{
    char* deviceName;
    int* computeCapability;
    int* maxBlocks;
    int* maxThreads;
    int warpSize;
}CudaProperties;

/* Prototypes */
char* HostCrypt(const char* rawPassword);
void GetCUDAProperties(CudaProperties* properties);
void PrintCUDAProperties(CudaProperties* properties);
__global__ void CudaEntryPoint(char* deviceTargetPassword, char* deviceCrackedPassword);
__device__ void generateGuess(char* guess, int passIndex);
__device__ int deviceStringCompare(char* strA, char* strB);
__device__ void deviceStringCopy(char* strD, char* strS);
__device__ void deviceCrypt(char* rawPassword, char* newPassword);


/* Main */
int main (int argc, char** argv) {

	// Recover CLA: Target Password
	if (argc != 2) {
		printf("Error: Expected 1 arguments. Received %i\n", argc-1);
		printf("Please run this program with the following commands..\n");
		printf("./<program-name> <encrypted-password>\n");
		return 1;
	}
	char* targetPassword = (char*)malloc(sizeof(char) * 11);
	targetPassword = argv[1];
	printf("Target Password recieved: %s\n", targetPassword);

	// Get Properties of Cuda-Compatible Device
	CudaProperties* properties = (CudaProperties*)malloc(sizeof(CudaProperties));
	GetCUDAProperties(properties);

	// Print Cuda Properties (or exit if incompatible)
	if (properties->computeCapability[0] == 0 && properties->computeCapability[1] == 0) {
		printf("Device does not appear to be CUDA compatible. Terminating program\n");
		return 2;
	}
	PrintCUDAProperties(properties);

	// Host variables
	char* crackedPassword = (char*)malloc(sizeof(char) * 5);

	// GPU variables
	char* deviceTargetPassword;
	char* deviceCrackedPassword;
	
	// Allocating memory on device
	cudaMalloc((void**) &deviceTargetPassword, sizeof(char) * 11);
	cudaMalloc((void**) &deviceCrackedPassword, sizeof(char) * 5);

	// Transfer data from host to device
	cudaMemcpy(deviceTargetPassword, (void*)targetPassword, sizeof(char) * 11, cudaMemcpyHostToDevice);

	// run kernel
	CudaEntryPoint<<<dim3(26, 26, 1), dim3(10, 10, 1)>>>(deviceTargetPassword, deviceCrackedPassword);

	// Transfer memory from device to host, and free GPU memory
	cudaMemcpy(crackedPassword, deviceCrackedPassword, sizeof(char) * 5, cudaMemcpyDeviceToHost);
	cudaFree(deviceTargetPassword);
	cudaFree(deviceCrackedPassword);

	// Output cracked password to terminal
    printf("Cracked Password: %s\n", crackedPassword);

	// Free remaining CPU memory
	free(crackedPassword);
	{	// free CUDA properties struct
        free(properties->deviceName);
        free(properties->maxBlocks);
        free(properties->maxThreads);
        free(properties->computeCapability);
        free(properties);
	}

    return 0;
}

/* CPU Functions (Host) */

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

char* HostCrypt(const char* rawPassword){

	static char newPassword[11]; //use static as a local pointer should not be returned

	newPassword[0] = rawPassword[0] + 2;
	newPassword[1] = rawPassword[0] - 2;
	newPassword[2] = rawPassword[0] + 1;
	newPassword[3] = rawPassword[1] + 3;
	newPassword[4] = rawPassword[1] - 3;
	newPassword[5] = rawPassword[1] - 1;
	newPassword[6] = rawPassword[2] + 2;
	newPassword[7] = rawPassword[2] - 2;
	newPassword[8] = rawPassword[3] + 4;
	newPassword[9] = rawPassword[3] - 4;
	newPassword[10] = '\0';

	for(int i =0; i<10; i++){
		if(i >= 0 && i < 6){ //checking all lower case letter limits
			if(newPassword[i] > 122){
				newPassword[i] = (newPassword[i] - 122) + 97;
			}else if(newPassword[i] < 97){
				newPassword[i] = (97 - newPassword[i]) + 97;
			}
		}else{ //checking number section
			if(newPassword[i] > 57){
				newPassword[i] = (newPassword[i] - 57) + 48;
			}else if(newPassword[i] < 48){
				newPassword[i] = (48 - newPassword[i]) + 48;
			}
		}
	}
	return newPassword;
}
/* GPU Functions (Device) */

/* Cuda Entry Point
* Uses the dimension sizes and thread/block ID's to quickly calculate a unique
* candidate password, then sends it to be encrypted. It then compares the result
* against the target encrypted password: Only if matched, will it copy the string
* to the cracked password memory.
* 	deviceTargetPassword	- Encrypted password to crack (aaaaaa####)
* 	deviceCrackedPassword	- Cracked unencrypted password (aa##)
*/
__global__ void CudaEntryPoint(char* deviceTargetPassword, char* deviceCrackedPassword){
	
	char guess[5];
	guess[0] = 'a' + blockIdx.x;
	guess[1] = 'a' + blockIdx.y;
	guess[2] = '0' + threadIdx.x;
	guess[3] = '0' + threadIdx.y;
	guess[4] = '\0';

	char encryptedGuess[11];
	deviceCrypt(guess, encryptedGuess);

	if(deviceStringCompare(deviceTargetPassword, encryptedGuess)){
		deviceStringCopy(deviceCrackedPassword, guess);
	}
}

/* Device String Compare
* Reviews both input strings character by character. If all characters
* match, then the function returns 1, otherwise it returns 0.
*	strA	- First String
*	strB	- Second String
*/
__device__ int deviceStringCompare(char* strA, char* strB){
	
	for (int i = 0; i < 11; i++){
		if (strA[i] != strB[i]) {
			return 0;
		}
	}

	return 1;
}

/* Device String Copy
* Copies the contents of the Source string into the Destination string.
* Assumes that both strings are 5 chars long - known requirement of design.
*	strD 	- Destination String
*	strS	- Source String
*/
__device__ void deviceStringCopy(char* strD, char* strS) {
	for (int i = 0; i < 5; i++){
		strD[i] = strS[i];
	}
}

/* Device Crypt
* Takes in an unencrypted 4-character password and performs a basic
* encryption process, saving the result to the provided char* address.
* Both the input and output should be null-terminated strings.
* 	rawPassword	- Source Password (aa##)
*	newPassword - Encrypted Password (aaaaaa####)
*/
__device__ void deviceCrypt(char* rawPassword, char* newPassword) {
	newPassword[0] = rawPassword[0] + 2;
	newPassword[1] = rawPassword[0] - 2;
	newPassword[2] = rawPassword[0] + 1;
	newPassword[3] = rawPassword[1] + 3;
	newPassword[4] = rawPassword[1] - 3;
	newPassword[5] = rawPassword[1] - 1;
	newPassword[6] = rawPassword[2] + 2;
	newPassword[7] = rawPassword[2] - 2;
	newPassword[8] = rawPassword[3] + 4;
	newPassword[9] = rawPassword[3] - 4;
	newPassword[10] = '\0';

	for(int i =0; i<10; i++){
		if(i >= 0 && i < 6){ //checking all lower case letter limits
			if(newPassword[i] > 122){
				newPassword[i] = (newPassword[i] - 122) + 97;
			}else if(newPassword[i] < 97){
				newPassword[i] = (97 - newPassword[i]) + 97;
			}
		}else{ //checking number section
			if(newPassword[i] > 57){
				newPassword[i] = (newPassword[i] - 57) + 48;
			}else if(newPassword[i] < 48){
				newPassword[i] = (48 - newPassword[i]) + 48;
			}
		}
	}
}