/* 
* Module: High Performance Computing (6CS005)
* Assessment: Task 1 - Matrix Multiplication
* Author: Adam Scatchard (2113690)
*
* This program has been designed to read matrix pairs from a supplied text file (CLA)
* calculate the dot product of the two matrices if they are compatible using appropriate
* multithreading, and putting the result to a new text file "matrices_output.txt" in the
* local directory. An example matrix input text file has been supplied with this program
* 
* Task Requirements:
* (20) Read data from file appropriately
*       Read data in pairs to minimize memory requirments
* (10) Use dynamic memory allocation for matrix A and B
*       Use malloc/free for matrix A and B, plus other structs
* (20) Create an algorithm to multiply matrices correctly
*       Algorithm computes dot product correctly
* (30) Use multithreading with equal computations
*       Threading based on output matrix indices. Thread calculation deviation <1 average
* (20) Storing output matrices to output file in correct format
*       Ouput saved to text file in same format as input files
*/

/* Includes */
#include <stdio.h>      /* printf(), fopen(), fclose(), fscanf(), fprintf(), FILE* */
#include <stdlib.h>     /* atoi(), malloc(), free() */
#include <errno.h>      /* errno, perror() */
#include <pthread.h>    /* pthread_t, pthread_create(), pthread_join(), pthread_exit() */

/* Globals */
const char* OUTPUT_FILENAME = "matices_output.txt";
const int THREAD_DEFAULT = 20;

/* Structs */
typedef struct matrix{
    int rows;
    int cols;
    double **values;
} matrix;

typedef struct matrixSet{
    matrix *A;
    matrix *B;
    matrix *C;
} matrixSet;

typedef struct threadStruct{
    matrixSet *ms;
    pthread_t pthreadID;
    int threadNum;
    int threadCount;
}threadStruct;

/* Prototypes */
void ReadMatrix(FILE *file, matrix *m);
void SaveMatrix(FILE *file, matrix *m);
void CalculateDotProduct(matrixSet *ms);
void MatrixFree(matrix *m);
void *CalculateDotProductThreaded(void *object);
void PrepareThreads(int *threadCount, int *threadLimit, threadStruct **threadStructs, matrixSet **ms);
void PrintThreadData(threadStruct *d);
void PrintMatrix(matrix *m);




/* Main */
int main(int argc, char **argv) {
    // Main variable assignments
    FILE *inputFile, *outputFile;
    matrixSet *mSet;
    errno = 0;
    threadStruct *threadStructs;
    int threadLimit;
    int threadCount;


    // Validate CLA's
    if (argc != 3) {
        printf("This program expects 2 command-line arguments. Received %d.\n", argc);
        printf("Please re-run this program with the following arguments\n");
        printf("<Program> <InputFileAddress> <Threads>\n");
        return 1;
    }

    // Sanity-check requested thread count
    threadLimit = atoi(argv[2]);
    printf("ThreadLimit = %d\n", threadLimit);
    if (threadLimit <= 0 || threadLimit > 1000) {
        printf("Thread limit of %d is unreasonable. Setting thread limit to %d.\n", threadLimit, THREAD_DEFAULT);
        threadLimit = THREAD_DEFAULT;
    }

    // Open matrix file, and create output matrix file
    inputFile = fopen(argv[1], "r");
    if (errno != 0) {
        perror("Error occured when trying to open the input file\n");
        return 2;
    }
    outputFile = fopen(OUTPUT_FILENAME, "w");

    // Malloc initial variables
    mSet = malloc(sizeof(matrixSet));

    // While file contains matrix pairs
    while (!feof(inputFile)){
        printf("Attempting to read in matrix pair\n\n");

        // Allocate space for A and B input matrices
        mSet->A = malloc(sizeof(matrix));
        mSet->B = malloc(sizeof(matrix));

        // Read and store a pair of matrices
        ReadMatrix(inputFile, mSet->A);
        ReadMatrix(inputFile, mSet->B);

        // Print input matrix pairs to terminal
        PrintMatrix(mSet->A);
        PrintMatrix(mSet->B);

        // Validate matrix math
        if (mSet->A->cols != mSet->B->rows){
            printf("Matrix A and B are incompatible: A.Col(%d), B.Row(%d)\n", mSet->A->cols, mSet->B->rows);
            continue;
        }
        printf("Matrix A and B are compatible: Beginning calculation process\n\n");

        // Allocate space for result matrix
        mSet->C = malloc(sizeof(matrix));
        mSet->C->rows = mSet->A->rows;
        mSet->C->cols = mSet->B->cols;
        mSet->C->values = malloc(sizeof(double*) * mSet->C->rows);
        for (int i = 0; i < mSet->C->rows; i++) {
            mSet->C->values[i] = malloc(sizeof(double) * mSet->C->cols);
        }

        // Allocate and initialise threadStructs
        PrepareThreads(&threadCount, &threadLimit, &threadStructs, &mSet);

        // Create and Start threads
        for (int i = 0; i < threadCount; i++) {
            pthread_create(&threadStructs[i].pthreadID, NULL, CalculateDotProductThreaded, &threadStructs[i]);
        }

        // Wait and Rejoin all threads
        for (int i = 0; i < threadCount; i++) {
            pthread_join(threadStructs[i].pthreadID, NULL);
        }

        // Print result matrix
        printf("Result matrix C calculated\n\n");
        PrintMatrix(mSet->C);

        // Save result to output file
        SaveMatrix(outputFile, mSet->C);

        // Free local allocated memory
        MatrixFree(mSet->A);
        MatrixFree(mSet->B);
        MatrixFree(mSet->C);
        free(threadStructs);
    }

    // Close both matrix files
    fclose(inputFile);
    fclose(outputFile);

    // Free matrixSet memory
    free(mSet);

    return 0;
}

/* Function - ReadMatrix
* Takes a file pointer and matrix structure pointer as input.
* Reads comma-separated row and column value from file.
* Reads comma-separated matrix values from file.
* Allocates memory for matrix values as approriate.
*/
void ReadMatrix(FILE *file, matrix *m){
    fscanf(file, "%d,%d", &m->rows, &m->cols);

    m->values = malloc(sizeof(double*) * m->rows);

    for (int i = 0; i < m->rows; i++) {
        m->values[i] = malloc(sizeof(double) * m->cols);

        for (int j = 0; j < m->cols-1; j++) {
            fscanf(file, "%lf,", &m->values[i][j]);
        }
        fscanf(file, "%lf\n", &m->values[i][m->cols-1]);
    }
}

/* Function - SaveMatrix
* Takes a file pointer and matrix structure pointer as input.
* Prints to file the row and column value from the supplied matrix.
* Prints to file the matrix values, separating by comma and newline as appropriate.
*/
void SaveMatrix(FILE *file, matrix *m){
    fprintf(file, "%d,%d\n", m->rows, m->cols);

    for (int i = 0; i < m->rows; i++){
        for (int j = 0; j < m->cols; j++){
            fprintf(file, "%lf", m->values[i][j]);
            if(j != m->cols-1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");
}

/* Function - PrintMatrix
* Takes a reference to a matrix structure as input
* Helper function to print the contents of a matrix structre to terminal in a
* convenient and readable format
*/
void PrintMatrix(matrix *m) {
    printf("Matrix: (%d, %d)\n", m->rows, m->cols);

    for (int i = 0; i < m->rows; i++){
        for (int j = 0; j < m->cols; j++){
            printf("%lf", m->values[i][j]);
            if (j < m->cols-1) {
                printf(",");
            }
        }
        printf("\n");
    }
    printf("\n");
}

/* Function - PrepareThreads
* Takes a reference to; threadCount, threadLimit, *threadStructs, *mSet as inputs
* Calculates the appropriate number of threads to use for the upcoming threaded operation,
* Allocates space for those threads in *threadStructs, and initializes their variables.
*/
void PrepareThreads(int *threadCount, int *threadLimit, threadStruct **threadStructs, matrixSet **ms){
    // Calculate appropriate threadCount for upcoming threaded operation
    *threadCount = (*ms)->A->rows * (*ms)->B->cols;
    if (*threadCount > *threadLimit) {
        *threadCount = *threadLimit;
    }
    
    // Allocate memory for *threadStructs reference according to threadCount
    *threadStructs = malloc(sizeof(threadStruct) * (*threadCount));
    
    // Initialize *threadStructs data
    for (int i = 0; i < *threadCount; i++){
        (*threadStructs)[i].ms = *ms;
        (*threadStructs)[i].pthreadID = i;
        (*threadStructs)[i].threadNum = i;
        (*threadStructs)[i].threadCount = *threadCount;
    }
}

/* Threaded Function - CalculateDotProduct - MT-Safe
* Takes a void-cast threadStruct object as input
* Uses references within threadStruct object to selectively isolate indices
* in the output matrix (C) and calculate the dot product from the input (A)
* and (B) matrices.
* Thread Safe: Data is only written to 'C->values[n][m]'. When run in this
* program, no collisions are possible, and only one thread can write to a
* single index space at any time.
*/
void *CalculateDotProductThreaded(void *object){
    // Recast object
    threadStruct *data = (threadStruct*)object;

    // Redefine variables for convenience
    matrix *A = data->ms->A;
    matrix *B = data->ms->B;
    matrix *C = data->ms->C;
    int maxIndex = C->rows * C->cols;
    int threadNum = data->threadNum;
    int threadCount = data->threadCount;

    // Loop through indexes starting at threadNum, incrementing by threadCount
    for (int index = threadNum; index < maxIndex; index += threadCount){
        
        // Precalculation values
        int row = index / C->cols;
        int col = index % C->cols;
        C->values[row][col] = 0;

        // Do calculation
        for (int i = 0; i < A->cols; i++){
            C->values[row][col] += A->values[row][i] * B->values[i][col];
        }
    }

    pthread_exit(NULL);
}

/* Function - MatrixFree
* Takes a matrix structure pointer as input.
* Frees all related allocated memory, including the matrix structure itself.
*/
void MatrixFree(matrix *m) {
    for(int i = 0; i < m->rows; i++){
        free(m->values[i]);
    }
    free(m->values);
    free(m);
}