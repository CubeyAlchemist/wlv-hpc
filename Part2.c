/* 
* Module: High Performance Computing (6CS005)
* Assessment: Task 2 - Password Cracking using Posix Threads
* Author: Adam Scatchard (2113690)
*
* This program uses a globally defined SHA-512 encrypted password and Posix threads
* to attempt to brute-force crack the password, and print the result to the terminal.
* Due to the nature of these encrypted passwords, I chose to define them in the code
* rather than requiring the user enter it correctly in CLA, especially given punctuation.
* To change the password, either switch out one of the commented lines, or alter the
* string value. Current string values have their unencrypted strings as comments at the
* end of the line.
*
* Task Requirements:
* (75) Crack password using multithreading & dynamic slicing based on thread count
* (25) Program finishes when password has been found.
*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#define __USE_GNU
#include "./crypt.h"
#define SALT "$6$AS$"

/* Globals */
int cracked = 1;
char password[5] = "NULL\0";
//char *encryptedPassword = "$6$AS$X/ohgGiyqvcl7ODpuXjKTL9OG6gZAaP2RQyD32qFzVneMrhfB/G9f6POl9FY1m/ERvWEfm9GvrnjWwjYu2AXH."; /* AA69 */
//char *encryptedPassword = "$6$AS$3zw4b627wBTGMmjuqlB3zxZUCsibT9dXT3HKu.Ws.Rsp5WV2ZLmMABpQdyV7c4pgeP85ciDiZZeAYeoNgVpVo.": /* AJ91 */
//char *encryptedPassword = "$6$AS$xXlHZZ3MV.a/4P/sjtU275uBYMPu./cOdyE3x74wGwzqmsbWS1sxU9o/GlpjiqsaptHAanhaNR70D2ZLzalmL1"; /* AX28 */
char *encryptedPassword = "$6$AS$Ig.vW9RG9J5gPFUvHwyV67GdVVndF.2ROH6.qZjQN1Nm5kqn0t/FKNf4.48qRHdyAWwIQOtKkCosTrwyj3SvJ."; /* HP93 */

/* Structs */
typedef struct Data {
    int threadID;                   // Unique thread ID num
    int numThreads;                 // Total threads spawned
    int maxPossible;                // Maximum possible password combinations
    char candidatePassword[5];      // Thread-specific calculated password (pre-encrypt)
    char *salt;                     // Encryption salt
    struct crypt_data *cryptData;   // Thread-specific crypt_data struct required for thread safety
} Data;

/* Prototypes */
void *crackThread(void *input);
void substring(char *dest, char *src, int start, int length);

/* Main */
int main (int argc, char *argv[]){
    int i;                      // Generic counter
    int requestedThreads;       // Requested thread count, accounting for bad values
    int possiblePasswords;      // Maximum possible password combinations
    Data *threadData;           // Pointer to array of threadData structs
    pthread_t *threadIDs;       // Pointer to array of pthread_t's
    char salt[7];               // Extracted salt from encrypted password to crack

    /* Process CLA's */
    if (argc < 2 || argc > 3) {
        printf("Incorrect number of arguments. Expected 2, Received %d\n", argc);
        printf("./Part2 <#Threads>\n");
        return 1;
    }
    requestedThreads = atoi(argv[1]);
    

    /* Sanity-check requested thread count */
    if (requestedThreads > 1000){
        requestedThreads = 1000;
    } else if (requestedThreads <= 0) {
        requestedThreads = 1;
    }


    /* Calculate consistent variables */
    possiblePasswords = 26 * 26 * 10 * 10;  // 67,600
    substring(salt, encryptedPassword, 0, 6);
    

    // Create 'n' threads as specified previously
        // Threads will operate based on filters rather than slices
    threadData = malloc(sizeof(Data) * requestedThreads);
    threadIDs = malloc(sizeof(pthread_t) * requestedThreads);
    for (int i = 0; i < requestedThreads; i++) {
        threadData[i].threadID = i;
        threadData[i].numThreads = requestedThreads;
        threadData[i].maxPossible = possiblePasswords;
        threadData[i].salt = &salt[0];
        threadData[i].cryptData = malloc(sizeof(struct crypt_data));
        threadData[i].cryptData->initialized = 0;
    }


    // Create threads
    printf("\n");
    for (int i = 0; i < requestedThreads; i++) {
        pthread_create(&threadIDs[i], NULL, crackThread, (void*)&threadData[i]);
    }
    // Rejoin all threads
    for (int i = 0; i < requestedThreads; i++){
        pthread_join(threadIDs[i], NULL);
    }


    // Output result to terminal
    printf("Found password: %s\n", password);


    // Free memory
    for (int i = 0; i < requestedThreads; i++) {
        free(threadData[i].cryptData);
    }
    free(threadData);
    free(threadIDs);

    // Return Success
    return 0;
}

void substring(char *dest, char *src, int start, int length){
    memcpy(dest, src + start, length);
    *(dest + length) = '\0';
}

void *crackThread(void *input) {
    Data *data = (Data*)input;
    int currentPassID = data->threadID;
    char char1, char2, char3, char4;
    char *encryptedCandidate;

    // Starting Password ID value
    currentPassID = data->threadID;

    while (cracked == 1 && currentPassID < data->maxPossible) {
        // Calculate password based on index
        data->candidatePassword[0] = 'A' + (currentPassID / 2600);
        data->candidatePassword[1] = 'A' + ((currentPassID/100) % 26);
        data->candidatePassword[2] = '0' + ((currentPassID/10) % 10);
        data->candidatePassword[3] = '0' + (currentPassID % 10);
        data->candidatePassword[4] = '\0';

        // Encrypt the candidate password
        crypt_r(data->candidatePassword, data->salt, data->cryptData);
        encryptedCandidate = data->cryptData->output;

        printf("Thread[%d]\t| Password Index [%06d]\t| Candidate (%s)\t| Salt(%s)\t| %s\n", data->threadID, currentPassID, data->candidatePassword, data->salt, encryptedCandidate);
        // Check if candidate encrypt matches password encrypt
        if(strcmp(encryptedPassword, encryptedCandidate) == 0) {
            printf("Found!\n");
            cracked = 0;
            memcpy(password, data->candidatePassword, 5);
            break;
        }

        currentPassID += data->numThreads;
    }
}