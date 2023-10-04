#include <iostream>
#include <cstring>
#include <mpi.h>

#define K 1024            /* One Kilobyte */
#define M K*K             /* One Megabyte */
#define MAXSIZE K*M       /* One Gigabyte */

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int numProcesses, my_id;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

    if (numProcesses < 2) {
        std::cerr << "Error: Number of processes must be at least 2." << std::endl;
        MPI_Finalize();
        return 1;
    }

    // Allocate memory size big enough to hold the maximum message size that the user inputs
    char* message = new char[MAXSIZE];
    /* Check that allocation succeeded */
    if (message == NULL) {
        std::cerr << "Could not allocate memory, exiting" << std::endl;
        MPI_Finalize();
        exit(0);
    }

    double startTime, endTime, totalTime;
    int next_proc = (my_id + 1) % numProcesses;                    // next process in the ring 
    int prev_proc = (my_id - 1 + numProcesses) % numProcesses;     // previous process in the ring
    // Size of message to be sent around the ring. Must be specified by user. Program ends if value is 0.
    int messageSize = 0;
    // Loop until size of message sent is 0
    while (true) {
        if (my_id == 0) {
            std::cout << "Enter message size (0 to exit):" << std::endl;
            fflush(stdout);
            std::cin >> messageSize;
            if (messageSize > MAXSIZE) {
                messageSize = 0;
                std::cout << "Message size is too large! Maximum value is " << MAXSIZE << " bytes" << std::endl;
            }
            // Broadcast message size to all other processes
            for (int proc_id = 1; proc_id < numProcesses; proc_id++) {
                MPI_Send(&messageSize, 1, MPI_INT, proc_id, 0, MPI_COMM_WORLD);
            }
            if (messageSize <= 0) {
                break;
            }
            // Only initialize the size of the message buffer to be sent, to some random value (in this case 72, the ascii value of the letter 'H'). 
            memset(message, 72, messageSize * sizeof(char));  
            // Start timer
            startTime = MPI_Wtime();
            // Send message of size 'messageSize' to the next process in the ring (in this case, process 1)
            MPI_Send(message, messageSize, MPI_CHAR, next_proc, 0, MPI_COMM_WORLD);
            // Receive message back from the  previous process (in this case, the last process in the ring, numProcesses-1)
            MPI_Recv(message, messageSize, MPI_CHAR, prev_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Stop timer
            endTime = MPI_Wtime();
            // Calculate transfer time
            totalTime = endTime - startTime;
            // Print number of processes, message size and transfer time
            std::cout << "Number of processes: " << numProcesses << std::endl;
            std::cout << "Message size: " << messageSize << " bytes" << std::endl;
            std::cout << "Time taken: " << totalTime << " seconds" << std::endl;
        }
        else {
            // Any other process apart from process 0 receives the message size from process 0
            MPI_Recv(&messageSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Break out of loop if message size is  less than or equal to 0. 
            if (messageSize <= 0) {
                break;
            }
            // Recieve message from previous process in the ring
            MPI_Recv(message, messageSize, MPI_CHAR, prev_proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Send the received message to the next process in the ring
            MPI_Send(message, messageSize, MPI_CHAR, next_proc, 0, MPI_COMM_WORLD);
        }
    }
    // Deallocate the reserved memory
    delete[] message;
    MPI_Finalize();
    return 0;
}