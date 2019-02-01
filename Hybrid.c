#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pthread.h>

double globalResult = 0;	
pthread_mutex_t mutex;
double** matrixA;
double** matrixB;
int columnsA, columnsB;

typedef struct task{
	int start;
	int end;
	double** matrix; // this is aPart
} task;

/**
	This method allows to store the values that are passed by parameter when executing the program
	int* seed: the seed used to generate random values
	int* rowsA, columnsA, columnsA, columnsB: sizes of rows and columns
**/
void readParameters(char* argv[], int* seed, int* rowsA, int* columnsA, int* columnsB, int* nThreads){
  *seed = atoi(argv[1]);
  *rowsA = atoi(argv[2]);
  *columnsA = atoi(argv[3]);
  *columnsB = atoi(argv[4]);
	*nThreads = atoi(argv[5]);
}

/**
	This method allows to initialise one matrix with random double values from -40 to 40
	double** matrix: the matrix that we want to initialise
	int rows: number of rows
	int columns: number of columns
**/
void initialiseMatrix(double** matrix, int rows, int columns){
	int i, j;
	for(i = 0; i<rows; ++i){
		for(j = 0; j<columns; ++j){
            matrix[i][j] = ((double)rand() * (40 - (-40) )) / (double)RAND_MAX + (-40);
		}
	}
}

/**
	This method allows to show the values of one matrix
	double** matrix: the matrix that we want to show
	int rows: number of rows
	int columns: number of columns
**/
void showMatrix(double** matrix, int rows, int columns){
	int i, j;
	for(i = 0; i<rows; ++i){
		for(j = 0; j<columns; ++j){
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
	}
}
/**
 * Calculates the sum of elements from matrixC, which is obtained from multiply Matrix A and B
 * double** matrixA, matrixB : matrices that are multiplied.
 * int columnsA, columnsB : number of columns from matrices A and B
 * int rowsA : number of rows from matrix A
 */
double sequentialMultiply(double** matrixA, double** matrixB, int rowsA, int columnsA, int columnsB){    
  double sum = 0.0;
  for(int i = 0; i < rowsA; ++i){	
		for(int j = 0; j < columnsA; ++j){
    	for(int k = 0; k < columnsB; ++k){
        sum += matrixA[i][j]*matrixB[j][k];
      }
    }
	}
  printf("Valor del método secuencial = %f\n", sum);
  return sum;
}
/**
 * Balances the work between all the processes
 * int length: length of the vector, which has to be divided
 * int world_size: number of process, between the task are balanced
 * int* sendCounts: this vector contains the number of tasks for each process
 * int* displs: this vector shows where each process will begin to work
**/
void balance(int rows, int columns, int world_size, int* sendCounts, int* displs){
  int pack = rows / world_size;
  int offset = rows % world_size;
	int i;
  for(i = 0; i<world_size; ++i){
    displs[i] = (i*pack + (i<offset?i:offset)) * columns; // Where each process starts
    sendCounts[i] = (pack + (i<offset)) * columns;
  }
}
/**
   Función hebrada
**/
void* threadsMultiply (void* args){
	task* work = (task*) args;
	double local = 0.0;
	int i, j, k;
	for(i = work->start; i < work->end ;++i){
		for(j=0; j < columnsA ; ++j){
			for (k = 0 ; k < columnsB	 ; ++k){
				local += work->matrix[i][j]*matrixB[j][k]; // work->matrix[i][j] is aPart 
			}
		}
	}
	//Mutex access
	pthread_mutex_lock(&mutex);
	globalResult+=local;
	pthread_mutex_unlock(&mutex);
	return 0;
}

int main(int argc, char* argv[]){

  // Declaration of variables
  int world_rank, world_size,i, j, k, rowsA, seed, nThreads, pack, offset;
  double t1, t2, result;

  // MPI Initialisation
  MPI_Init(&argc, &argv);

  // Start the timer
  t1 = MPI_Wtime();

  // Get the number of MPI processes and the rank of this process
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the values from program arguments
  result = 0;
  rowsA = 0;
  columnsA = 0;
  columnsB = 0;
  seed = 0;
  nThreads = 0; 

	// Check the number of params
	if(argc!=6){
		if(world_rank==0){
			printf("Error, los parametros indicados no son correctos : seed rowsA columnsA columnsB numThreads\n");
		}
    MPI_Finalize();
    return 0;
	}
  readParameters(argv, &seed, &rowsA, &columnsA, &columnsB, &nThreads);
  srand(seed);

  // Memory Allocation
  matrixA = malloc(sizeof(double*)*rowsA);
  double* matrixA_container = malloc(sizeof(double)*columnsA*rowsA);
  matrixB = malloc(sizeof(double*)*columnsA);
	double* matrixB_container = malloc(sizeof(double)*columnsB*columnsA);

  for(i = 0; i<rowsA; ++i){
    matrixA[i] = &(matrixA_container[i*columnsA]); // Save the start of each row
  }
  for(i = 0; i<columnsA; ++i){
    matrixB[i] = &(matrixB_container[i*columnsB]); // Save the start of each row
  }

  int* aCounts=(int*)malloc(world_size*sizeof(int));
  int* aDispls=(int*)malloc(world_size*sizeof(int));

	// Reservamos memoria en cada proceso para los threads y sus estructuras
	task* work = malloc(sizeof(task)*nThreads);
	pthread_t* threads = malloc(sizeof(pthread_t)*nThreads);

	// Mutex init
	pthread_mutex_init(&mutex, 0);

	// Proceso Raíz
  if(world_rank==0){
    // Matrix initialisation
  	initialiseMatrix(matrixA,rowsA,columnsA);
    initialiseMatrix(matrixB,columnsA,columnsB);
		// Show matrices
		if(rowsA*columnsA < 20 && columnsA*columnsB < 20){ 
				printf("Matrix A\n");
        showMatrix(matrixA, rowsA, columnsA); 
        printf("-----------------------\n"); 
				printf("Matrix B\n");
        showMatrix(matrixB, columnsA, columnsB); 
				printf("-----------------------\n");
    }
    // Check the sequential result
    //sequentialMultiply(matrixA, matrixB, rowsA, columnsA, columnsB);
	}
	
	// Send matrix B to all the processes
	MPI_Bcast(&matrixB[0][0],columnsA*columnsB,MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Balance the work between processes
	balance(rowsA, columnsA, world_size, aCounts,aDispls); 

  /* Vector where MPI_Scatterv() result is received*/
	double** aPart = malloc(sizeof(double*)*(aCounts[world_rank]/columnsA));
	double* aPart_container = malloc(sizeof(double)*(aCounts[world_rank]));
  	for(i = 0; i<(aCounts[world_rank]/columnsA); ++i){
    	aPart[i] = &(aPart_container[i*columnsA]); // Save the start of each row
  }
  // Communication
  // Se distribuyen las filas de A
  MPI_Scatterv(&matrixA[0][0],aCounts,aDispls,MPI_DOUBLE,&aPart[0][0],aCounts[world_rank],MPI_DOUBLE,
		    0, MPI_COMM_WORLD);

  // Each process do his task 
	// Now each process balance the work between the threads
	pack = ((aCounts[world_rank]/columnsA) / nThreads);
	offset = ((aCounts[world_rank]/columnsA) % nThreads);
	for(i = 0; i<nThreads; ++i){	
		work[i].start = (i*pack + (i<offset?i:offset)); // Where each process starts
		work[i].end = work[i].start + (pack + (i<offset));
		work[i].matrix = aPart;
		pthread_create(&threads[i], 0, threadsMultiply, &work[i]);
	}
	
	// Clean threads
	for(i = 0; i<nThreads; ++i){
		pthread_join(threads[i], 0);
	}
	
	// Retrieve the result from each process and obtain the result
	MPI_Reduce(&globalResult, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
  // Process 0 prints the result
	if (world_rank == 0) {
		printf("El resultado del cálculo en paralelo es: %f\n", result);
  }  

	// Release memory
  free(matrixA[0]);
	free(matrixA);
  free(matrixB[0]);
  free(matrixB);
	free(aCounts);
  free(aDispls);
	free(aPart[0]);
	free(aPart);
	free(work);
	free(threads);
	
	// Stop timer
  t2 = MPI_Wtime();

	// Process 0 prints the time
  if (world_rank == 0) {
    printf("El tiempo de ejecución es: %f\n", t2-t1);
  }

  //End of each proccess execution
  MPI_Finalize();

	
}

