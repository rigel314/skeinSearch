#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <errno.h>
#include <assert.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <unistd.h>

#define minBYpH 16
#define BYpH 64L
#define HpT 50000L
#define TpB 128L
#define BLOCKS 15L

#define NO_MAIN
#define GPU
#include "threefish.c"

struct threadBestResult
{
	int min;
	void* addr;
};

__global__
void search(uint8_t* rands, struct threadBestResult* ans)
{
	const uint64_t target[16] = {0x8082a05f5fa94d5b,0xc818f444df7998fc,0x7d75b724a42bf1f9,0x4f4c0daefbbd2be0,0x04fec50cc81793df,0x97f26c46739042c6,0xf6d2dd9959c2b806,0x877b97cc75440d54,0x8f9bf123e07b75f4,0x88b7862872d73540,0xf99ca716e96d8269,0x247d34d49cc74cc9,0x73a590233eaa67b5,0x4066675e8aa473a3,0xe7c5e19701c79cc7,0xb65818ca53fb02f9};
	uint64_t hash[16];

	int minVal = 1024;
	int minIdx = 0;
	int idx = blockIdx.x*TpB+threadIdx.x;

	for(int i = 0; i < HpT; i++)
	{
		skeinhash1024x1024(rands+blockIdx.x*TpB*HpT*BYpH+threadIdx.x*HpT*BYpH+i*BYpH, BYpH, hash);

		int c = 0;
		for (int j = 0; j < Nw; j++)
		{
			uint64_t tmp = target[j] ^ hash[j];
			for (int k = 0; k < 64; k++)
			{
				if(tmp & 1)
					c++;
				tmp >>= 1;
			}
		}
		if(c < minVal)
		{
			minVal = c;
			minIdx = i;
		}
	}

	ans[idx].min = minVal;
	ans[idx].addr = rands+blockIdx.x*TpB*HpT*BYpH+threadIdx.x*HpT*BYpH+minIdx*BYpH;
}

__global__
void findMin(int* array)
{
	int a = blockIdx.x*TpB+threadIdx.x;
	int b = a+1;
	int ctr = 0;

	do
	{
		array[a] = min(array[a], array[b]);
		if((a > ctr) & 1)
			break;
		b = 0;
	} while(b < BLOCKS*TpB);
}

void printLog(char* msg);
void printfLog(char* fmt, ...);

int main()
{
	// uint8_t* rands;
	struct threadBestResult* answers;
	uint8_t* cuRands;
	struct threadBestResult* cuAns;
	curandStatus_t s;

	// Asserts on constants
	assert(BYpH%sizeof(int) == 0); // cuRand generator makes ints

	answers = (struct threadBestResult*) malloc(BLOCKS*TpB*sizeof(struct threadBestResult)); assert(answers != NULL);

	// The default is to spin the CPU while waiting in cudaDeviceSynchronize() (for synchronous CUDA kernel calls)
	// Spin loops are bullshit, so this tells it to sleep instead.
	// Also, device flags have to be set before any other CUDA calls.
	// I should probably care about using events, but I won't: https://devtalk.nvidia.com/default/topic/755859/cpu-core-is-busy-while-gpu-runs-its-kernel/
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); assert(cudaGetLastError() == cudaSuccess);

	// allocate everything.
	cudaMalloc((void**)&cuRands, BLOCKS*TpB*HpT*BYpH); assert(cudaGetLastError() == cudaSuccess);
	cudaMalloc((void**)&cuAns, BLOCKS*TpB*sizeof(struct threadBestResult)); assert(cudaGetLastError() == cudaSuccess);

	FILE* fp = fopen("/dev/urandom", "r");
	uint64_t seed;
	int ret = fread(&seed, sizeof(uint64_t), 1, fp);
	fclose(fp);

	printLog((char*)"generating/transferring random data for computation");

	// All of the cuRand stuff.
	curandGenerator_t gen;
	s = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW); assert(s == CURAND_STATUS_SUCCESS);
	s = curandSetPseudoRandomGeneratorSeed(gen,seed); assert(s == CURAND_STATUS_SUCCESS);
	s = curandGenerate(gen, (unsigned int*) cuRands, BYpH/4*HpT*TpB*BLOCKS); assert(s == CURAND_STATUS_SUCCESS);
	cudaDeviceSynchronize(); assert(cudaGetLastError() == cudaSuccess);
	s = curandDestroyGenerator(gen); assert(s == CURAND_STATUS_SUCCESS);
	
	printLog((char*)"starting search");
	search<<<BLOCKS, TpB>>>(cuRands, cuAns); assert(cudaGetLastError() == cudaSuccess);
	// Wait for kernel to finish.
	cudaDeviceSynchronize(); assert(cudaGetLastError() == cudaSuccess);
	printLog((char*)"search finished");

	// get the results from all threads
	cudaMemcpy(answers, cuAns, BLOCKS*TpB*sizeof(struct threadBestResult), cudaMemcpyDeviceToHost); assert(cudaGetLastError() == cudaSuccess);
	cudaFree(cuAns); assert(cudaGetLastError() == cudaSuccess);

	printLog((char*)"finding best match");
	int min = 1024;
	int lowestI = -1;
	for(int i = 0; i < BLOCKS*TpB; i++)
	{
		if(answers[i].min < min)
		{
			min = answers[i].min;
			lowestI = i;
		}
	}

	uint8_t bestMatch[BYpH];
	cudaMemcpy(&bestMatch, answers[lowestI].addr, BYpH, cudaMemcpyDeviceToHost); assert(cudaGetLastError() == cudaSuccess);
	printfLog((char*)"best match(%d incorrect bits) (hash index %d) (pointer offset %p):", min, lowestI, (uint8_t*)answers[lowestI].addr - cuRands);
	for(int i = 0; i < BYpH; i++)
	{
		printf("%02x ", bestMatch[i]);
	}
	printf("\n");
	// printMsg(answers[lowestI].hash);

	cudaFree(cuRands); assert(cudaGetLastError() == cudaSuccess);
	free(answers);

	return EXIT_SUCCESS;
}

void printLog(char* msg)
{
	time_t now;
	struct tm* 	lcltime;

	now = time(NULL);
	lcltime = localtime(&now);
	FILE* fp = stdout;
	fprintf(fp, "%d-%02d-%02d %02d:%02d:%02d ~ %s\n", lcltime->tm_year + 1900, lcltime->tm_mon + 1, lcltime->tm_mday, lcltime->tm_hour, lcltime->tm_min, lcltime->tm_sec, msg);
	fflush(stdout);
}

void printfLog(char* fmt, ...)
{
	char* msg;

	va_list ap;
	va_start(ap, fmt);

	msg = (char*) malloc(vsnprintf(NULL, 0, fmt, ap) + 1);
	va_end(ap);
	if(!msg)
	{
		printLog(strerror(errno));
		return;
	}

	va_start(ap, fmt); // The vsnprintf call clobbered my va_list.  So starting it again.

	vsprintf(msg, fmt, ap);
	printLog(msg);

	free(msg);
	va_end(ap);
}
