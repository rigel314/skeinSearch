#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>

#ifndef GPU
#define __device__
#endif

// Number of words in key/plaintext
// #define Nw 4
// #define Nw 8
#define Nw 16

// Number of rounds
#if Nw == 16
	#define Nr 80
	__device__
	inline int permute(uint64_t state[Nw])
	{
		uint64_t orig[Nw];
		memcpy(orig, state, sizeof(uint64_t)*Nw);
		state[0] = orig[0];
		state[1] = orig[9];
		state[2] = orig[2];
		state[3] = orig[13];
		state[4] = orig[6];
		state[5] = orig[11];
		state[6] = orig[4];
		state[7] = orig[15];
		state[8] = orig[10];
		state[9] = orig[7];
		state[10] = orig[12];
		state[11] = orig[3];
		state[12] = orig[14];
		state[13] = orig[5];
		state[14] = orig[8];
		state[15] = orig[1];

		return 0;
	}
	__device__
	const int Rdj[8][Nw/2] = {
		{24, 13,  8, 47,  8, 17, 22, 37},
		{38, 19, 10, 55, 49, 18, 23, 52},
		{33,  4, 51, 13, 34, 41, 59, 17},
		{ 5, 20, 48, 41, 47, 28, 16, 25},
		{41,  9, 37, 31, 12, 47, 44, 30},
		{16, 34, 56, 51,  4, 53, 42, 41},
		{31, 44, 47, 46, 19, 42, 44, 25},
		{ 9, 48, 35, 52, 23, 31, 37, 20}
	};
#elif Nw == 8
	#define Nr 72
	__device__
	inline int permute(uint64_t state[Nw])
	{
		uint64_t orig[Nw];
		memcpy(orig, state, sizeof(uint64_t)*Nw);
		state[0] = orig[2];
		state[1] = orig[1];
		state[2] = orig[4];
		state[3] = orig[7];
		state[4] = orig[6];
		state[5] = orig[5];
		state[6] = orig[0];
		state[7] = orig[3];

		return 0;
	}
	__device__
	const int Rdj[8][Nw/2] = {
		{46, 36, 19, 37},
		{33, 27, 14, 42},
		{17, 49, 36, 39},
		{44,  9, 54, 56},
		{39, 30, 34, 24},
		{13, 50, 10, 17},
		{25, 29, 39, 43},
		{ 8, 35, 56, 22}
	};
#elif Nw == 4
	#define Nr 72
	__device__
	inline int permute(uint64_t state[Nw])
	{
		uint64_t orig[Nw];
		memcpy(orig, state, sizeof(uint64_t)*Nw);
		state[0] = orig[0];
		state[1] = orig[3];
		state[2] = orig[2];
		state[3] = orig[1];

		return 0;
	}
	__device__
	const int Rdj[8][Nw/2] = {
		{14, 16},
		{52, 57},
		{23, 40},
		{ 5, 37},
		{25, 33},
		{46, 12},
		{58, 22},
		{32, 32}
	};
#endif

#define Nb (Nw*8)

#ifdef DEBUG
	#define debugPrintf(...) printf(__VA_ARGS__)
	#define debugPrintMsg(var) \
		do {\
			for(int _i = 0; _i < Nw; _i++)\
			{\
				printf("%016" PRIx64 " ", var[_i]);\
			}\
			printf("\n");\
		} while (0)
#else
	#define debugPrintf(...)
	#define debugPrintMsg(...)
#endif

#define printMsg(var) \
	do {\
		for(int _i = 0; _i < Nw; _i++)\
		{\
			printf("%016" PRIx64 " ", (var)[_i]);\
		}\
		printf("\n");\
	} while (0)


__device__
uint64_t rotl(uint64_t in, int numBits)
{
	uint64_t out = in << numBits;
	uint64_t tmp = in >> (64 - numBits);
	out |= tmp;
	return out;
}

__device__
int mix(uint64_t* y0, uint64_t* y1, uint64_t x0, uint64_t x1, int d, int j)
{
	*y0 = x0+x1;
	*y1 = rotl(x1, Rdj[d%8][j]) ^ *y0;

	return 0;
}

__device__
int threefish(uint64_t key[Nw], uint64_t tweak[2], uint64_t plaintext[Nw])
{
	uint64_t subkeyTable[Nr/4+1][Nw];

	// t is one word longer than tweak, with an extra tweak value
	uint64_t t[3];
	memcpy(t, tweak, sizeof(uint64_t)*2);
	t[2] = tweak[0] ^ tweak[1];
	
	// k is one word longer than key, with an extra key word
	uint64_t k[Nw+1];
	memcpy(k, key, sizeof(uint64_t)*Nw);
	// k[Nw] = 6148914691236517205; // 2^64 / 3
	k[Nw] = 0x1BD11BDAA9FC1A22;
	// printf("%016" PRIx64 "\n", k[Nw]);
	for(int i = 0; i < Nw; i++)
		k[Nw] ^= key[i];

	// generate subkey table
	debugPrintf("Subkeys:\n");
	for(int s = 0; s < Nr/4+1; s++)
	{
		int i;
		for(i = 0; i < Nw-3; i++)
		{
			subkeyTable[s][i] = k[(s+i) % (Nw+1)];
		}
		subkeyTable[s][i] = k[(s+i) % (Nw+1)] + t[s%3];
		i++;
		subkeyTable[s][i] = k[(s+i) % (Nw+1)] + t[(s+1)%3];
		i++;
		subkeyTable[s][i] = k[(s+i) % (Nw+1)] + s;

		debugPrintMsg(subkeyTable[s]);
	}

	for(int d = 0; d < Nr; d++)
	{
		if (d % 4 == 0)
		{ // Add subkey
			for(int i = 0; i < Nw; i++)
				plaintext[i] += subkeyTable[d/4][i];
		}

		// Mix
		for(int j = 0; j < Nw/2; j++)
		{
			mix(&plaintext[2*j], &plaintext[2*j+1], plaintext[2*j], plaintext[2*j+1], d, j);
		}

		// permutation
		permute(plaintext);
	}

	for(int i = 0; i < Nw; i++)
	{
		plaintext[i] += subkeyTable[Nr/4][i];
	}

	return 0;
}

__device__
int ubi(uint64_t G[Nw], uint8_t* M, int Mlen, uint64_t Ts[2])
{
	uint64_t block[Nw];
	uint64_t pt[Nw];
	int Moffset = 0;

	uint64_t tweak[2];

	int numBlocks = Mlen/Nb + (Mlen%Nb != 0);

	for(int i = 0; i < numBlocks; i++)
	{
		int bytesThisMsg = 0;

		// Make current block from input M.
		if(Moffset + Nb < Mlen)
		{
			memcpy(block, M+Moffset, Nb);
			bytesThisMsg = Nb;
		}
		else
		{ // if M isn't an integral number of blocks
			memset(block, 0, Nb);
			bytesThisMsg = Mlen - Moffset;
			memcpy(block, M+Moffset, bytesThisMsg);
		}

		// My threefish implementation overwrites the input block with its output.
		// backing up the input block in pt.
		memcpy(pt, block, Nb);

		// Handle tweak 128bit math.
		tweak[1] = Ts[1];
		tweak[0] = Ts[0] + Moffset + bytesThisMsg;

		// // Check for carry.
		// if( (Ts[0] > Moffset + bytesThisMsg && UINT64_MAX - Ts[0] < Moffset + bytesThisMsg) ||
		// 	(Ts[0] < Moffset + bytesThisMsg && UINT64_MAX - Moffset + bytesThisMsg < Ts[0]))
		// {
		// 	tweak[1]++;
		// }

		// check first and last blocks
		if(i == 0)
			tweak[1] |= 0x4000000000000000;

		if(i == numBlocks - 1)
			tweak[1] |= 0x8000000000000000;

		// actually run the encryption
		threefish(G, tweak, block);

		// xor the output from threefish with the plaintext input.
		for(int i = 0; i < Nw; i++)
			block[i] ^= pt[i];

		// set starting value for next iteration
		memcpy(G, block, Nb);

		Moffset += Nb;
	}

	return 0;
}

__device__
int skeinhash1024x1024(uint8_t* bytes, int len, uint64_t out[Nw])
{
	uint64_t Kprime[Nw] = {0};

	uint8_t configStr[32] = "SHA3";
	*((uint16_t*)(configStr+4)) = (uint16_t)1;
	*((uint16_t*)(configStr+6)) = (uint16_t)0;
	*((uint64_t*)(configStr+8)) = (uint64_t)1024;
	configStr[16] = 0;
	configStr[17] = 0;
	configStr[18] = 0;
	for(int i = 19; i < 32; i++)
		configStr[i] = 0;

	uint64_t typeCfg[2] = {0};
	uint64_t typeMsg[2] = {0};
	uint64_t typeOut[2] = {0};
	typeCfg[1] |= ((uint64_t)4)<<56;
	typeMsg[1] |= ((uint64_t)48)<<56;
	typeOut[1] |= ((uint64_t)63)<<56;

	ubi(Kprime, configStr, 32, typeCfg);
	uint64_t* G0 = Kprime;

	ubi(G0, bytes, len, typeMsg);
	uint64_t* G1 = G0;

	uint64_t zero = 0;
	ubi(G1, (uint8_t*)&zero, 8, typeOut);
	uint64_t* H = G1;

	memcpy(out, H, Nb);
	
	return 0;
}

#ifndef NO_MAIN
int mkrand(void* out, size_t size, size_t nmemb)
{
	FILE* fp = fopen("/dev/urandom", "r");
	if(!fp)
		return 4;
	int ret = fread(out, size, nmemb, fp);
	fclose(fp);
	if(ret < size * nmemb)
		return 5;

	return 0;
}

int threefish_test()
{
	uint64_t tvkey[Nw] = {0};
	uint64_t tvtweak[2] = {0};
	uint64_t tvpt[Nw] = {0};
	threefish(tvkey, tvtweak, tvpt);
	printf("3f-tv:\n");
	printMsg(tvpt);

	uint64_t ansKey[Nw] = {0x04B3053D0A3D5CF0L, 0x0136E0D1C7DD85F7L, 0x067B212F6EA78A5CL, 0x0DA9C10B4C54E1C6L, 0x0F4EC27394CBACF0L, 0x32437F0568EA4FD5L, 0xCFF56D1D7654B49CL, 0xA2D5FB14369B2E7BL, 0x540306B460472E0BL, 0x71C18254BCEA820DL, 0xC36B4068BEAF32C8L, 0xFA4329597A360095L, 0xC4A36C28434A5B9AL, 0xD54331444B1046CFL, 0xDF11834830B2A460L, 0x1E39E8DFE1F7EE4FL};
	for(int i=0; i<Nw; i++)
	{
		assert(tvpt[i]==ansKey[i]);
	}

	uint64_t tv2key[Nw] = {0x1716151413121110,0x1f1e1d1c1b1a1918,0x2726252423222120,0x2f2e2d2c2b2a2928,0x3736353433323130,0x3f3e3d3c3b3a3938,0x4746454443424140,0x4f4e4d4c4b4a4948,0x5756555453525150,0x5f5e5d5c5b5a5958,0x6766656463626160,0x6f6e6d6c6b6a6968,0x7776757473727170,0x7f7e7d7c7b7a7978,0x8786858483828180,0x8f8e8d8c8b8a8988};
	uint64_t tv2tweak[2] = {0x0706050403020100, 0x0f0e0d0c0b0a0908};
	uint64_t tv2pt[Nw] = {0xf8f9fafbfcfdfeff,0xf0f1f2f3f4f5f6f7,0xe8e9eaebecedeeef,0xe0e1e2e3e4e5e6e7,0xd8d9dadbdcdddedf,0xd0d1d2d3d4d5d6d7,0xc8c9cacbcccdcecf,0xc0c1c2c3c4c5c6c7,0xb8b9babbbcbdbebf,0xb0b1b2b3b4b5b6b7,0xa8a9aaabacadaeaf,0xa0a1a2a3a4a5a6a7,0x98999a9b9c9d9e9f,0x9091929394959697,0x88898a8b8c8d8e8f,0x8081828384858687};
	threefish(tv2key, tv2tweak, tv2pt);
	printf("3f-tv2:\n");
	printMsg(tv2pt);

	uint64_t key[Nw];
	uint64_t tweak[2];
	uint64_t plaintext[Nw];

	mkrand(key, sizeof(uint64_t), Nw);
	mkrand(tweak, sizeof(uint64_t), 2);
	mkrand(plaintext, sizeof(uint64_t), Nw);

	printf("message:\n");
	printMsg(plaintext);

	threefish(key, tweak, plaintext);

	printf("ciphertext:\n");
	printMsg(plaintext);

	return 0;
}

int ubi_test()
{
	uint64_t G[Nw];
	uint8_t str[8000];
	uint64_t tweak[2];

	mkrand(G, sizeof(uint64_t), Nw);
	mkrand(str, sizeof(uint64_t), 1000);
	mkrand(tweak, sizeof(uint64_t), 2);

	tweak[1] &= 0x000000007FFFFFFF;

	ubi(G, str, 8000, tweak);

	printf("hash:\n");
	printMsg(G);

	return 0;
}

int skeinhash1024x1024_test()
{
	uint64_t output[Nw];

	uint8_t tv1 = 0xFF;
	printf("tv1:\n");
	skeinhash1024x1024(&tv1, 1, output);
	printMsg(output);

	uint8_t tv2[] = {0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8, 0xF7, 0xF6, 0xF5, 0xF4, 0xF3, 0xF2, 0xF1, 0xF0, 0xEF, 0xEE, 0xED, 0xEC, 0xEB, 0xEA, 0xE9, 0xE8, 0xE7, 0xE6, 0xE5, 0xE4, 0xE3, 0xE2, 0xE1, 0xE0, 0xDF, 0xDE, 0xDD, 0xDC, 0xDB, 0xDA, 0xD9, 0xD8, 0xD7, 0xD6, 0xD5, 0xD4, 0xD3, 0xD2, 0xD1, 0xD0, 0xCF, 0xCE, 0xCD, 0xCC, 0xCB, 0xCA, 0xC9, 0xC8, 0xC7, 0xC6, 0xC5, 0xC4, 0xC3, 0xC2, 0xC1, 0xC0, 0xBF, 0xBE, 0xBD, 0xBC, 0xBB, 0xBA, 0xB9, 0xB8, 0xB7, 0xB6, 0xB5, 0xB4, 0xB3, 0xB2, 0xB1, 0xB0, 0xAF, 0xAE, 0xAD, 0xAC, 0xAB, 0xAA, 0xA9, 0xA8, 0xA7, 0xA6, 0xA5, 0xA4, 0xA3, 0xA2, 0xA1, 0xA0, 0x9F, 0x9E, 0x9D, 0x9C, 0x9B, 0x9A, 0x99, 0x98, 0x97, 0x96, 0x95, 0x94, 0x93, 0x92, 0x91, 0x90, 0x8F, 0x8E, 0x8D, 0x8C, 0x8B, 0x8A, 0x89, 0x88, 0x87, 0x86, 0x85, 0x84, 0x83, 0x82, 0x81, 0x80};
	printf("tv2:\n");
	skeinhash1024x1024(tv2, 128, output);
	printMsg(output);

	uint8_t tv3[] = {0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8, 0xF7, 0xF6, 0xF5, 0xF4, 0xF3, 0xF2, 0xF1, 0xF0,0xEF, 0xEE, 0xED, 0xEC, 0xEB, 0xEA, 0xE9, 0xE8, 0xE7, 0xE6, 0xE5, 0xE4, 0xE3, 0xE2, 0xE1, 0xE0,0xDF, 0xDE, 0xDD, 0xDC, 0xDB, 0xDA, 0xD9, 0xD8, 0xD7, 0xD6, 0xD5, 0xD4, 0xD3, 0xD2, 0xD1, 0xD0,0xCF, 0xCE, 0xCD, 0xCC, 0xCB, 0xCA, 0xC9, 0xC8, 0xC7, 0xC6, 0xC5, 0xC4, 0xC3, 0xC2, 0xC1, 0xC0,0xBF, 0xBE, 0xBD, 0xBC, 0xBB, 0xBA, 0xB9, 0xB8, 0xB7, 0xB6, 0xB5, 0xB4, 0xB3, 0xB2, 0xB1, 0xB0,0xAF, 0xAE, 0xAD, 0xAC, 0xAB, 0xAA, 0xA9, 0xA8, 0xA7, 0xA6, 0xA5, 0xA4, 0xA3, 0xA2, 0xA1, 0xA0,0x9F, 0x9E, 0x9D, 0x9C, 0x9B, 0x9A, 0x99, 0x98, 0x97, 0x96, 0x95, 0x94, 0x93, 0x92, 0x91, 0x90,0x8F, 0x8E, 0x8D, 0x8C, 0x8B, 0x8A, 0x89, 0x88, 0x87, 0x86, 0x85, 0x84, 0x83, 0x82, 0x81, 0x80,0x7F, 0x7E, 0x7D, 0x7C, 0x7B, 0x7A, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x72, 0x71, 0x70,0x6F, 0x6E, 0x6D, 0x6C, 0x6B, 0x6A, 0x69, 0x68, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, 0x60,0x5F, 0x5E, 0x5D, 0x5C, 0x5B, 0x5A, 0x59, 0x58, 0x57, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51, 0x50,0x4F, 0x4E, 0x4D, 0x4C, 0x4B, 0x4A, 0x49, 0x48, 0x47, 0x46, 0x45, 0x44, 0x43, 0x42, 0x41, 0x40,0x3F, 0x3E, 0x3D, 0x3C, 0x3B, 0x3A, 0x39, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x30,0x2F, 0x2E, 0x2D, 0x2C, 0x2B, 0x2A, 0x29, 0x28, 0x27, 0x26, 0x25, 0x24, 0x23, 0x22, 0x21, 0x20,0x1F, 0x1E, 0x1D, 0x1C, 0x1B, 0x1A, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00};
	printf("tv3:\n");
	skeinhash1024x1024(tv3, 256, output);
	printMsg(output);

	uint8_t str[8000];
	mkrand(str, sizeof(uint64_t), 1000);
	printf("random:\n");
	skeinhash1024x1024(str, 8000, output);
	printMsg(output);

	return 0;
}

#define numTries 1000
#define numSearches 50000
int xkcdSearch(int startMin)
{
	int min = startMin; // 424
	uint64_t bu[Nw] = {0};

	// for (int t = 0; t < numTries; t++)
	while(1)
	{
		uint64_t tests[numSearches][Nw];
		uint64_t out[Nw];
		uint64_t ans[Nw] = {0x8082a05f5fa94d5b,0xc818f444df7998fc,0x7d75b724a42bf1f9,0x4f4c0daefbbd2be0,0x04fec50cc81793df,0x97f26c46739042c6,0xf6d2dd9959c2b806,0x877b97cc75440d54,0x8f9bf123e07b75f4,0x88b7862872d73540,0xf99ca716e96d8269,0x247d34d49cc74cc9,0x73a590233eaa67b5,0x4066675e8aa473a3,0xe7c5e19701c79cc7,0xb65818ca53fb02f9};

		mkrand(tests, sizeof(uint64_t)*Nw, numSearches);
		for (int i = 0; i < numSearches; i++)
		{
			skeinhash1024x1024((uint8_t*)&tests[i], Nw, out);
			int c = 0;
			for (int j = 0; j < Nw; j++)
			{
				uint64_t tmp = ans[j] ^ out[j];
				for (int k = 0; k < 64; k++)
				{
					if(tmp & 1)
						c++;
					tmp >>= 1;
				}
			}
			if(c < min)
			{
				min = c;
				for (int j = 0; j < Nw; j++)
				{
					bu[j] = out[j];
				}
				printf("min: %d\n", min);
				printMsg(bu);
			}
		}
	}

	return min;
}

int main(int argc, char** argv)
{
	// threefish_test();

	// ubi_test();

	// skeinhash1024x1024_test();

	int start = 1025;
	if(argc > 1)
		start = atoi(argv[1]);
	xkcdSearch(start);

	return 0;
}

#endif
