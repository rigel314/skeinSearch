#include <stdio.h>
#include <stdint.h>

#define NO_MAIN
#include "threefish.c"

int main(int argc, char** argv)
{
	if(argc < 2 || argc > 3)
	{
		printf("bad args\n");
		printf(	"Usage: skeinhash file [compare-hash] \n"
				"calculates skein1024x1024 hash of file, optionally comparing to compare-hash.\n");
		return 1;
	}

	FILE* fp = fopen(argv[1], "r");
	if(!fp)
	{
		printf("bad file\n");
		return 2;
	}

	fseek(fp, 0, SEEK_END);
	size_t size = ftell(fp);
	rewind(fp);

	uint8_t* data = malloc(size);

	fread(data, size, 1, fp);

	uint64_t hash[16];
	skeinhash1024x1024(data, size, hash);

	if(argc == 3)
	{
		// uint64_t ans[16] = {0x8082a05f5fa94d5b,0xc818f444df7998fc,0x7d75b724a42bf1f9,0x4f4c0daefbbd2be0,0x04fec50cc81793df,0x97f26c46739042c6,0xf6d2dd9959c2b806,0x877b97cc75440d54,0x8f9bf123e07b75f4,0x88b7862872d73540,0xf99ca716e96d8269,0x247d34d49cc74cc9,0x73a590233eaa67b5,0x4066675e8aa473a3,0xe7c5e19701c79cc7,0xb65818ca53fb02f9};
		uint64_t ans[16] = {0};

		if(strlen(argv[2]) != 256)
		{
			printf("bad hash of length %d\n", strlen(argv[2]));
			return 3;
		}

		char hexTable[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8,9,0,0,0,0,0,0,0,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

		for(int i = 0; i < 16; i++)
		{
			for(int j = 0; j < 8; j++)
			{
				ans[i] |= (uint64_t)hexTable[argv[2][i*16+2*j]] << 4*(2*j+1);
				ans[i] |= (uint64_t)hexTable[argv[2][i*16+2*j+1]] << 4*2*j;
			}
		}

		// printMsg(ans);

		int c = 0;
		for (int j = 0; j < 16; j++)
		{
			uint64_t tmp = ans[j] ^ hash[j];
			for (int k = 0; k < 64; k++)
			{
				if(tmp & 1)
					c++;
				tmp >>= 1;
			}
		}
		printf("differs by %d bits\n", c);
	}

	printMsg(hash);

	free(data);
	fclose(fp);
}
