CFLAGS=-O3 -ggdb3 --std=gnu11 -Wall
CXXFLAGS=-O3 -ggdb3 -Wall

none:

threefish: threefish.c
	${CC} ${CFLAGS} -o threefish threefish.c

skeinsearch: skeinsearch.cu threefish.c
	nvcc skeinsearch.cu -o skeinsearch -lcurand --compiler-options "${CXXFLAGS}"

gpuskeintest: gpuskeintest.cu threefish.c
	nvcc gpuskeintest.cu -o gpuskeintest --compiler-options "${CXXFLAGS}"

skeinhash: skeinhash.c
	${CC} ${CFLAGS} -o skeinhash skeinhash.c

run: skeinsearch
	./skeinsearch

clean:
	-rm threefish
	-rm skeinsearch
	-rm gpuskeintest

.PHONY: clean run
