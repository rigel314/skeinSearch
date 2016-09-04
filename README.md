skeinSearch
==============

## About
I got an NVIDIA 1070 and wanted to use it for somthing other than gaming.  So I started learned CUDA programming.
There was a hashing competition in an [xkcd](https://www.explainxkcd.com/wiki/index.php/1193:_Externalities) a while ago, so I thought I'd see how well I could do.
I implemented the skein hash algorithm from the specification in [this document](http://www.skein-hash.info/sites/default/files/skein1.3.pdf).

## Warning
While I have compile-time options to change the skein internal block size.  I have only tested any of this code in skein 1024x1024 mode.  I would expect skein 1024x? to work for any `?`.
I may have the rotation constants or permutation functions wrong for skein 256x? and/or skein 512x?.

Also, I implemented this without any concern for security, so it's probably vulnerable to timing attacks and other such side channel attacks.

## Working
 - Threefish block cipher
 - UBI chaining mode
 - Skein hashing
 - C search
 - GPU search

## Planned Features
 - Generating random data in each GPU thread, so I don't have to allocate ~6GiB of graphics memory.
 - Command line options for CUDA kernel launch sizes.
 - GPU threads using different lengths of the same random data.
 - Sequential brute-force mode, as opposed to the random data, I'm using now.
