[Top level](../README.md)

# System Requirements

## Operating System

The SVT-AV1 Encoder has been validated to run on Windows, Linux and macOS 64-bit
operating systems. The encoder application and library are built and tested in
CI on the latest GitHub-hosted runners for:

- __Windows (64-bit):__
  - Windows Server (MSVC and MinGW UCRT64)
- __Linux (64-bit):__
  - Ubuntu LTS (GCC and Clang)
- __Other Unix-like (64-bit):__
  - macOS (x86-64 and Apple silicon)

## Hardware

The SVT-AV1 Encoder library supports the x86-64 and AArch64 architectures with handwritten
simd code, however, the encoder can be compiled and run on any architecture
that a valid C99 compiler can target.

- __CPU Requirements__

  In order to achieve the performance targeted by the SVT-AV1 Encoder, a modern
  x86-64 CPU with at least AVX2 support, or an AArch64 CPU with NEON support, is
  highly recommended. On x86-64, simd support is extended all the way back to sse2
  and up to AVX-512. On AArch64, the encoder also uses the DotProd, I8MM, SVE and
  SVE2 extensions when available (see the [AArch64 build guide](ARM-Build-Guide.md)).

- __RAM Requirements__

  The SVT-AV1 Encoder will adapt itself to the system on which it is being run,
  albeit only to a slight degree. The memory requirements for encoding will
  depend primarily on the number of cores the encoder will target (`--lp`),
  the input resolution and bit-depth (`-h`, `-w`, and `--input-depth`), look
  ahead distance (`--lookahead`), and hierarchical levels (`--hierarchical-levels`).
  The encoder will display an error if the system lacks enough RAM prior to the
  start of the encode.
