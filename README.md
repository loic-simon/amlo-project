# amlo-project
Small C project for the Architecture matérielle et logicielle des ordinateurs course at Mines Paris - PSL.


## About

See [project instructions](Projet_S1825_2022.pdf) for all details.


## Usage

### Requirements

  * C99+ with GCC;
  * A processor supporting AVX/AVX2 instruction sets, with serveral cores.

### Compilation

In this repo root folder,
```bash
make
```

Or, equivalently,
```bash
gcc -mavx2 -pthread -O3 -Wall -o lsimon_S1825_2022 src/lsimon_S1825_2022.c -lm
```

### Run

```bash
./lsimon_S1825_2022
```

The number of threads to use (default 8) can be passed as first CLI argument:
```bash
./lsimon_S1825_2022 16
```

The size of the test vector (default 1024²) can be passed as second CLI argument:
```bash
./lsimon_S1825_2022 16 1000
```
