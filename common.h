#pragma once

//#include <stdio.h>
//#include <vector>
//#include <sstream>

#define DIMENSION               3
#define DIRECTION_SIZE          19
#define SIM_DIR_EXT             19
#define cs                      1.0/sqrt(3)
#define len                     4

#define XDIM                    0
#define ZDIM                    1
#define YDIM                    2

//total number of processes = xdecomp*ydecomp*zdecomp
#define XDECOMP                 20
#define YDECOMP                 20
#define ZDECOMP                 20

#define INPUT                   (*pinput)
#define SOLVER                  (*psolver)

#define SAVE_PATH "./result/"

#define CAVITY			1
#define POISEUILLE		0

#define BINARYTYPE		0
#define CSVTYPE			1

#define NODE_FLUID              (1 << 1)
#define NODE_FIXEDWALL          (1 << 2)
#define NODE_MOVINGWALL         (1 << 3)
#define NODE_INLET              (1 << 4)
#define NODE_OUTLET             (1 << 5)
#define NODE_STRUCT             (1 << 6)

#define TILE_NUM		1471
