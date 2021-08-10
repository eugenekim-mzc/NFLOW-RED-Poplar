- E8ight LBM3D code porting to IPU with poplar 
2021.08.03 
Eugene Kim, eugenekim@megazone.com 

- How to run 
make
./test.x 

- Output sliced Tensor in Vertex : arF
  size [ny+2][nz+2][nx+2][DIRECTION_SIZE] 
  2 Ghost cells in each direction 

- Domain decomposition 
XDIM, YDIM, ZDIM in 'common.h'
   * 1,1,1 : not sliced Tensor
   * 1,2,1 : Divide in 1/2 in y direction



