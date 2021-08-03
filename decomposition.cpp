#include <iostream>
#include <vector>
#include <cmath>
#include "struct.h"

using namespace std;



int index_ghostcell(int nx, int nSlave, int x, int y, int z)
{
	//+1 : convert normal index to index with ghost cell 
	int inc =(x+1) + (y+1)*nSlave + (z+1)*nx; 
	return inc; 
}

int set_move_decomp(const int NTILE, const int Slice[][DIMENSION][2], int Move[][DIRECTION_SIZE]) 
{
	int xs,xe,ys,ye,zs,ze;
	int nx,nz,nSlave;
	
	for(int tile=0; tile<NTILE; tile++)
	{
		xs = Slice[tile][0][0];
		xe = Slice[tile][0][1];
		ys = Slice[tile][1][0];
		ye = Slice[tile][1][1];
		zs = Slice[tile][2][0];
		ze = Slice[tile][2][1];

		//printf("TILE=%4d : %3d %3d | %3d %3d | %3d %3d \n",tile, xs,xe,ys,ye,zs,ze);

		nx = (xe - xs) + 2;
		nz = (ze - zs) + 2;
		nSlave = nx*nz; 

		//====================================================================
        // Org index(x,y,z) --> Ghost Cell index(x+1,y+1,z+1)
        // nNodeID = (x+1) + (y+1)*nSlave + (z+1)*nx 
		//====================================================================
    	Move[tile][0]  = index_ghostcell(nx, nSlave, 0, 0, 0); 			//
    	Move[tile][1]  = index_ghostcell(nx, nSlave, 1, 0, 0);   		//+x
    	Move[tile][2]  = index_ghostcell(nx, nSlave,-1, 0, 0);   		//-x
    	Move[tile][3]  = index_ghostcell(nx, nSlave, 0, 0,-1);   		//-z
    	Move[tile][4]  = index_ghostcell(nx, nSlave, 0, 0, 1);   		//+z
    	Move[tile][5]  = index_ghostcell(nx, nSlave, 0, 1, 0);   		//+y
    	Move[tile][6]  = index_ghostcell(nx, nSlave, 0,-1, 0);   		//-y
    	Move[tile][7]  = index_ghostcell(nx, nSlave, 1, 0,-1);    	 	//+x -z
    	Move[tile][8]  = index_ghostcell(nx, nSlave,-1, 0,-1);    	 	//-x -z
    	Move[tile][9]  = index_ghostcell(nx, nSlave, 1, 0, 1);    	 	//+x +z
    	Move[tile][10] = index_ghostcell(nx, nSlave,-1, 0, 1);    	 	//-x +z
    	Move[tile][11] = index_ghostcell(nx, nSlave, 1, 1, 0);    	 	//+x +y
    	Move[tile][12] = index_ghostcell(nx, nSlave,-1, 1, 0);    	 	//-x +y
    	Move[tile][13] = index_ghostcell(nx, nSlave, 1,-1, 0);    	 	//+x -y
    	Move[tile][14] = index_ghostcell(nx, nSlave,-1,-1, 0);    	 	//-x -y
    	Move[tile][15] = index_ghostcell(nx, nSlave, 0, 1,-1);    	 	//+y -z
    	Move[tile][16] = index_ghostcell(nx, nSlave, 0, 1, 1);    	 	//+y +z
    	Move[tile][17] = index_ghostcell(nx, nSlave, 0,-1,-1);    	 	//-y -z
    	Move[tile][18] = index_ghostcell(nx, nSlave, 0,-1, 1);    	 	//-y +z
	}
    return 0;
}



int slice_chunk(const unsigned tile, const int Slice[][DIMENSION][2], int *xs, int *xe, \
                int *ys, int *ye, int *zs, int *ze, \
                int *chunk_nx, int *chunk_ny, int *chunk_nz, int *chunk_len) 
{
	*xs = Slice[tile][0][0];
	*xe = Slice[tile][0][1];
	*ys = Slice[tile][1][0];
	*ye = Slice[tile][1][1];
	*zs = Slice[tile][2][0];
	*ze = Slice[tile][2][1];

	//printf("Tile=%d : %3d %3d | %3d %3d | %3d %3d \n",tile, *xs,*xe,*ys,*ye,*zs,*ze);

	int dx,dy,dz;
	dx = (*xe - *xs);
	dy = (*ye - *ys);
	dz = (*ze - *zs);

	*chunk_nx = dx;
	*chunk_ny = dy;
	*chunk_nz = dz;
	*chunk_len = dx*dy*dz;

    if(dx<1) return 1;
    if(dy<1) return 1;
    if(dz<1) return 1;

    return 0;
}



int decomposition(Inputval *pinput, int Slice[][DIMENSION][2]) 
{

	int tiles=0;

	//Chunk size in each direction 
	int chunk[3] = {INPUT.nx/XDECOMP, INPUT.ny/YDECOMP, INPUT.nz/ZDECOMP};

	//Start&End index in each direction 
	int st[3]={0,0,0};
	int ed[3]={0,0,0};
	int start,end;

	for(int i=0; i<XDECOMP; i++)
	{
	    start = i*chunk[0];
		end   = (i+1)*chunk[0];
		if(i == XDECOMP-1) end = INPUT.nx;		
		if(end>INPUT.nx) end = INPUT.nx;		
	    st[0] = start;
		ed[0] = end;

		for(int j=0; j<YDECOMP; j++)
		{
	    	start = j*chunk[1];
			end   = (j+1)*chunk[1];
			if(j == YDECOMP-1) end = INPUT.ny;		
			if(end>INPUT.ny) end = INPUT.ny;		
	    	st[1] = start;
			ed[1] = end;
			for(int k=0; k<ZDECOMP; k++)
			{
		    	start = k*chunk[2];
				end   = (k+1)*chunk[2];
				if(k == ZDECOMP-1) end = INPUT.nz;		
				if(end>INPUT.nz) end = INPUT.nz;		
	    		st[2] = start;
				ed[2] = end;
				
				//printf("ST|ED : %3d %3d | %3d %3d | %3d %3d \n",st[0],ed[0],st[1],ed[1],st[2],ed[2]);

				for(int m=0; m<DIMENSION; m++)
				{
					Slice[tiles][m][0] = st[m];
					Slice[tiles][m][1] = ed[m];
				}
				tiles++;
			}	
		}
	}

	/*
	for(int i=0; i<tiles; i++)
	{
		for(int dir=0; dir<DIMENSION; dir++)
		{
			printf("%3d %3d ",Slice[i][dir][0],Slice[i][dir][1]);
		}
		printf("\n");
	}
	*/

	return tiles;
}
		
