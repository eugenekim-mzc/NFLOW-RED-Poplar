// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <poplar/Vertex.hpp>
#include <ipudef.h>
#include <cmath>
#include <print.h>

#include "common.h"

using namespace poplar;


float dot(float a[], float b[])
{ 
	float sum = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
	return sum;
}

/* ================================================================== */
/* keStreaming  												      */
/* ================================================================== */
class Streaming : public Vertex
{
public:
	// Fields
	Input<Vector<int>> nNodeType;
	Output<Vector<float>> arF;
	Input<Vector<float>> arFTemp;
	Input<Vector<int>> DirectionState;
	Input<Vector<int>> Move;
	Input<Vector<int>> nOppIndex;
	Input<int> nx;
	Input<int> ny;
	Input<int> nz;
	Input<int> length;

	// Compute function
	bool compute()
	{
		int nSlave  = nx*nz;
		int nx2 	= nx+2;
		int nSlave2 = nx2*(nz+2);
		int nMaxNode = length;

    	// Redefining the index definition : input(donor) perspective --> output(receiver) perspective : nOpp

		for(unsigned y=0; y<ny; y++)
		{
  			for(unsigned z=0; z<nz; z++)
			{
  				for(unsigned x=0; x<nx; x++)
				{
					int nNodeID = x + z*nx + y*nSlave;
    				int nodetype = nNodeType[nNodeID];
    				if (nodetype == NODE_STRUCT) continue;

					int dirstate = DirectionState[nNodeID];
					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
						if ((dirstate >> nDir) & 1)   								//Donor point is inside 
						{
							//int nMoveNodeID = nNodeID + Move[nDir];
							int nMoveNodeID = x + z*nx2 + y*nSlave2 + Move[nDir];
						
							/* Must be modified!!!!! 
							int movenodetype = nNodeType[nMoveNodeID];
							if (movenodetype == NODE_STRUCT) continue;
							*/
                
							int nDir_pair = nOppIndex[nDir];
							/*
							if(z==2){
							printf("%d %d %d | DIR=%d | CONVdir=%d | %d <-- %d | %f\n",x,y,z,nDir,nDir_pair, nNodeID, nMoveNodeID,arFTemp[nDir_pair + nMoveNodeID*DIRECTION_SIZE]);
							}
							*/
							arF[nDir_pair + nNodeID*DIRECTION_SIZE] = arFTemp[nDir_pair + nMoveNodeID*DIRECTION_SIZE];
						}
					}


					/*
					int dirstate = DirectionState[nNodeID];
					float ftemp[DIRECTION_SIZE] = { 0.0f, };
					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
						ftemp[nDir] = arFTemp[nDir + nNodeID*DIRECTION_SIZE];
					}

					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
						if ((dirstate >> nDir) & 1)   						//InSide
						{
							int nMoveNodeID = nNodeID + Move[nDir];
							int movenodetype = nNodeType[nMoveNodeID];
							if (movenodetype == NODE_STRUCT) continue;
                
							arF[nDir + nMoveNodeID*DIRECTION_SIZE] = ftemp[nDir];
						}
					}
					*/
				}
			}
		}
  		return true;
	}
};


/* ================================================================== */
/* keInitialNodeDirection + keSetNodeType(CAVITY ONLY) + keInitialize */
/* ================================================================== */
class Initialize : public Vertex
{
public:
	// Fields
	Output<Vector<int>> DirectionState;
	Output<Vector<int>> nNodeType;
	InOut<Vector<float>> arF;
	Output<Vector<float>> CurrDen;
	Output<Vector<float>> CurrVelx;
	Output<Vector<float>> CurrVely;
	Output<Vector<float>> CurrVelz;
	Input<Vector<float>> arW;
	Input<Vector<float>> Direction;
	Input<float> RefDen;
	Input<float> RefVelx;
	Input<float> RefVely;
	Input<float> RefVelz;
	Input<int> nx;
	Input<int> ny;
	Input<int> nz;
	Input<int> length;
	Input<int> NX;
	Input<int> NY;
	Input<int> NZ;
	Input<int> PADX;
	Input<int> PADY;
	Input<int> PADZ;

	// Compute function
	bool compute()
	{
		int nSlave = nx*nz;
		//int nMaxNode = length;

		/**********************************************************************
		i,j,k : index of Variables in IPU(decomposed Tensor)
		x,y,z : index of Variables in Host(full size, not decomposed)
		**********************************************************************/
		for(unsigned j=0; j<ny; j++)
		{
			int y = j+PADY;
  			for(unsigned k=0; k<nz; k++)
			{
				int z = k+PADZ;
  				for(unsigned i=0; i<nx; i++)
				{
					int x = i+PADX;
					int nNodeID = i + k*nx + j*nSlave;

					/*=========================================================================
					keInitialNodeDirection 
					=========================================================================*/
					int dirstate =0;
					for(unsigned nDir=DIRECTION_SIZE-1; nDir>0; --nDir)
					{
						int y1 = y + Direction[YDIM*DIRECTION_SIZE + nDir];
						int z1 = z + Direction[ZDIM*DIRECTION_SIZE + nDir];
						int x1 = x + Direction[XDIM*DIRECTION_SIZE + nDir];
						if(x1<NX && y1<NY && z1<NZ && x1>=0 && y1>=0 && z1>=0)	//InSide
						{
							dirstate = (dirstate | 1) << 1;      
						}
						else 													//OutSide
						{
							dirstate = dirstate << 1;      
						}
					}
					DirectionState[nNodeID] = (dirstate | 1);

					/*=========================================================================
					keSetNodeType (Only CAVITY flow)
					=========================================================================*/
					int nType=NODE_FLUID;
					if (y == 0)
					{
						nType = NODE_FIXEDWALL;
						//printf("(Bottom)Fixed Wall : %d %d %d | %d %d %d\n",i,j,k,x,y,z);
					}

					if (y == NY - 1)
					{
						nType = NODE_MOVINGWALL;
						//printf("Moving Wall : %d %d %d | %d %d %d\n",i,j,k,x,y,z);
					}

					if ((x == 0 || x == NX - 1) || (z == 0 || z == NZ - 1))
					{
						nType = NODE_FIXEDWALL;
						//printf("(Side)Fixed Wall : %d %d %d | %d %d %d\n",i,j,k,x,y,z);
					}
					nNodeType[nNodeID] = nType;
				

					/*=========================================================================
					keInitialize
					=========================================================================*/
					if(nType == NODE_STRUCT) continue;

					// Initialize flow variables 
					float den  = RefDen;
					float velx = 0.0f;
					float vely = 0.0f;
					float velz = 0.0f;

					if(nType == NODE_MOVINGWALL)
					{
						velx = RefVelx;
						vely = RefVely;
						velz = RefVelz;
					}
					
    				// equilbrium 값으로 초기화
    				for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
    				{
        				float edotu = Direction[XDIM*DIRECTION_SIZE + nDir] * velx + Direction[YDIM*DIRECTION_SIZE + nDir] * vely + Direction[ZDIM*DIRECTION_SIZE + nDir] * velz;
        				float usquare = velx * velx + vely * vely + velz * velz;

						/* Different from Original E8ight CUDA version indexing!!!! */
        				//arF[nDir*nMaxNode + nNodeID] = arW[nDir] * den * (1.0f + 3.0f*edotu + 4.5f*edotu*edotu - 1.5f*usquare);
        				arF[nDir + nNodeID*DIRECTION_SIZE] = arW[nDir] * den * (1.0f + 3.0f*edotu + 4.5f*edotu*edotu - 1.5f*usquare);
    				}
    				CurrDen[nNodeID]  = den;
    				CurrVelx[nNodeID] = velx;
    				CurrVely[nNodeID] = vely;
    				CurrVelz[nNodeID] = velz;
				}
			}
  		}

  		return true;
	}
};


/* ================================================================== */
/* keMacroscopic                                                      */
/* ================================================================== */
class Macroscopic : public Vertex
{
public:
	// Fields
	Input<Vector<int>> nNodeType;
	InOut<Vector<float>> CurrDen;
	Output<Vector<float>> PrevDen;
	Output<Vector<float>> CurrVelx;
	Output<Vector<float>> CurrVely;
	Output<Vector<float>> CurrVelz;
	Input<Vector<float>> arF;
	Input<Vector<float>> Direction;
	Input<int> nx;
	Input<int> ny;
	Input<int> nz;
	Input<int> length;

	// Compute function
	bool compute()
	{
		int nSlave = nx*nz;
		//int nMaxNode = length;

		for(unsigned y=0; y<ny; y++)
		{
  			for(unsigned z=0; z<nz; z++)
			{
  				for(unsigned x=0; x<nx; x++)
				{
					int nNodeID = x + z*nx + y*nSlave;
    				int nodetype = nNodeType[nNodeID];
    				if (nodetype == NODE_STRUCT) continue;

					float den = 0.f;
					float velx = 0.f;
					float vely = 0.f;
					float velz = 0.f;
					float f[DIRECTION_SIZE] = { 0.0f, };

					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
						/* Different from Original E8ight CUDA version indexing!!!! */
					    //f[nDir] = arF[nDir*nMaxNode + nNodeID];
					    f[nDir] = arF[nDir + nNodeID*DIRECTION_SIZE];
					}
                
					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
					    den  += f[nDir];
					    velx += f[nDir] * Direction[XDIM*DIRECTION_SIZE + nDir];
					    vely += f[nDir] * Direction[YDIM*DIRECTION_SIZE + nDir];
					    velz += f[nDir] * Direction[ZDIM*DIRECTION_SIZE + nDir];
					}

					//printf("%d %d %d | %f | %f %f %f\n",x,y,z,den, velx, vely, velz);
                
					velx /= den;
					vely /= den;
					velz /= den;
                
					PrevDen[nNodeID] = CurrDen[nNodeID];
					CurrDen[nNodeID] = den;
					CurrVelx[nNodeID] = velx;
					CurrVely[nNodeID] = vely;
					CurrVelz[nNodeID] = velz;
				}
			}
		}
  		return true;
	}
};



/* ================================================================== */
/* keBoundaryCavity 								                  */
/* ================================================================== */
class BoundaryCavity : public Vertex
{
public:
	// Fields
	Input<Vector<int>> nNodeType;
	Output<Vector<float>> arF;
	Input<Vector<float>> arFTemp;
	Input<Vector<float>> arW;
	Input<Vector<float>> Direction;
	Input<Vector<int>> DirectionState;
	Input<Vector<int>> nOppIndex;
	Input<float> RefDen;
	Input<float> velx;
	Input<float> vely;
	Input<float> velz;
	Input<int> nx;
	Input<int> ny;
	Input<int> nz;
	Input<int> length;


	// Compute function
	bool compute()
	{
		int nSlave = nx*nz;
		//int nMaxNode = length;

		for(unsigned y=0; y<ny; y++)
		{
  			for(unsigned z=0; z<nz; z++)
			{
  				for(unsigned x=0; x<nx; x++)
				{
					int nNodeID = x + z*nx + y*nSlave;
    				int nodetype = nNodeType[nNodeID];
    				if (nodetype == NODE_FLUID) continue;
					int dirstate = DirectionState[nNodeID];

					float f[DIRECTION_SIZE] = { 0.0f, };
					float ftemp[DIRECTION_SIZE] = { 0.0f, };

					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
						/* Different from Original E8ight CUDA version indexing!!!! */
						//ftemp[nDir] = arFTemp[nDir*nMaxNode + nNodeID];
						ftemp[nDir] = arFTemp[nDir + nNodeID*DIRECTION_SIZE];
					}

					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
						if (!((dirstate >> nDir) & 1))   						//OutSide
						{
							int nOPP = nOppIndex[nDir];

            				if (nodetype == NODE_FIXEDWALL)
            				{
								/* Different from Original E8ight CUDA version indexing!!!! */
                				//arF[nOPP*nMaxNode + nNodeID] = ftemp[nDir];
                				arF[nOPP + nNodeID*DIRECTION_SIZE] = ftemp[nDir];
								//printf("Fixed Wall : %d %d %d | %d %d\n",x,y,z,nDir,nOPP);
            				}
            				else if (nodetype == NODE_MOVINGWALL)
            				{
								//printf("Moving Wall : %d %d %d | %d %d\n",x,y,z,nDir,nOPP);
                				float edotu = Direction[XDIM*DIRECTION_SIZE + nDir] * velx + Direction[YDIM*DIRECTION_SIZE + nDir] * vely + Direction[ZDIM*DIRECTION_SIZE + nDir] * velz;

								/* Different from Original E8ight CUDA version indexing!!!! */
                				//arF[nOPP*nMaxNode + nNodeID] = ftemp[nDir] - 6.0f*arW[nDir] * RefDen * edotu;
                				arF[nOPP + nNodeID*DIRECTION_SIZE] = ftemp[nDir] - 6.0f*arW[nDir] * RefDen * edotu;
            				}
        				}
					}
				}
			}
		}
  		return true;
	}
};



/* ================================================================== */
/* keCollsion_MRT 													  */
/* ================================================================== */
class Collision_MRT : public Vertex
{
public:
	// Fields
	Input<Vector<int>> nNodeType;
	Input<Vector<float>> arF;
	Output<Vector<float>> arFTemp;
	Input<Vector<float>> CurrDen;
	Input<Vector<float>> CurrVelx;
	Input<Vector<float>> CurrVely;
	Input<Vector<float>> CurrVelz;
	Input<Vector<float>> dTauEddy;
	Input<float> RefDen;
	Input<float> dTau;
	Input<float> dWeps;
	Input<float> dWepsj;
	Input<float> dWXX;
	Input<Vector<float>> m_MatMRT;
	Input<Vector<float>> m_MatInv;
	Input<int> nx;
	Input<int> ny;
	Input<int> nz;
	Input<int> length;

	// Compute function
	bool compute()
	{
		int nSlave = nx*nz;
		int nMaxNode = length;

		for(unsigned y=0; y<ny; y++)
		{
  			for(unsigned z=0; z<nz; z++)
			{
  				for(unsigned x=0; x<nx; x++)
				{
					int nNodeID = x + z*nx + y*nSlave;
    				int nodetype = nNodeType[nNodeID];
    				if (nodetype == NODE_STRUCT) continue;

					float den = CurrDen[nNodeID];
					float dtaueddy = dTauEddy[nNodeID];

					float jVector[3] = {den*CurrVelx[nNodeID], den*CurrVely[nNodeID], den*CurrVelz[nNodeID]}; 

					float f[DIRECTION_SIZE] = { 0.0f, };
					float ftemp[DIRECTION_SIZE] = { 0.0f, };

					for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
					{
						/* Different from Original E8ight CUDA version indexing!!!! */
						//f[nDir] = arF[nDir * nMaxNode + nNodeID];
						f[nDir] = arF[nDir + nNodeID*DIRECTION_SIZE];
					}

					float matMeq[DIRECTION_SIZE] = { 0.0f, };
					float matS[DIRECTION_SIZE] = { 0.0f, };
					float matM1[DIRECTION_SIZE] = { 0.0f, };

					matMeq[0] = den;
					matMeq[1] = -11.0f *den + 19.0f / RefDen * dot(jVector, jVector);
					matMeq[2] = dWeps * den + dWepsj / RefDen * dot(jVector, jVector);
					matMeq[3] = jVector[0];
					matMeq[4] = -2.0f / 3.0f * jVector[0];
					matMeq[5] = jVector[1];
					matMeq[6] = -2.0f / 3.0f * jVector[1];
					matMeq[7] = jVector[2];
					matMeq[8] = -2.0f / 3.0f * jVector[2];
					matMeq[9] = 1.0f / RefDen * (2.0f * pow(jVector[0], 2) - (pow(jVector[1], 2) + pow(jVector[2], 2)));
					matMeq[10] = dWXX * matMeq[9];
					matMeq[11] = 1.0f / RefDen * (pow(jVector[1], 2) - pow(jVector[2], 2));
					matMeq[12] = dWXX * matMeq[11];
					matMeq[13] = jVector[0] * jVector[1] / RefDen;
					matMeq[14] = jVector[1] * jVector[2] / RefDen;
					matMeq[15] = jVector[2] * jVector[0] / RefDen;
					matMeq[16] = 0.0f;
					matMeq[17] = 0.0f;
					matMeq[18] = 0.0f;
                
					matS[0] = 0.0f;
					matS[1] = 1.19f;
					matS[2] = 1.4f;
					matS[3] = 0.0f;
					matS[4] = 1.2f;
					matS[5] = 0.0f;
					matS[6] = 1.2f;
					matS[7] = 0.0f;
					matS[8] = 1.2f;
					matS[9] = 1.0f / (dTau + dtaueddy);
					matS[10] = 1.4f;
					matS[11] = 1.0f / (dTau + dtaueddy);
					matS[12] = 1.4f;
					matS[13] = 1.0f / (dTau + dtaueddy);
					matS[14] = 1.0f / (dTau + dtaueddy);
					matS[15] = 1.0f / (dTau + dtaueddy);
					matS[16] = 1.98f;
					matS[17] = 1.98f;
					matS[18] = 1.98f;
                
					for (int nDir_I = 0; nDir_I < DIRECTION_SIZE; ++nDir_I)
					{
						matM1[nDir_I] = 0.0f;
						// velocity moment 계산
						for (int nDir_J = 0; nDir_J < DIRECTION_SIZE; ++nDir_J)
						{
							//matM1[nDir_I] += m_MatMRT[nDir_I][nDir_J] * f[nDir_J];
							matM1[nDir_I] += m_MatMRT[nDir_J + nDir_I*DIRECTION_SIZE] * f[nDir_J];
						}
					}
                
					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
						ftemp[nDir] = matS[nDir] * (matM1[nDir] - matMeq[nDir]);
					}
                
					for (int nDir_I = 0; nDir_I < DIRECTION_SIZE; ++nDir_I)
					{
						matM1[nDir_I] = 0.0;
						for (int nDir_J = 0; nDir_J < DIRECTION_SIZE; ++nDir_J)
						{
							//matM1[nDir_I] += m_MatInv[nDir_I][nDir_J] * ftemp[nDir_J];
							matM1[nDir_I] += m_MatInv[nDir_J + nDir_I*DIRECTION_SIZE] * ftemp[nDir_J];
						}
					}
                
					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
						ftemp[nDir] = f[nDir] - matM1[nDir];
					}
                
					for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
					{
						/* Different from Original E8ight CUDA version indexing!!!! */
						//arFTemp[nDir * nMaxNode + nNodeID] = ftemp[nDir];
						arFTemp[nDir + nNodeID*DIRECTION_SIZE] = ftemp[nDir];
					}
   				}
			}
		}
  		return true;
	}
};




/* ================================================================== */
/* keComputeTurbulence_LES                                            */
/* ================================================================== */
class ComputeTurbulence_LES : public Vertex
{
public:
	// Fields
	Input<Vector<int>> nNodeType;
	Input<Vector<float>> arF;
	Input<Vector<float>> CurrDen;
	Input<Vector<float>> CurrVelx;
	Input<Vector<float>> CurrVely;
	Input<Vector<float>> CurrVelz;
	Output<Vector<float>> dTauEddy;
	Input<Vector<float>> arW;
	Input<Vector<float>> Direction;
	Input<float> dTau;
	Input<float> dSmagorinsky;
	Input<int> nx;
	Input<int> ny;
	Input<int> nz;
	Input<int> length;

	// Compute function
	bool compute()
	{
		int nSlave = nx*nz;
		int nMaxNode = length;
		//float dTau2 = pow(dTau,2);
		//float dSmagorinsky2 = 18.0f*sqrt(2.0f)*pow(dSmagorinsky,2);

		for(unsigned y=0; y<ny; y++)
		{
  			for(unsigned z=0; z<nz; z++)
			{
  				for(unsigned x=0; x<nx; x++)
				{
					int nNodeID = x + z*nx + y*nSlave;

    				int nodetype = nNodeType[nNodeID];
    				if (nodetype == NODE_STRUCT) continue;

    				float den = CurrDen[nNodeID];
    				float velx = CurrVelx[nNodeID];
    				float vely = CurrVely[nNodeID];
    				float velz = CurrVelz[nNodeID];

    				float f[DIRECTION_SIZE] = { 0.0f, };

    				for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
    				{
						/* Different from Original E8ight CUDA version indexing!!!! */
        				//f[nDir] = arF[nDir*nMaxNode + nNodeID];
        				f[nDir] = arF[nDir + nNodeID*DIRECTION_SIZE];
    				}

    				float usquare = velx * velx + vely * vely + velz * velz;

    				float dPhi0[3] = { 0.0f, 0.0f, 0.0f };
    				float dPhi1[3] = { 0.0f, 0.0f, 0.0f };
    				float dPhi2[3] = { 0.0f, 0.0f, 0.0f };

    				for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
    				{
        				float edotu = Direction[XDIM*DIRECTION_SIZE + nDir] * velx + Direction[YDIM*DIRECTION_SIZE + nDir] * vely + Direction[ZDIM*DIRECTION_SIZE + nDir] * velz;
        				float feq = arW[nDir] * den * (1.0f + 3.0f*edotu + 4.5f*edotu * edotu - 1.5f*usquare);

						// Direction in x,y,z 
						float dir1 = Direction[XDIM*DIRECTION_SIZE + nDir];
						float dir2 = Direction[YDIM*DIRECTION_SIZE + nDir];
						float dir3 = Direction[ZDIM*DIRECTION_SIZE + nDir];
						float mul  = f[nDir] - feq;
						float dir11 = dir1*mul;
						float dir22 = dir2*mul;
						float dir33 = dir3*mul;

						dPhi0[0] += dir1 * dir11;
        				dPhi0[1] += dir1 * dir22;
        				dPhi0[2] += dir1 * dir33;

        				dPhi1[0] += dir2 * dir11;
        				dPhi1[1] += dir2 * dir22;
        				dPhi1[2] += dir2 * dir33;

        				dPhi2[0] += dir3 * dir11;
        				dPhi2[1] += dir3 * dir22;
        				dPhi2[2] += dir3 * dir33;

						/*
        				dPhi0[0] += Direction[XDIM*DIRECTION_SIZE + nDir] * Direction[XDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
        				dPhi0[1] += Direction[XDIM*DIRECTION_SIZE + nDir] * Direction[YDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
        				dPhi0[2] += Direction[XDIM*DIRECTION_SIZE + nDir] * Direction[ZDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);

        				dPhi1[0] += Direction[YDIM*DIRECTION_SIZE + nDir] * Direction[XDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
        				dPhi1[1] += Direction[YDIM*DIRECTION_SIZE + nDir] * Direction[YDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
        				dPhi1[2] += Direction[YDIM*DIRECTION_SIZE + nDir] * Direction[ZDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);

        				dPhi2[0] += Direction[ZDIM*DIRECTION_SIZE + nDir] * Direction[XDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
        				dPhi2[1] += Direction[ZDIM*DIRECTION_SIZE + nDir] * Direction[YDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
        				dPhi2[2] += Direction[ZDIM*DIRECTION_SIZE + nDir] * Direction[ZDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
						*/
    				}

    				float dQles = dot(dPhi0, dPhi0) + dot(dPhi1, dPhi1) + dot(dPhi2, dPhi2);
    				//float dSmag = 0.5f * sqrt(dTau2 + dSmagorinsky2 *sqrt(dQles) / den);
    				float dSmag = 0.5f * (sqrt(pow(dTau, 2) + 18.0f*1.41421356237f*pow(dSmagorinsky, 2)*sqrt(dQles) / den));

    				dTauEddy[nNodeID] = -0.5f*dTau + dSmag;
				}
			}
		}
  		return true;
	}
};



