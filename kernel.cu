
#include "kernel.cuh"

#include <chrono>
#include <iostream>
using namespace std;

__device__ float dot(float3 a, float3 b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(b * a.x, b * a.y, b * a.z);
}

void runSimulation(Inputval *pinput, LBMSolver* psolver)
{
	int nMaxNode = INPUT.length;
	int	blockSize = 512;
	int3 nNodeSize = { INPUT.nx, INPUT.ny, INPUT.nz };

	int gridSize = (nMaxNode % blockSize == 0) ? nMaxNode / blockSize : nMaxNode / blockSize + 1;

	std::vector<int> arIndex;
	arIndex.resize(nMaxNode * 8);
	int nIndexCount = 0;
	for (int nNodeID = 0; nNodeID < nMaxNode; nNodeID++)
	{
		int nSlave = nNodeSize.x * nNodeSize.z;

		int nType = 0;

		int nX = (nNodeID % nSlave) % nNodeSize.x;
		int nY = (nNodeID / nSlave);
		int nZ = (nNodeID % nSlave) / nNodeSize.x;

		if (nX >= nNodeSize.x - 1 || nY >= nNodeSize.y - 1 || nZ >= nNodeSize.z - 1)
			continue;  // 바깥쪽 가장자리 위치임
		for (int nY2 = nY; nY2 < nY + 2; nY2++)
		{
			for (int nXZ2 = 0; nXZ2 < 4; nXZ2++)
			{
				int nX2, nZ2;
				switch (nXZ2)
				{
				case 0: nX2 = nX + 0; nZ2 = nZ + 0;	break;
				case 1: nX2 = nX + 1; nZ2 = nZ + 0;	break;
				case 2: nX2 = nX + 1; nZ2 = nZ + 1;	break;
				case 3: nX2 = nX + 0; nZ2 = nZ + 1;	break;
				}

				int nNodeIndexVtx = nNodeSize.z * nNodeSize.x * nY2 + nNodeSize.x * nZ2 + nX2;
				arIndex[nIndexCount++] = nNodeIndexVtx;
			}
		}
	}

	std::vector<std::vector<float>> mat;
	std::vector<std::vector<float>> matInv = std::vector<std::vector<float>>(DIRECTION_SIZE, std::vector<float>(DIRECTION_SIZE));

	mat =
	{
		{ 1.0,  1.0,  1.0,  1.0,  1.0,		   1.0,  1.0,  1.0,  1.0,  1.0,		   1.0,  1.0,  1.0,  1.0,  1.0,		   1.0,  1.0,  1.0,  1.0 }, // 0
		{ -30.0,-11.0,-11.0,-11.0,-11.0,		-11.0,-11.0,  8.0,  8.0,  8.0,		   8.0,  8.0,  8.0,  8.0,  8.0,		   8.0,  8.0,  8.0,  8.0 }, // 1
		{ 12.0, -4.0, -4.0, -4.0, -4.0,		  -4.0, -4.0,  1.0,  1.0,  1.0,		   1.0,  1.0,  1.0,  1.0,  1.0,		   1.0,  1.0,  1.0,  1.0 }, // 2
		{ 0.0,  1.0, -1.0,  0.0,  0.0,		   0.0,  0.0,  1.0, -1.0,  1.0,		  -1.0,  1.0, -1.0,  1.0, -1.0,		   0.0,  0.0,  0.0,  0.0 }, // 3
		{ 0.0, -4.0,  4.0,  0.0,  0.0,		   0.0,  0.0,  1.0, -1.0,  1.0,		  -1.0,  1.0, -1.0,  1.0, -1.0,		   0.0,  0.0,  0.0,  0.0 }, // 4
		{ 0.0,	 0.0,  0.0,  0.0,  0.0,        1.0, -1.0,  0.0,  0.0,  0.0,        0.0,  1.0,  1.0, -1.0, -1.0,        1.0,  1.0, -1.0, -1.0 }, // 5
		{ 0.0,  0.0,  0.0,  0.0,  0.0,       -4.0,  4.0,  0.0,  0.0,  0.0,        0.0,  1.0,  1.0, -1.0, -1.0,        1.0,  1.0, -1.0, -1.0 }, // 6
		{ 0.0,  0.0,  0.0, -1.0,  1.0,        0.0,  0.0, -1.0, -1.0,  1.0,        1.0,  0.0,  0.0,  0.0,  0.0,       -1.0,  1.0, -1.0,  1.0 }, // 7
		{ 0.0,  0.0,  0.0,  4.0, -4.0,        0.0,  0.0, -1.0, -1.0,  1.0,        1.0,  0.0,  0.0,  0.0,  0.0,       -1.0,  1.0, -1.0,  1.0 }, // 8
		{ 0.0,  2.0,  2.0, -1.0, -1.0,       -1.0, -1.0,  1.0,  1.0,  1.0,        1.0,  1.0,  1.0,  1.0,  1.0,       -2.0, -2.0, -2.0, -2.0 }, // 9
		{ 0.0, -4.0, -4.0,  2.0,  2.0,        2.0,  2.0,  1.0,  1.0,  1.0,        1.0,  1.0,  1.0,  1.0,  1.0,       -2.0, -2.0, -2.0, -2.0 }, // 10
		{ 0.0,  0.0,  0.0, -1.0, -1.0,        1.0,  1.0, -1.0, -1.0, -1.0,       -1.0,  1.0,  1.0,  1.0,  1.0,        0.0,  0.0,  0.0,  0.0 }, // 11
		{ 0.0,  0.0,  0.0,  2.0,  2.0,       -2.0, -2.0, -1.0, -1.0, -1.0,       -1.0,  1.0,  1.0,  1.0,  1.0,        0.0,  0.0,  0.0,  0.0 }, // 12
		{ 0.0,  0.0,  0.0,  0.0,  0.0,        0.0,  0.0,  0.0,  0.0,  0.0,        0.0,  1.0, -1.0, -1.0,  1.0,        0.0,  0.0,  0.0,  0.0 }, // 13
		{ 0.0,  0.0,  0.0,  0.0,  0.0,        0.0,  0.0,  0.0,  0.0,  0.0,        0.0,  0.0,  0.0,  0.0,  0.0,       -1.0,  1.0,  1.0, -1.0 }, // 14
		{ 0.0,  0.0,  0.0,  0.0,  0.0,        0.0,  0.0, -1.0,  1.0,  1.0,       -1.0,  0.0,  0.0,  0.0,  0.0,        0.0,  0.0,  0.0,  0.0 }, // 15
		{ 0.0,  0.0,  0.0,  0.0,  0.0,        0.0,  0.0, -1.0,  1.0, -1.0,        1.0,  1.0, -1.0,  1.0, -1.0,        0.0,  0.0,  0.0,  0.0 }, // 16
		{ 0.0,  0.0,  0.0,  0.0,  0.0,        0.0,  0.0,  0.0,  0.0,  0.0,        0.0, -1.0, -1.0,  1.0,  1.0,        1.0,  1.0, -1.0, -1.0 }, // 17
		{ 0.0,  0.0,  0.0,  0.0,  0.0,        0.0,  0.0, -1.0, -1.0,  1.0,        1.0,  0.0,  0.0,  0.0,  0.0,        1.0, -1.0,  1.0, -1.0 }  // 18
	};

	inverseMRT(mat, matInv);
	MRTMatrix temp;
	for (int i = 0; i < DIRECTION_SIZE; i++)
	{
		for (int j = 0; j < DIRECTION_SIZE; j++)
		{
			temp.m_MatMRT[i][j] = mat[i][j];
			temp.m_MatInv[i][j] = matInv[i][j];
		}
	}

	cudaMalloc(&SOLVER.m_DeviceMatrix, sizeof(MRTMatrix));
	cudaMalloc(&SOLVER.m_DeviceDirection, sizeof(float) * DIMENSION * DIRECTION_SIZE);
	cudaMalloc(&SOLVER.m_DeviceW, sizeof(float) * DIRECTION_SIZE);
	cudaMalloc(&SOLVER.m_DeviceOppIndex, sizeof(int) * DIRECTION_SIZE);
	cudaMalloc(&SOLVER.m_DeviceMove, sizeof(int) * DIRECTION_SIZE);

	cudaMemcpy(SOLVER.m_DeviceMatrix, &temp, sizeof(MRTMatrix), cudaMemcpyHostToDevice);
	cudaMemcpy(SOLVER.m_DeviceDirection, SOLVER.m_HostDirection, sizeof(float) * DIMENSION * DIRECTION_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(SOLVER.m_DeviceW, SOLVER.m_HostW, sizeof(float) * DIRECTION_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(SOLVER.m_DeviceOppIndex, SOLVER.m_HostOppIndex, sizeof(int) * DIRECTION_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(SOLVER.m_DeviceMove, SOLVER.m_HostMove, sizeof(int) * DIRECTION_SIZE, cudaMemcpyHostToDevice);

	cudaMalloc(&SOLVER.m_DeviceNode.m_arF, sizeof(float)*nMaxNode*DIRECTION_SIZE);
	cudaMalloc(&SOLVER.m_DeviceNode.m_arFTemp, sizeof(float)*nMaxNode*DIRECTION_SIZE);

	cudaMalloc(&SOLVER.m_DeviceNode.m_dCurrDen, sizeof(float)*nMaxNode);
	cudaMalloc(&SOLVER.m_DeviceNode.m_dPrevDen, sizeof(float)*nMaxNode);

	cudaMalloc(&SOLVER.m_DeviceNode.m_dCurrVelx, sizeof(float)*nMaxNode);
	cudaMalloc(&SOLVER.m_DeviceNode.m_dCurrVely, sizeof(float)*nMaxNode);
	cudaMalloc(&SOLVER.m_DeviceNode.m_dCurrVelz, sizeof(float)*nMaxNode);

	cudaMalloc(&SOLVER.m_DeviceNode.m_dPrevVelx, sizeof(float)*nMaxNode);
	cudaMalloc(&SOLVER.m_DeviceNode.m_dPrevVely, sizeof(float)*nMaxNode);
	cudaMalloc(&SOLVER.m_DeviceNode.m_dPrevVelz, sizeof(float)*nMaxNode);

	cudaMalloc(&SOLVER.m_DeviceNode.m_dTauEddy, sizeof(float)*nMaxNode);

	cudaMalloc(&SOLVER.m_DeviceNode.m_nNodeType, sizeof(int)*nMaxNode);
	cudaMalloc(&SOLVER.m_DeviceNode.m_nDirectionState, sizeof(int)*nMaxNode);

	// 변수 초기화
	cudaMemset(SOLVER.m_DeviceNode.m_arF, 0, sizeof(float)*nMaxNode*DIRECTION_SIZE);
	cudaMemset(SOLVER.m_DeviceNode.m_arFTemp, 0, sizeof(float)*nMaxNode*DIRECTION_SIZE);

	cudaMemset(SOLVER.m_DeviceNode.m_dCurrDen, 0, sizeof(float)*nMaxNode);
	cudaMemset(SOLVER.m_DeviceNode.m_dPrevDen, 0, sizeof(float)*nMaxNode);

	cudaMemset(SOLVER.m_DeviceNode.m_dCurrVelx, 0, sizeof(float)*nMaxNode);
	cudaMemset(SOLVER.m_DeviceNode.m_dCurrVely, 0, sizeof(float)*nMaxNode);
	cudaMemset(SOLVER.m_DeviceNode.m_dCurrVelz, 0, sizeof(float)*nMaxNode);

	cudaMemset(SOLVER.m_DeviceNode.m_dPrevVelx, 0, sizeof(float)*nMaxNode);
	cudaMemset(SOLVER.m_DeviceNode.m_dPrevVely, 0, sizeof(float)*nMaxNode);
	cudaMemset(SOLVER.m_DeviceNode.m_dPrevVelz, 0, sizeof(float)*nMaxNode);

	cudaMemset(SOLVER.m_DeviceNode.m_dTauEddy, 0, sizeof(float)*nMaxNode);

	cudaMemset(SOLVER.m_DeviceNode.m_nNodeType, 0, sizeof(int)*nMaxNode);
	cudaMemset(SOLVER.m_DeviceNode.m_nDirectionState, 0, sizeof(int)*nMaxNode);

	INPUT.icycle = 0;

    FILE* pFile0 = fopen("iteration.dat", "wt");
    fprintf(pFile0, "%s\n", "iter, error");

	//main iteration
	for (int m_itr = 0; m_itr < INPUT.maxitr; ++m_itr)
	{
		if (INPUT.icycle == 0)
		{
			keInitialNodeDirection << <gridSize, blockSize >> > (SOLVER.m_DeviceNode.m_nDirectionState, nNodeSize, SOLVER.m_DeviceDirection, nMaxNode);

			keSetNodeType << < gridSize, blockSize >> > (SOLVER.m_DeviceNode.m_nNodeType, nNodeSize, nMaxNode, INPUT.casetype);

			keInitialize << < gridSize, blockSize >> > (SOLVER.m_DeviceNode.m_nNodeType, SOLVER.m_DeviceNode.m_dCurrDen, SOLVER.m_DeviceNode.m_dCurrVelx, SOLVER.m_DeviceNode.m_dCurrVely, SOLVER.m_DeviceNode.m_dCurrVelz,
				SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceW, SOLVER.m_DeviceDirection, INPUT.m_dRefDen, INPUT.m_dRefVelx, INPUT.m_dRefVely, INPUT.m_dRefVelz, nMaxNode);
		}
		
		else
		{
			if (INPUT.bEnableLES)
			{
				keComputeTurbulence_LES << < gridSize, blockSize >> > (SOLVER.m_DeviceNode.m_nNodeType, SOLVER.m_DeviceNode.m_dCurrDen, SOLVER.m_DeviceNode.m_dCurrVelx, SOLVER.m_DeviceNode.m_dCurrVely, SOLVER.m_DeviceNode.m_dCurrVelz,
					SOLVER.m_DeviceNode.m_dTauEddy, SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceW, SOLVER.m_DeviceDirection, INPUT.m_dTau, INPUT.m_dSmagorinsky, nMaxNode);
			}

			if (INPUT.bEnableMRT)
			{
				keCollision_MRT << <gridSize, blockSize >> > (SOLVER.m_DeviceNode.m_nNodeType, SOLVER.m_DeviceNode.m_dCurrDen, SOLVER.m_DeviceNode.m_dCurrVelx, SOLVER.m_DeviceNode.m_dCurrVely, SOLVER.m_DeviceNode.m_dCurrVelz, SOLVER.m_DeviceNode.m_dTauEddy,
					SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceNode.m_arFTemp, INPUT.m_dRefDen, INPUT.m_dTau, INPUT.dWeps, INPUT.dWepsj, INPUT.dWXX, SOLVER.m_DeviceMatrix, nMaxNode);
			}
			else
			{
				keCollision_SRT << <gridSize, blockSize >> > (SOLVER.m_DeviceNode.m_nNodeType, SOLVER.m_DeviceNode.m_dCurrDen, SOLVER.m_DeviceNode.m_dCurrVelx, SOLVER.m_DeviceNode.m_dCurrVely, SOLVER.m_DeviceNode.m_dCurrVelz,
					SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceNode.m_arFTemp, SOLVER.m_DeviceW, SOLVER.m_DeviceDirection, INPUT.m_dTau, SOLVER.m_DeviceNode.m_dTauEddy, nMaxNode);
			}
			keStreaming << <gridSize, blockSize >> > (SOLVER.m_DeviceNode.m_nNodeType, SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceNode.m_arFTemp, SOLVER.m_DeviceNode.m_nDirectionState, SOLVER.m_DeviceMove, nMaxNode);

			if (INPUT.casetype == CAVITY)
			{
				keBoundaryCavity << <gridSize, blockSize >> > (nNodeSize, SOLVER.m_DeviceNode.m_nNodeType, INPUT.m_dRefVelx, INPUT.m_dRefVely, INPUT.m_dRefVelz, SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceNode.m_arFTemp, SOLVER.m_DeviceW, SOLVER.m_DeviceDirection, SOLVER.m_DeviceNode.m_nDirectionState, SOLVER.m_DeviceOppIndex, INPUT.m_dRefDen, nMaxNode);
			}
			else if (INPUT.casetype == POISEUILLE)
			{
				//keBoundaryCavity << <gridSize, blockSize >> > (nNodeSize, SOLVER.m_DeviceNode.m_nNodeType, INPUT.m_dRefVelx, INPUT.m_dRefVely, INPUT.m_dRefVelz, SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceNode.m_arFTemp, SOLVER.m_DeviceW, SOLVER.m_DeviceDirection, SOLVER.m_DeviceNode.m_nDirectionState, SOLVER.m_DeviceOppIndex, INPUT.m_dRefDen, nMaxNode);
				keBoundaryStruct << <gridSize, blockSize >> > (nNodeSize, SOLVER.m_DeviceNode.m_nNodeType, SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceNode.m_arFTemp, SOLVER.m_DeviceNode.m_nDirectionState, SOLVER.m_DeviceOppIndex, SOLVER.m_DeviceMove, nMaxNode);
				keBoundaryInlet << <gridSize, blockSize >> > (nNodeSize, SOLVER.m_DeviceNode.m_dCurrDen, SOLVER.m_DeviceNode.m_nNodeType, INPUT.m_dRefVelx, INPUT.m_dRefVely, INPUT.m_dRefVelz, SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceNode.m_arFTemp, SOLVER.m_DeviceW, SOLVER.m_DeviceDirection, SOLVER.m_DeviceNode.m_nDirectionState, SOLVER.m_DeviceOppIndex, SOLVER.m_DeviceMove, nMaxNode);
				keBoundaryOutlet << <gridSize, blockSize >> > (nNodeSize, SOLVER.m_DeviceNode.m_dCurrVelx, SOLVER.m_DeviceNode.m_dCurrVely, SOLVER.m_DeviceNode.m_dCurrVelz, SOLVER.m_DeviceNode.m_nNodeType, SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceNode.m_arFTemp, SOLVER.m_DeviceW, SOLVER.m_DeviceDirection, SOLVER.m_DeviceNode.m_nDirectionState, SOLVER.m_DeviceOppIndex, INPUT.m_dRefDen, SOLVER.m_DeviceMove, nMaxNode);

			}
			keMacroscopic << <gridSize, blockSize >> > (SOLVER.m_DeviceNode.m_nNodeType, SOLVER.m_DeviceNode.m_dCurrDen, SOLVER.m_DeviceNode.m_dPrevDen, SOLVER.m_DeviceNode.m_dCurrVelx, SOLVER.m_DeviceNode.m_dCurrVely, SOLVER.m_DeviceNode.m_dCurrVelz,
				SOLVER.m_DeviceNode.m_arF, SOLVER.m_DeviceDirection, nMaxNode);

		}

		cudaDeviceSynchronize();

		if (INPUT.icycle % INPUT.nwrite == 0)
		{
			SOLVER.m_HostNode.m_dCurrDen.clear();
			SOLVER.m_HostNode.m_dCurrVelx.clear();
			SOLVER.m_HostNode.m_dCurrVely.clear();
			SOLVER.m_HostNode.m_dCurrVelz.clear();
			SOLVER.m_HostNode.m_nNodeType.clear();

			
			SOLVER.m_HostNode.m_dCurrDen.resize(nMaxNode);
			SOLVER.m_HostNode.m_dCurrVelx.resize(nMaxNode);
			SOLVER.m_HostNode.m_dCurrVely.resize(nMaxNode);
			SOLVER.m_HostNode.m_dCurrVelz.resize(nMaxNode);
			SOLVER.m_HostNode.m_nNodeType.resize(nMaxNode);

			cudaMemcpy(&SOLVER.m_HostNode.m_dCurrDen.front(), SOLVER.m_DeviceNode.m_dCurrDen, sizeof(float)*nMaxNode, cudaMemcpyDeviceToHost);
			cudaMemcpy(&SOLVER.m_HostNode.m_dCurrVelx.front(), SOLVER.m_DeviceNode.m_dCurrVelx, sizeof(float)*nMaxNode, cudaMemcpyDeviceToHost);
			cudaMemcpy(&SOLVER.m_HostNode.m_dCurrVely.front(), SOLVER.m_DeviceNode.m_dCurrVely, sizeof(float)*nMaxNode, cudaMemcpyDeviceToHost);
			cudaMemcpy(&SOLVER.m_HostNode.m_dCurrVelz.front(), SOLVER.m_DeviceNode.m_dCurrVelz, sizeof(float)*nMaxNode, cudaMemcpyDeviceToHost);

			cudaMemcpy(&SOLVER.m_HostNode.m_nNodeType.front(), SOLVER.m_DeviceNode.m_nNodeType, sizeof(int)*nMaxNode, cudaMemcpyDeviceToHost);

			saveFile(pinput, psolver, arIndex, INPUT.savetype);
			//printf("File saving: %d/%d is completed\n", INPUT.icycle / INPUT.nwrite, INPUT.maxitr / INPUT.nwrite);
		}
		//std::cout << INPUT.icycle << std::endl;

        if (INPUT.icycle % 100 == 0)
        {
            //Print ERROR!!
            SOLVER.m_HostNode.m_dCurrDen.clear();
            SOLVER.m_HostNode.m_dPrevDen.clear();
            SOLVER.m_HostNode.m_dCurrDen.resize(nMaxNode);
            SOLVER.m_HostNode.m_dPrevDen.resize(nMaxNode);
            cudaMemcpy(&SOLVER.m_HostNode.m_dCurrDen.front(), SOLVER.m_DeviceNode.m_dCurrDen, sizeof(float)*nMaxNode, cudaMemcpyDeviceToHost);
            cudaMemcpy(&SOLVER.m_HostNode.m_dPrevDen.front(), SOLVER.m_DeviceNode.m_dPrevDen, sizeof(float)*nMaxNode, cudaMemcpyDeviceToHost);
            float ERROR = 0.0;
            for (int nNodeID = 0; nNodeID < nMaxNode; nNodeID++)
            {
                ERROR += abs(SOLVER.m_HostNode.m_dPrevDen[nNodeID] - SOLVER.m_HostNode.m_dCurrDen[nNodeID]);
            }
            printf("ITER=%d | ERROR=%.5e\n",INPUT.icycle, ERROR);
            fprintf(pFile0,"%d %.5e\n",INPUT.icycle, ERROR);
        }


		/*
        SOLVER.m_HostNode.m_dCurrVelx.clear();
        SOLVER.m_HostNode.m_nNodeType.clear();

        SOLVER.m_HostNode.m_dCurrVelx.resize(nMaxNode);
        SOLVER.m_HostNode.m_nNodeType.resize(nMaxNode);

        cudaMemcpy(&SOLVER.m_HostNode.m_dCurrVelx.front(), SOLVER.m_DeviceNode.m_dCurrVelx, sizeof(float)*nMaxNode, cudaMemcpyDeviceToHost);
        cudaMemcpy(&SOLVER.m_HostNode.m_nNodeType.front(), SOLVER.m_DeviceNode.m_nNodeType, sizeof(int)*nMaxNode, cudaMemcpyDeviceToHost);

        for (int nNodeID = 0; nNodeID < nMaxNode; nNodeID++)
        {
            int nSlave = nNodeSize.x * nNodeSize.z;

            int nX = (nNodeID % nSlave) % nNodeSize.x;
            int nY = (nNodeID / nSlave);
            int nZ = (nNodeID % nSlave) / nNodeSize.x;
            if(nX==40 && nY==40 && nZ==40) printf("ITER=%d | VELX=%lf\n",m_itr,SOLVER.m_HostNode.m_dCurrVelx[nNodeID]);
        }
		*/



		INPUT.icycle++;
		//printf("%d times\n", m_itr + 1);

	}
	fclose(pFile0);

}

__global__ void keInitialize(int* nNodeType, float* CurrDen, float *CurrVelx, float *CurrVely, float *CurrVelz, float *arF, float* arW, float* Direction, float RefDen, float RefVelx, float RefVely, float RefVelz, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];

	float den = RefDen;
	float velx = 0.0;
	float vely = 0.0;
	float velz = 0.0;

	if (nodetype == NODE_STRUCT) return;

	if (nodetype == NODE_MOVINGWALL)
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
		arF[nDir*nMaxNode + nNodeID] = arW[nDir] * den * (1.0 + 3.0*edotu + 4.5*edotu*edotu - 1.5*usquare);
	}

	CurrDen[nNodeID] = den;
	CurrVelx[nNodeID] = velx;
	CurrVely[nNodeID] = vely;
	CurrVelz[nNodeID] = velz;
}

__global__ void keMacroscopic(int* nNodeType, float *CurrDen, float *PrevDen, float *CurrVelx, float *CurrVely, float *CurrVelz, float *arF, float* Direction, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];
	if (nodetype == NODE_STRUCT) return;

	float den = 0;
	float velx = 0;
	float vely = 0;
	float velz = 0;
	float f[DIRECTION_SIZE] = { 0.0, };

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		f[nDir] = arF[nDir*nMaxNode + nNodeID];
	}

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		den += f[nDir];
		velx += f[nDir] * Direction[XDIM*DIRECTION_SIZE + nDir];
		vely += f[nDir] * Direction[YDIM*DIRECTION_SIZE + nDir];
		velz += f[nDir] * Direction[ZDIM*DIRECTION_SIZE + nDir];
	}

	velx /= den;
	vely /= den;
	velz /= den;

	PrevDen[nNodeID] = CurrDen[nNodeID];
	CurrDen[nNodeID] = den;
	CurrVelx[nNodeID] = velx;
	CurrVely[nNodeID] = vely;
	CurrVelz[nNodeID] = velz;
}

__global__ void keCollision_MRT(int* nNodeType, float* CurrDen, float* CurrVelx, float* CurrVely, float* CurrVelz, float* dTauEddy,
	float*	arF, float* arFTemp, float RefDen, float dTau, float dWeps, float dWepsj, float dWXX, MRTMatrix* Matrix, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];
	if (nodetype == NODE_STRUCT) return;

	float den = CurrDen[nNodeID];
	float dtaueddy = dTauEddy[nNodeID];

	float3 vel = make_float3(CurrVelx[nNodeID], CurrVely[nNodeID], CurrVelz[nNodeID]);
	float3 jVector = den * vel;

	float f[DIRECTION_SIZE] = { 0.0, };
	float ftemp[DIRECTION_SIZE] = { 0.0, };

	for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
	{
		f[nDir] = arF[nDir * nMaxNode + nNodeID];
	}

	float matMeq[DIRECTION_SIZE] = { 0.0, };
	float matS[DIRECTION_SIZE] = { 0.0, };
	float matM1[DIRECTION_SIZE] = { 0.0, };

	matMeq[0] = den;
	matMeq[1] = -11.0 *den + 19.0 / RefDen * dot(jVector, jVector);
	matMeq[2] = dWeps * den + dWepsj / RefDen * dot(jVector, jVector);
	matMeq[3] = jVector.x;
	matMeq[4] = -2.0 / 3.0 * jVector.x;
	matMeq[5] = jVector.y;
	matMeq[6] = -2.0 / 3.0 * jVector.y;
	matMeq[7] = jVector.z;
	matMeq[8] = -2.0 / 3.0 * jVector.z;
	matMeq[9] = 1.0 / RefDen * (2.0 * pow(jVector.x, 2) - (pow(jVector.y, 2) + pow(jVector.z, 2)));
	matMeq[10] = dWXX * matMeq[9];
	matMeq[11] = 1.0 / RefDen * (pow(jVector.y, 2) - pow(jVector.z, 2));
	matMeq[12] = dWXX * matMeq[11];
	matMeq[13] = jVector.x * jVector.y / RefDen;
	matMeq[14] = jVector.y * jVector.z / RefDen;
	matMeq[15] = jVector.z * jVector.x / RefDen;
	matMeq[16] = 0.0;
	matMeq[17] = 0.0;
	matMeq[18] = 0.0;

	matS[0] = 0.0;
	matS[1] = 1.19;
	matS[2] = 1.4;
	matS[3] = 0.0;
	matS[4] = 1.2;
	matS[5] = 0.0;
	matS[6] = 1.2;
	matS[7] = 0.0;
	matS[8] = 1.2;
	matS[9] = 1.0 / (dTau + dtaueddy);
	matS[10] = 1.4;
	matS[11] = 1.0 / (dTau + dtaueddy);
	matS[12] = 1.4;
	matS[13] = 1.0 / (dTau + dtaueddy);
	matS[14] = 1.0 / (dTau + dtaueddy);
	matS[15] = 1.0 / (dTau + dtaueddy);
	matS[16] = 1.98;
	matS[17] = 1.98;
	matS[18] = 1.98;

	for (int nDir_I = 0; nDir_I < DIRECTION_SIZE; ++nDir_I)
	{
		matM1[nDir_I] = 0.0;
		// velocity moment 계산
		for (int nDir_J = 0; nDir_J < DIRECTION_SIZE; ++nDir_J)
		{
			matM1[nDir_I] += Matrix->m_MatMRT[nDir_I][nDir_J] * f[nDir_J];
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
			matM1[nDir_I] += Matrix->m_MatInv[nDir_I][nDir_J] * ftemp[nDir_J];
		}
	}

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		ftemp[nDir] = f[nDir] - matM1[nDir];
	}

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		arFTemp[nDir * nMaxNode + nNodeID] = ftemp[nDir];
	}

}

__global__ void keCollision_SRT(int* nNodeType, float *CurrDen, float *CurrVelx, float *CurrVely, float *CurrVelz, float *arF, float *arFTemp, float* arW, float* Direction, float dTau, float* dTauEddy, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];
	if (nodetype == NODE_STRUCT) return;

	float den = CurrDen[nNodeID];
	float velx = CurrVelx[nNodeID];
	float vely = CurrVely[nNodeID];
	float velz = CurrVelz[nNodeID];
	float taueddy = dTauEddy[nNodeID];

	float f[DIRECTION_SIZE] = { 0.0, };

	for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
	{
		f[nDir] = arF[nDir*nMaxNode + nNodeID];
	}

	for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
	{
		float edotu = Direction[XDIM*DIRECTION_SIZE + nDir] * velx + Direction[YDIM*DIRECTION_SIZE + nDir] * vely + Direction[ZDIM*DIRECTION_SIZE + nDir] * velz;
		float usquare = velx * velx + vely * vely + velz * velz;
		float feq = arW[nDir] * den * (1.0 + 3.0*edotu + 4.5*edotu * edotu - 1.5*usquare);

		arFTemp[nDir*nMaxNode + nNodeID] = f[nDir] - (f[nDir] - feq) / (dTau + taueddy);
	}
}

__global__ void keStreaming(int* nNodeType, float *arF, float *arFTemp, int* DirectionState, int* Move, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];
	if (nodetype == NODE_STRUCT) return;

	int dirstate = DirectionState[nNodeID];
	float ftemp[DIRECTION_SIZE] = { 0.0, };

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		ftemp[nDir] = arFTemp[nDir*nMaxNode + nNodeID];
	}

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		if ((dirstate >> nDir) & 1) 
		{
			int nMoveNodeID = nNodeID + Move[nDir];
			int movenodetype = nNodeType[nMoveNodeID];
			if (movenodetype == NODE_STRUCT) continue;

			arF[nDir*nMaxNode + nMoveNodeID] = ftemp[nDir];
		}
	}
}

__global__ void keBoundaryCavity(int3 nNodeSize, int* nNodeType, float RefVelx, float RefVely, float RefVelz, float *arF, float *arFTemp, float* arW, float* Direction, int* DirectionState, int* nOppIndex, float RefDen, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];
	int dirstate = DirectionState[nNodeID];

	if (nodetype == NODE_FLUID) return;

	int nSlave = nNodeSize.x * nNodeSize.z;

	int x = (nNodeID % nSlave) % nNodeSize.x;
	int y = (nNodeID / nSlave);
	int z = (nNodeID % nSlave) / nNodeSize.x;

	float velx = RefVelx;
	float vely = RefVely;
	float velz = RefVelz;

	float f[DIRECTION_SIZE] = { 0.0, };
	float ftemp[DIRECTION_SIZE] = { 0.0, };

	for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
	{
		ftemp[nDir] = arFTemp[nDir*nMaxNode + nNodeID];
	}

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		if (!((dirstate >> nDir) & 1)) 
		{
			int nOPP = nOppIndex[nDir];

			if (nodetype == NODE_FIXEDWALL)
			{
				arF[nOPP*nMaxNode + nNodeID] = ftemp[nDir];
			}
			else if (nodetype == NODE_MOVINGWALL)
			{
				float edotu = Direction[XDIM*DIRECTION_SIZE + nDir] * velx + Direction[YDIM*DIRECTION_SIZE + nDir] * vely + Direction[ZDIM*DIRECTION_SIZE + nDir] * velz;

				arF[nOPP*nMaxNode + nNodeID] = ftemp[nDir] - 6.0*arW[nDir] * RefDen * edotu;
			}
		}
	}
}

__global__ void keBoundaryInlet(int3 nNodeSize, float* CurrDen, int* nNodeType, float RefVelx, float RefVely, float RefVelz, float *arF, float *arFTemp, float* arW, float* Direction, int* DirectionState, int* nOppIndex, int* Move, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];

	if ((nodetype & NODE_INLET) == 0) return;

	int dirstate = DirectionState[nNodeID];

	int nSlave = nNodeSize.x * nNodeSize.z;

	int x = (nNodeID % nSlave) % nNodeSize.x;
	int y = (nNodeID / nSlave);
	int z = (nNodeID % nSlave) / nNodeSize.x;

	float f[DIRECTION_SIZE] = { 0.0, };
	float ftemp[DIRECTION_SIZE] = { 0.0, };

	for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
	{
		ftemp[nDir] = arFTemp[nDir*nMaxNode + nNodeID];
	}

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		if (!((dirstate >> nDir) & 1)) 
		{
			int nOPP = nOppIndex[nDir];
			int nInerNodeID = nNodeID + Move[1];
			float inerden = CurrDen[nInerNodeID];

			//if (nodetype == NODE_INLET)
			{
				float velx = RefVelx;
				float vely = RefVely;
				float velz = RefVelz;

				float edotu = Direction[XDIM*DIRECTION_SIZE + nDir] * velx + Direction[YDIM*DIRECTION_SIZE + nDir] * vely + Direction[ZDIM*DIRECTION_SIZE + nDir] * velz;

				arF[nOPP*nMaxNode + nNodeID] = ftemp[nDir] - 6.0*arW[nDir] * inerden * edotu;

			}

		}
	}
}

__global__ void keBoundaryOutlet(int3 nNodeSize, float *CurrVelx, float *CurrVely, float *CurrVelz, int* nNodeType, float *arF, float *arFTemp, float* arW, float* Direction, int* DirectionState, int* nOppIndex, float RefDen, int* Move, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];

	if ((nodetype & NODE_OUTLET) == 0) return;

	int dirstate = DirectionState[nNodeID];

	int nSlave = nNodeSize.x * nNodeSize.z;

	int x = (nNodeID % nSlave) % nNodeSize.x;
	int y = (nNodeID / nSlave);
	int z = (nNodeID % nSlave) / nNodeSize.x;

	float ftemp[DIRECTION_SIZE] = { 0.0, };

	for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
	{
		ftemp[nDir] = arFTemp[nDir*nMaxNode + nNodeID];
	}

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		if (!((dirstate >> nDir) & 1))
		{
			{
				int nOPP = nOppIndex[nDir];
				int nInerNodeID = nNodeID + Move[2];
				float inervelx = CurrVelx[nNodeID] + 0.5 * (CurrVelx[nNodeID] - CurrVelx[nInerNodeID]);
				float inervely = CurrVely[nNodeID] + 0.5 * (CurrVely[nNodeID] - CurrVely[nInerNodeID]);
				float inervelz = CurrVelz[nNodeID] + 0.5 * (CurrVelz[nNodeID] - CurrVelz[nInerNodeID]);

				float usquare = inervelx * inervelx + inervely * inervely + inervelz * inervelz;

				float edotu = Direction[XDIM*DIRECTION_SIZE + nDir] * inervelx + Direction[YDIM*DIRECTION_SIZE + nDir] * inervely + Direction[ZDIM*DIRECTION_SIZE + nDir] * inervelz;
				arF[nOPP*nMaxNode + nNodeID] = -ftemp[nDir] + 2.0 * arW[nDir] * RefDen * (1.0 + 4.5*pow(edotu, 2) - 1.5*usquare);
			}
		}
	}
}

__global__ void keBoundaryStruct(int3 nNodeSize, int* nNodeType, float *arF, float *arFTemp, int* DirectionState, int* nOppIndex, int* Move, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];
	int dirstate = DirectionState[nNodeID];

	if ((nodetype & NODE_FLUID) == 0) return;

	int nSlave = nNodeSize.x * nNodeSize.z;

	int x = (nNodeID % nSlave) % nNodeSize.x;
	int y = (nNodeID / nSlave);
	int z = (nNodeID % nSlave) / nNodeSize.x;

	float ftemp[DIRECTION_SIZE] = { 0.0, };

	for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
	{
		ftemp[nDir] = arFTemp[nDir*nMaxNode + nNodeID];
	}

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		if ((dirstate >> nDir) & 1)
		{
			int nMoveNodeID = nNodeID + Move[nDir];
			int movenodetype = nNodeType[nMoveNodeID];


			if (movenodetype == NODE_STRUCT)
			{
				int nOPP = nOppIndex[nDir];
				arF[nOPP*nMaxNode + nNodeID] = ftemp[nDir];
			}
		}
	}
}

void saveFile(Inputval *pinput, LBMSolver *psolver, std::vector<int> arIndex, int savetype)
{
	//std::cout << savetype << std::endl;
	// Binary 형식의 Tecplot 파일 저장
	if (savetype == BINARYTYPE)
	{
		char  name[1024] = {};
		sprintf(name, "%sOutput%06d.vtu", SAVE_PATH, (pinput->icycle - pinput->irestart) / pinput->nwrite);
		std::cout << name << std::endl;
		FILE* pFile = NULL;
		pFile = fopen(name, "wb");

		if (pFile == nullptr) exit(0);

		std::string str;
		str.clear();

		int VALUE_SIZE = 8;
		int INT_SIZE = sizeof(int32_t);
		int SIZE_T_SIZE = 8;

		// Header 전에 각 항목의 크기 미리 계산함		
		int nPointCount = INPUT.length;
		int nGridIndexCount = arIndex.size();
		int nGridCount = arIndex.size() / 8;

		size_t point_Byte = nPointCount * VALUE_SIZE * 3;
		size_t connectivity_Byte = nGridIndexCount * INT_SIZE;
		size_t offset_Byte = nGridCount * INT_SIZE;
		size_t type_Byte = nGridCount * sizeof(uint8_t);
		size_t nSingleData_Byte = nPointCount * VALUE_SIZE;
		size_t nVec3Data_Byte = nSingleData_Byte * 3;
		size_t nFlagData_Byte = nPointCount * INT_SIZE;

		size_t offset = 0;

		///// Header 시작 /////
		str = "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		str = "<UnstructuredGrid>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		///// Piece /////
		//// 점 개수, 격자 개수 정의 ///
		str = std::string("<Piece NumberOfPoints=\"") + std::to_string(nPointCount) + std::string("\" NumberOfCells=\"") + std::to_string(nGridCount) + std::string("\">\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		///// Points /////
		//// 점의 좌표 정의 ///
		str = "<Points>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		//4개 격자점의 4byte로 이루어진 3차원 자표 정보
		str = std::string("<DataArray type=\"Float64\" Name=\"\" NumberOfComponents=\"3\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = point_Byte + SIZE_T_SIZE;
		str.clear();

		str = "</Points>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		///// cells /////
		//// 셀의 정보 정의 ///
		str = "<Cells>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		//4개 격자점의 4byte로 이루어진 연결성 정보
		str = std::string("<DataArray type=\"Int32\" Name=\"connectivity\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + connectivity_Byte + SIZE_T_SIZE;
		str.clear();

		//CSR format의 cummulant 자료 구조
		str = std::string("<DataArray type=\"Int32\" Name=\"offsets\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + offset_Byte + SIZE_T_SIZE;
		str.clear();

		//각 요소들의 type
		str = std::string("<DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + type_Byte + SIZE_T_SIZE;
		str.clear();

		str = "</Cells>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		///// PointData /////
		//// 각 점의 값들 정의 (밀도, 속도, 압력 등) ///
		str = "<PointData>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		// 저장하는 정보 기입
		// Velocity
		str = std::string("<DataArray type = \"Float64\" Name=\"Velocity\" NumberOfComponents= \"3\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + nVec3Data_Byte + SIZE_T_SIZE;
		str.clear();

		str = std::string("<DataArray type = \"Float64\" Name=\"Density\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + nSingleData_Byte + SIZE_T_SIZE;
		str.clear();

		str = std::string("<DataArray type = \"Int32\" Name=\"Fluid\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + nFlagData_Byte + SIZE_T_SIZE;
		str.clear();

		str = std::string("<DataArray type = \"Int32\" Name=\"Fixed Wall\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + nFlagData_Byte + SIZE_T_SIZE;
		str.clear();

		str = std::string("<DataArray type = \"Int32\" Name=\"Moving Wall\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + nFlagData_Byte + SIZE_T_SIZE;
		str.clear();

		str = std::string("<DataArray type = \"Int32\" Name=\"Inlet\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + nFlagData_Byte + SIZE_T_SIZE;
		str.clear();

		str = std::string("<DataArray type = \"Int32\" Name=\"Outlet\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + nFlagData_Byte + SIZE_T_SIZE;
		str.clear();

		str = std::string("<DataArray type = \"Int32\" Name=\"Struct\" NumberOfComponents= \"1\" offset=\"") + std::to_string(offset) + std::string("\" format=\"appended\"/>\n");
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		offset = offset + nFlagData_Byte + SIZE_T_SIZE;
		str.clear();

		str = "</PointData>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		str = "</Piece>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		str = "</UnstructuredGrid>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();
		///// Header 종료 /////


		int DOUBLE_BYTE = 8;
		int INT_BYTE = 4;
		///// Data /////
		str = "<AppendedData encoding=\"raw\">\n_";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		// Points data //
		fwrite(&point_Byte, SIZE_T_SIZE, 1, pFile);

		int3 nNodeSize = { INPUT.nx, INPUT.ny, INPUT.nz };
		int nSlave = nNodeSize.x * nNodeSize.z;

		for (int i = 0; i < nPointCount; i++)
		{
			float x = (i % nSlave) % nNodeSize.x;
			float y = (i / nSlave);
			float z = (i % nSlave) / nNodeSize.x;
			fwrite(&x, VALUE_SIZE, 1, pFile);
			fwrite(&y, VALUE_SIZE, 1, pFile);
			fwrite(&z, VALUE_SIZE, 1, pFile);
		}

		// Cells data //
		// connectivity
		fwrite(&connectivity_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nGridIndexCount; i++)
			fwrite(&arIndex[i], INT_SIZE, 1, pFile);

		std::vector<int> arOffset; arOffset.resize(nGridCount);
		std::vector<int> arType;   arType.resize(nGridCount);
		for (int i = 0; i < nGridCount; i++)
		{
			arOffset[i] = 8 + i * 8;
			arType[i] = 12;
		}

		// offsets
		fwrite(&offset_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nGridCount; i++)
			fwrite(&arOffset[i], INT_SIZE, 1, pFile);

		// types
		fwrite(&type_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nGridCount; i++)
			fwrite(&arType[i], sizeof(uint8_t), 1, pFile);

		fwrite(&nVec3Data_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nPointCount; i++)
		{
			fwrite(&SOLVER.m_HostNode.m_dCurrVelx[i], VALUE_SIZE, 1, pFile);
			fwrite(&SOLVER.m_HostNode.m_dCurrVely[i], VALUE_SIZE, 1, pFile);
			fwrite(&SOLVER.m_HostNode.m_dCurrVelz[i], VALUE_SIZE, 1, pFile);
		}

		// density
		fwrite(&nSingleData_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nPointCount; i++)
			fwrite(&SOLVER.m_HostNode.m_dCurrDen[i], VALUE_SIZE, 1, pFile);

		int nFlag = 0;

		fwrite(&nFlagData_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nPointCount; i++)
		{
			nFlag = SOLVER.m_HostNode.m_nNodeType[i] & NODE_FLUID ? 1 : 0;
			fwrite(&nFlag, INT_SIZE, 1, pFile);
		}

		fwrite(&nFlagData_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nPointCount; i++)
		{
			nFlag = SOLVER.m_HostNode.m_nNodeType[i] & NODE_FIXEDWALL ? 1 : 0;
			fwrite(&nFlag, INT_SIZE, 1, pFile);
		}

		fwrite(&nFlagData_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nPointCount; i++)
		{
			nFlag = SOLVER.m_HostNode.m_nNodeType[i] & NODE_MOVINGWALL ? 1 : 0;
			fwrite(&nFlag, INT_SIZE, 1, pFile);
		}

		fwrite(&nFlagData_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nPointCount; i++)
		{
			nFlag = SOLVER.m_HostNode.m_nNodeType[i] & NODE_INLET ? 1 : 0;
			fwrite(&nFlag, INT_SIZE, 1, pFile);
		}

		fwrite(&nFlagData_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nPointCount; i++)
		{
			nFlag = SOLVER.m_HostNode.m_nNodeType[i] & NODE_OUTLET ? 1 : 0;
			fwrite(&nFlag, INT_SIZE, 1, pFile);
		}

		fwrite(&nFlagData_Byte, SIZE_T_SIZE, 1, pFile);
		for (int i = 0; i < nPointCount; i++)
		{
			nFlag = SOLVER.m_HostNode.m_nNodeType[i] & NODE_STRUCT ? 1 : 0;
			fwrite(&nFlag, INT_SIZE, 1, pFile);
		}

		// 파일 종료
		str = "</AppendedData>\n";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		str = "</VTKFile>";
		fwrite(str.c_str(), sizeof(char), str.size(), pFile);
		str.clear();

		fclose(pFile);
	}

	// CSV 형식의 파일저장
	else if (savetype == CSVTYPE)
	{	
		// char  name[1024] = {};
		// sprintf(name, "%sOutput%06d.vtu", SAVE_PATH, (INPUT.icycle - INPUT.irestart) / INPUT.nwrite);
		FILE* pFile = fopen("result.csv", "wt");

		if (pFile == nullptr) exit(0);

		int nMaxNode = INPUT.length;
		int	blockSize = 512;
		int3 nNodeSize = { INPUT.nx, INPUT.ny, INPUT.nz };

		fprintf(pFile, "%s\n", "x,y,z,rho,u,v,w");
		for (int nNodeID = 0; nNodeID < nMaxNode; nNodeID++)
		{
			int nSlave = nNodeSize.x * nNodeSize.z;

			int nX = (nNodeID % nSlave) % nNodeSize.x;
			int nY = (nNodeID / nSlave);
			int nZ = (nNodeID % nSlave) / nNodeSize.x;
			//                 x    y   z, rho   u, v, w
			fprintf(pFile, "%d, %d, %d, %lf, %lf, %lf, %lf\n",
				nX, nY, nZ, SOLVER.m_HostNode.m_dCurrDen[nNodeID], SOLVER.m_HostNode.m_dCurrVelx[nNodeID], SOLVER.m_HostNode.m_dCurrVely[nNodeID], SOLVER.m_HostNode.m_dCurrVelz[nNodeID]);
		}
		fclose(pFile);

		FILE* pFile1 = fopen("section.csv", "wt");
		if (pFile1 == nullptr) exit(0);

		fprintf(pFile1, "%s\n", "y/Y,u");
		for (int nNodeID = 0; nNodeID < nMaxNode; nNodeID++)
		{
			int nSlave = nNodeSize.x * nNodeSize.z;

			int nX = (nNodeID % nSlave) % nNodeSize.x;
			int nY = (nNodeID / nSlave);
			int nZ = (nNodeID % nSlave) / nNodeSize.x;
			if(nZ == 49 && nX == 49) {
			//                  y   u/U
				fprintf(pFile1, "%d, %lf\n", nY, SOLVER.m_HostNode.m_dCurrVelx[nNodeID]*10);
			}
		}
		fclose(pFile1);

	}
}

__global__ void keInitialNodeDirection(int* DirectionState, int3 nNodeSize, float* Direction, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nSlave = nNodeSize.x * nNodeSize.z;

	int dirstate = 0;
	for (int nDir = DIRECTION_SIZE - 1; nDir > 0; --nDir)
	{
		int x = (nNodeID % nSlave) % nNodeSize.x + Direction[XDIM*DIRECTION_SIZE + nDir];
		int y = (nNodeID / nSlave) + Direction[YDIM*DIRECTION_SIZE + nDir];
		int z = (nNodeID % nSlave) / nNodeSize.x + Direction[ZDIM*DIRECTION_SIZE + nDir];

		if (x < nNodeSize.x && y < nNodeSize.y && z < nNodeSize.z && x >= 0 && y >= 0 && z >= 0)
		{
			dirstate = (dirstate | 1) << 1; 	//InSide 
		}
		else
		{
			dirstate = dirstate << 1; 			//OutSide
		}
	}
	DirectionState[nNodeID] = (dirstate | 1); 
}

__global__ void keSetNodeType(int* NodeType, int3 nNodeSize, int nMaxNode, int casetype)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nSlave = nNodeSize.x * nNodeSize.z;

	int nType = 0;

	int x = (nNodeID % nSlave) % nNodeSize.x;
	int y = (nNodeID / nSlave);
	int z = (nNodeID % nSlave) / nNodeSize.x;

	nType = NODE_FLUID;

	if (casetype == CAVITY)
	{
		if (y == 0)
		{
			nType = NODE_FIXEDWALL;
		}

		if (y == nNodeSize.y - 1)
		{
			nType = NODE_MOVINGWALL;
		}

		if ((x == 0 || x == nNodeSize.x - 1) || (z == 0 || z == nNodeSize.z - 1))
		{
			nType = NODE_FIXEDWALL;
		}
	}
	else if (casetype == POISEUILLE)
	{

		if (x == 0)
		{
			nType = NODE_INLET;
		}
		if ((x == nNodeSize.x - 1))
		{
			nType = NODE_OUTLET;
		}

		if ((y - 25) * (y - 25) + (z - 25) * (z - 25) > 400)
		{
			nType = NODE_STRUCT;
		}

	}
	NodeType[nNodeID] = nType;
}

__global__ void keComputeTurbulence_LES(int* nNodeType, float* CurrDen, float *CurrVelx, float *CurrVely, float *CurrVelz, float* dTauEddy, float *arF, float* arW, float* Direction, float dTau, float dSmagorinsky, int nMaxNode)
{
	int nNodeID = blockIdx.x*blockDim.x + threadIdx.x;
	if (nNodeID >= nMaxNode) return;

	int nodetype = nNodeType[nNodeID];
	if (nodetype == NODE_STRUCT) return;

	float3 dPhi0 = { 0.0, 0.0, 0.0 };
	float3 dPhi1 = { 0.0, 0.0, 0.0 };
	float3 dPhi2 = { 0.0, 0.0, 0.0 };

	float feq = 0.0;

	float den = CurrDen[nNodeID];
	float velx = CurrVelx[nNodeID];
	float vely = CurrVely[nNodeID];
	float velz = CurrVelz[nNodeID];
	float taueddy = dTauEddy[nNodeID];

	float f[DIRECTION_SIZE] = { 0.0, };

	for (int nDir = 0; nDir < DIRECTION_SIZE; nDir++)
	{
		f[nDir] = arF[nDir*nMaxNode + nNodeID];
	}

	float usquare = velx * velx + vely * vely + velz * velz;

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		float edotu = Direction[XDIM*DIRECTION_SIZE + nDir] * velx + Direction[YDIM*DIRECTION_SIZE + nDir] * vely + Direction[ZDIM*DIRECTION_SIZE + nDir] * velz;
		float feq = arW[nDir] * den * (1.0 + 3.0*edotu + 4.5*edotu * edotu - 1.5*usquare);

		dPhi0.x += Direction[XDIM*DIRECTION_SIZE + nDir] * Direction[XDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
		dPhi0.y += Direction[XDIM*DIRECTION_SIZE + nDir] * Direction[YDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
		dPhi0.z += Direction[XDIM*DIRECTION_SIZE + nDir] * Direction[ZDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);

		dPhi1.x += Direction[YDIM*DIRECTION_SIZE + nDir] * Direction[XDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
		dPhi1.y += Direction[YDIM*DIRECTION_SIZE + nDir] * Direction[YDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
		dPhi1.z += Direction[YDIM*DIRECTION_SIZE + nDir] * Direction[ZDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);

		dPhi2.x += Direction[ZDIM*DIRECTION_SIZE + nDir] * Direction[XDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
		dPhi2.y += Direction[ZDIM*DIRECTION_SIZE + nDir] * Direction[YDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
		dPhi2.z += Direction[ZDIM*DIRECTION_SIZE + nDir] * Direction[ZDIM*DIRECTION_SIZE + nDir] * (f[nDir] - feq);
	}

	float dQles = dot(dPhi0, dPhi0) + dot(dPhi1, dPhi1) + dot(dPhi2, dPhi2);
	float dSmag = 0.5 * (sqrt(pow(dTau, 2) + 18.0*1.41421356237*pow(dSmagorinsky, 2)*sqrt(dQles) / den));

	dTauEddy[nNodeID] = -0.5*dTau + dSmag;
}

