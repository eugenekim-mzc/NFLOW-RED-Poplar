#pragma once
#include <cuda_runtime.h>
#include "common.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <typeinfo>       // operator typeid

using namespace std;

void inverseMRT(std::vector<std::vector<float>> &matori, std::vector<std::vector<float>> &pInvMat);

void saveFile(Inputval *pinput, LBMSolver* psolver, std::vector<int> arIndex, int savetype);

void runSimulation(Inputval *pinput, LBMSolver* psolver);

__global__ void keInitialize(int* nNodeType, float* CurrDen, float *CurrVelx, float *CurrVely, float *CurrVelz, float *arF, float* arW, float* Direction, float RefDen, float RefVelx, float RefVely, float RefVelz, int nMaxNode);

__global__ void keInitialNodeDirection(int* DirectionState, int3 nNodeSize, float* Direction, int nMaxNode);

__global__ void keSetNodeType(int* NodeType, int3 nNodeSize, int nMaxNode, int casetype);

__global__ void keComputeTurbulence_LES(int* nNodeType, float* CurrDen, float *CurrVelx, float *CurrVely, float *CurrVelz, float* dTauEddy, float *arF, float* arW, float* Direction, float dTau, float dSmagorinsky, int nMaxNode);

__global__ void keCollision_MRT(int* nNodeType, float* CurrDen, float* CurrVelx, float* CurrVely, float* CurrVelz, float* dTauEddy, float* arF, float* arFTemp, float RefDen, float dTau, float dWeps, float dWepsj, float dWXX, MRTMatrix* Matrix, int nMaxNode);

__global__ void keCollision_SRT(int* nNodeType, float *CurrDen, float *CurrVelx, float *CurrVely, float *CurrVelz, float *arF, float *arFTemp, float* arW, float* Direction, float dTau, float* dTauEddy, int nMaxNode);

__global__ void keStreaming(int* nNodeType, float *arF, float *arFTemp, int* DirectionState, int* Move, int nMaxNode);

__global__ void keBoundaryCavity(int3 nNodeSize, int* nNodeType, float RefVelx, float RefVely, float RefVelz, float *arF, float *arFTemp, float* arW, float* Direction, int* DirectionState, int* nOppIndex, float RefDen, int nMaxNode);

__global__ void keBoundaryInlet(int3 nNodeSize, float *CurrDen, int* nNodeType, float RefVelx, float RefVely, float RefVelz, float *arF, float *arFTemp, float* arW, float* Direction, int* DirectionState, int* nOppIndex, int* Move, int nMaxNode);

__global__ void keBoundaryOutlet(int3 nNodeSize, float *CurrVelx, float *CurrVely, float *CurrVelz, int* nNodeType, float *arF, float *arFTemp, float* arW, float* Direction, int* DirectionState, int* nOppIndex, float RefDen, int* Move, int nMaxNode);

__global__ void keBoundaryStruct(int3 nNodeSize, int* nNodeType, float *arF, float *arFTemp, int* DirectionState, int* nOppIndex, int* Move, int nMaxNode);

__global__ void keMacroscopic(int* nNodeType, float *CurrDen, float *PrevDen, float *CurrVelx, float *CurrVely, float *CurrVelz, float *arF, float* Direction, int nMaxNode);
