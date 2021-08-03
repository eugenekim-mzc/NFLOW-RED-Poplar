#pragma once

//#include <stdio.h>
//#include <vector>
//#include <sstream>
#include "common.h"

typedef struct HostNodeData
{
	std::vector<float>     m_arF;
	std::vector<float>     m_arFTemp;

	std::vector<float>     m_dCurrDen;
	std::vector<float>     m_dPrevDen;

	std::vector<float>     m_dCurrVelx;
	std::vector<float>     m_dCurrVely;
	std::vector<float>     m_dCurrVelz;

	std::vector<float>     m_dPrevVelx;
	std::vector<float>     m_dPrevVely;
	std::vector<float>     m_dPrevVelz;

	std::vector<int>        m_nNodeType;

	std::vector<float>		m_dTauEddy;
};


typedef struct DeviceNodeData
{
	float*     m_arF;
	float*     m_arFTemp;

	float*     m_dCurrDen;
	float*     m_dPrevDen;

	float*     m_dCurrVelx;
	float*     m_dCurrVely;
	float*     m_dCurrVelz;

	float*     m_dPrevVelx;
	float*     m_dPrevVely;
	float*     m_dPrevVelz;

	float*		m_dTauEddy;

	int*        m_nNodeType;
	int*        m_nDirectionState;
};

struct MRTMatrix
{
	float                                              m_MatMRT[DIRECTION_SIZE][DIRECTION_SIZE];
	float                                              m_MatInv[DIRECTION_SIZE][DIRECTION_SIZE];
};

class LBMSolver
{
public:
	LBMSolver();
	virtual ~LBMSolver();

	float			                                    m_HostDirection[DIMENSION][DIRECTION_SIZE];
	int				                                    m_HostOppIndex[DIRECTION_SIZE];
	float			                                    m_HostW[DIRECTION_SIZE];
	int                                                 m_HostMove[DIRECTION_SIZE];
	HostNodeData                                        m_HostNode;

	float*			                                    m_DeviceDirection;
	int*				                                m_DeviceOppIndex;
	float*			                                    m_DeviceW;
	int*                                                m_DeviceMove;
	MRTMatrix*                                          m_DeviceMatrix;
	DeviceNodeData                                      m_DeviceNode;
};

typedef struct Inputval
{
	int h;
	int	nx, ny, nz;
	int ngrid;
	int maxitr;
	int nwrite, irestart, irst;
	int icycle;
	float m_dSmagorinsky;
	float Re, length, nu;
	float m_dRefViscosity;
	float m_dRefDen;
	float m_dTau;
	float m_dRefVelx, m_dRefVely, m_dRefVelz;

	float dWeps, dWepsj, dWXX;

	bool bEnableMRT;
	bool bEnableLES;

	int savetype;
	int casetype;
};
