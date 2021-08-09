// 축 정보: [y][z][x]
#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <cmath>
#include <poplar/Graph.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poputil/TileMapping.hpp>
#include "struct.h"

#define RED        "\x1b[31m"
#define RESET      "\x1b[0m"

#define SAVETYPE            CSVTYPE
#define CASETYPE            CAVITY

using namespace std;
using namespace poplar;
using namespace poplar::program;


LBMSolver::LBMSolver()
{
//    cudaSetDevice(0);
}

LBMSolver::~LBMSolver()
{
//    cudaSetDevice(0);
//    cudaDeviceReset();
}

Program initialize(Graph &graph, Inputval *pinput, Tensor Direction, Tensor DirectionState, Tensor NodeType, 	\
				   Tensor arF, Tensor arFTemp, Tensor TauEddy, Tensor CurrDen, Tensor PrevDen, 					\
				   Tensor CurrVelx, Tensor CurrVely, Tensor CurrVelz, Tensor arW, Tensor Move,					\ 
				   int N_PROCS, int Slice[][DIMENSION][2]);

Program runSimulation(Graph &graph, Inputval *pinput, Tensor arF, Tensor arFTemp, Tensor CurrDen, 				\
					  Tensor PrevDen, Tensor CurrVelx, Tensor CurrVely, Tensor CurrVelz, 						\
				      Tensor TauEddy, Tensor NodeType, Tensor DirectionState, 									\
                      Tensor Matrix, Tensor MatInv, Tensor Direction, Tensor arW, Tensor OppIndex, 				\
					  Tensor Move, int N_PROCS, int Slice[][DIMENSION][2]);

int decomposition(Inputval *pinput, int Slice[][DIMENSION][2]);
int set_move_decomp(const int N_PROCS, const int Slice[][DIMENSION][2], int Move[][DIRECTION_SIZE]);
void inverseMRT(std::vector<std::vector<float>> &matori, std::vector<std::vector<float>> &pInvMat);

int main(int argc, char** argv)
{
    //============================================================================================
    // Create and initialize variables on Host(CPU)
    std::cout << " - Create and initialize variables on Host(CPU)\n";

	Inputval *pinput = new Inputval();
	LBMSolver *psolver = new LBMSolver();

	INPUT.h = 100;
	INPUT.nx = INPUT.h;
	INPUT.ny = INPUT.h;
	INPUT.nz = INPUT.h;

	//INPUT.maxitr = 199;
	INPUT.maxitr = 5000;

	INPUT.bEnableMRT = true;
	INPUT.bEnableLES = true;

	INPUT.m_dRefDen  = 1.0f;
	INPUT.m_dRefVelx = 0.1f;
	INPUT.m_dRefVely = 0.0f;
	INPUT.m_dRefVelz = 0.0f;

	INPUT.Re = 1000;
	INPUT.length = INPUT.nx * INPUT.ny * INPUT.nz;
	INPUT.nu = INPUT.m_dRefVelx * INPUT.h / INPUT.Re;

	INPUT.nwrite = 10000;
	INPUT.savetype = SAVETYPE;
	INPUT.casetype = CASETYPE;
	INPUT.m_dTau = 3.0f*INPUT.m_dRefVelx*INPUT.h / INPUT.Re + 0.50f;
	
	INPUT.dWeps = 3.0f;
	INPUT.dWepsj = -5.5f;
	INPUT.dWXX = -0.5f;

	SOLVER.m_HostDirection[XDIM][0] = 0.0f;		SOLVER.m_HostDirection[YDIM][0] = 0.0f;		SOLVER.m_HostDirection[ZDIM][0] = 0.0f;
	SOLVER.m_HostDirection[XDIM][1] = 1.0f;		SOLVER.m_HostDirection[YDIM][1] = 0.0f;		SOLVER.m_HostDirection[ZDIM][1] = 0.0f;
	SOLVER.m_HostDirection[XDIM][2] = -1.0f;	SOLVER.m_HostDirection[YDIM][2] = 0.0f;		SOLVER.m_HostDirection[ZDIM][2] = 0.0f;
	SOLVER.m_HostDirection[XDIM][3] = 0.0f;		SOLVER.m_HostDirection[YDIM][3] = 0.0f;		SOLVER.m_HostDirection[ZDIM][3] = -1.0f;
	SOLVER.m_HostDirection[XDIM][4] = 0.0f;		SOLVER.m_HostDirection[YDIM][4] = 0.0f;		SOLVER.m_HostDirection[ZDIM][4] = 1.0f;
	SOLVER.m_HostDirection[XDIM][5] = 0.0f;		SOLVER.m_HostDirection[YDIM][5] = 1.0f;		SOLVER.m_HostDirection[ZDIM][5] = 0.0f;
	SOLVER.m_HostDirection[XDIM][6] = 0.0f;		SOLVER.m_HostDirection[YDIM][6] = -1.0f;	SOLVER.m_HostDirection[ZDIM][6] = 0.0f;
	SOLVER.m_HostDirection[XDIM][7] = 1.0f;		SOLVER.m_HostDirection[YDIM][7] = 0.0f;		SOLVER.m_HostDirection[ZDIM][7] = -1.0f;
	SOLVER.m_HostDirection[XDIM][8] = -1.0f;	SOLVER.m_HostDirection[YDIM][8] = 0.0f;		SOLVER.m_HostDirection[ZDIM][8] = -1.0f;
	SOLVER.m_HostDirection[XDIM][9] = 1.0f;		SOLVER.m_HostDirection[YDIM][9] = 0.0f;		SOLVER.m_HostDirection[ZDIM][9] = 1.0f;
	SOLVER.m_HostDirection[XDIM][10] = -1.0f;	SOLVER.m_HostDirection[YDIM][10] = 0.0f;	SOLVER.m_HostDirection[ZDIM][10] = 1.0f;
	SOLVER.m_HostDirection[XDIM][11] = 1.0f;	SOLVER.m_HostDirection[YDIM][11] = 1.0f;	SOLVER.m_HostDirection[ZDIM][11] = 0.0f;
	SOLVER.m_HostDirection[XDIM][12] = -1.0f;	SOLVER.m_HostDirection[YDIM][12] = 1.0f;	SOLVER.m_HostDirection[ZDIM][12] = 0.0f;
	SOLVER.m_HostDirection[XDIM][13] = 1.0f;	SOLVER.m_HostDirection[YDIM][13] = -1.0f;	SOLVER.m_HostDirection[ZDIM][13] = 0.0f;
	SOLVER.m_HostDirection[XDIM][14] = -1.0f;	SOLVER.m_HostDirection[YDIM][14] = -1.0f;	SOLVER.m_HostDirection[ZDIM][14] = 0.0f;
	SOLVER.m_HostDirection[XDIM][15] = 0.0f;	SOLVER.m_HostDirection[YDIM][15] = 1.0f;	SOLVER.m_HostDirection[ZDIM][15] = -1.0f;
	SOLVER.m_HostDirection[XDIM][16] = 0.0f;	SOLVER.m_HostDirection[YDIM][16] = 1.0f;	SOLVER.m_HostDirection[ZDIM][16] = 1.0f;
	SOLVER.m_HostDirection[XDIM][17] = 0.0f;	SOLVER.m_HostDirection[YDIM][17] = -1.0f;	SOLVER.m_HostDirection[ZDIM][17] = -1.0f;
	SOLVER.m_HostDirection[XDIM][18] = 0.0f;	SOLVER.m_HostDirection[YDIM][18] = -1.0f;	SOLVER.m_HostDirection[ZDIM][18] = 1.0f;

	SOLVER.m_HostOppIndex[0] = 0;			SOLVER.m_HostOppIndex[1] = 2;			SOLVER.m_HostOppIndex[2] = 1;
	SOLVER.m_HostOppIndex[3] = 4;			SOLVER.m_HostOppIndex[4] = 3;			SOLVER.m_HostOppIndex[5] = 6;
	SOLVER.m_HostOppIndex[6] = 5;			SOLVER.m_HostOppIndex[7] = 10;			SOLVER.m_HostOppIndex[8] = 9;
	SOLVER.m_HostOppIndex[9] = 8;			SOLVER.m_HostOppIndex[10] = 7;			SOLVER.m_HostOppIndex[11] = 14;
	SOLVER.m_HostOppIndex[12] = 13;		    SOLVER.m_HostOppIndex[13] = 12;		    SOLVER.m_HostOppIndex[14] = 11;
	SOLVER.m_HostOppIndex[15] = 18;		    SOLVER.m_HostOppIndex[16] = 17;		    SOLVER.m_HostOppIndex[17] = 16;
	SOLVER.m_HostOppIndex[18] = 15;

	int nSlave = INPUT.nx * INPUT.nz;

	/*
	// 이동 방향 설정
    SOLVER.m_HostMove[0] = 0;                   //ORG
    SOLVER.m_HostMove[1] = 1;                   //+x
    SOLVER.m_HostMove[2] = -1;                  //-x
    SOLVER.m_HostMove[3] = -INPUT.nx;           //-z
    SOLVER.m_HostMove[4] = INPUT.nx;            //+z
    SOLVER.m_HostMove[5] = nSlave;              //+y
    SOLVER.m_HostMove[6] = -nSlave;             //-y
    SOLVER.m_HostMove[7] = -INPUT.nx + 1;       //+x -z
    SOLVER.m_HostMove[8] = -INPUT.nx - 1;       //-x -z
    SOLVER.m_HostMove[9] = INPUT.nx + 1;        //+x +z
    SOLVER.m_HostMove[10] = INPUT.nx - 1;       //-x +z
    SOLVER.m_HostMove[11] = nSlave + 1;         //+x +y
    SOLVER.m_HostMove[12] = nSlave - 1;         //-x +y
    SOLVER.m_HostMove[13] = -nSlave + 1;        //+x -y
    SOLVER.m_HostMove[14] = -nSlave - 1;        //-x -y
    SOLVER.m_HostMove[15] = nSlave - INPUT.nx;  //+y -z
    SOLVER.m_HostMove[16] = nSlave + INPUT.nx;  //+y +z
    SOLVER.m_HostMove[17] = -nSlave - INPUT.nx; //-y -z
    SOLVER.m_HostMove[18] = -nSlave + INPUT.nx; //-y +z
	*/
	

	for (int nDir = 0; nDir < DIRECTION_SIZE; ++nDir)
	{
		if (nDir == 0) 					    	SOLVER.m_HostW[nDir] = 1.0f / 3.0f;
		else if (nDir > 0 && nDir < 7)			SOLVER.m_HostW[nDir] = 1.0f / 18.0f;
		else							        SOLVER.m_HostW[nDir] = 1.0f / 36.0f;
	}

    vector<float> h_arF(INPUT.length*DIRECTION_SIZE, 0);
	vector<vector<float>> mat(DIRECTION_SIZE, vector<float>(DIRECTION_SIZE));
    mat=
    {
        {   1.0,  1.0,  1.0,  1.0,  1.0,      1.0,  1.0,  1.0,  1.0,  1.0,     1.0,  1.0,  1.0,  1.0,  1.0,     1.0,  1.0,  1.0,  1.0 }, //  0
        { -30.0,-11.0,-11.0,-11.0,-11.0,    -11.0,-11.0,  8.0,  8.0,  8.0,     8.0,  8.0,  8.0,  8.0,  8.0,     8.0,  8.0,  8.0,  8.0 }, //  1
        {  12.0, -4.0, -4.0, -4.0, -4.0,     -4.0, -4.0,  1.0,  1.0,  1.0,     1.0,  1.0,  1.0,  1.0,  1.0,     1.0,  1.0,  1.0,  1.0 }, //  2
        {   0.0,  1.0, -1.0,  0.0,  0.0,      0.0,  0.0,  1.0, -1.0,  1.0,    -1.0,  1.0, -1.0,  1.0, -1.0,     0.0,  0.0,  0.0,  0.0 }, //  3
        {   0.0, -4.0,  4.0,  0.0,  0.0,      0.0,  0.0,  1.0, -1.0,  1.0,    -1.0,  1.0, -1.0,  1.0, -1.0,     0.0,  0.0,  0.0,  0.0 }, //  4
        {   0.0,  0.0,  0.0,  0.0,  0.0,      1.0, -1.0,  0.0,  0.0,  0.0,     0.0,  1.0,  1.0, -1.0, -1.0,     1.0,  1.0, -1.0, -1.0 }, //  5
        {   0.0,  0.0,  0.0,  0.0,  0.0,     -4.0,  4.0,  0.0,  0.0,  0.0,     0.0,  1.0,  1.0, -1.0, -1.0,     1.0,  1.0, -1.0, -1.0 }, //  6
        {   0.0,  0.0,  0.0, -1.0,  1.0,      0.0,  0.0, -1.0, -1.0,  1.0,     1.0,  0.0,  0.0,  0.0,  0.0,    -1.0,  1.0, -1.0,  1.0 }, //  7
        {   0.0,  0.0,  0.0,  4.0, -4.0,      0.0,  0.0, -1.0, -1.0,  1.0,     1.0,  0.0,  0.0,  0.0,  0.0,    -1.0,  1.0, -1.0,  1.0 }, //  8
        {   0.0,  2.0,  2.0, -1.0, -1.0,     -1.0, -1.0,  1.0,  1.0,  1.0,     1.0,  1.0,  1.0,  1.0,  1.0,    -2.0, -2.0, -2.0, -2.0 }, //  9
        {   0.0, -4.0, -4.0,  2.0,  2.0,      2.0,  2.0,  1.0,  1.0,  1.0,     1.0,  1.0,  1.0,  1.0,  1.0,    -2.0, -2.0, -2.0, -2.0 }, // 10
        {   0.0,  0.0,  0.0, -1.0, -1.0,      1.0,  1.0, -1.0, -1.0, -1.0,    -1.0,  1.0,  1.0,  1.0,  1.0,     0.0,  0.0,  0.0,  0.0 }, // 11
        {   0.0,  0.0,  0.0,  2.0,  2.0,     -2.0, -2.0, -1.0, -1.0, -1.0,    -1.0,  1.0,  1.0,  1.0,  1.0,     0.0,  0.0,  0.0,  0.0 }, // 12
        {   0.0,  0.0,  0.0,  0.0,  0.0,      0.0,  0.0,  0.0,  0.0,  0.0,     0.0,  1.0, -1.0, -1.0,  1.0,     0.0,  0.0,  0.0,  0.0 }, // 13
        {   0.0,  0.0,  0.0,  0.0,  0.0,      0.0,  0.0,  0.0,  0.0,  0.0,     0.0,  0.0,  0.0,  0.0,  0.0,    -1.0,  1.0,  1.0, -1.0 }, // 14
        {   0.0,  0.0,  0.0,  0.0,  0.0,      0.0,  0.0, -1.0,  1.0,  1.0,    -1.0,  0.0,  0.0,  0.0,  0.0,     0.0,  0.0,  0.0,  0.0 }, // 15
        {   0.0,  0.0,  0.0,  0.0,  0.0,      0.0,  0.0, -1.0,  1.0, -1.0,     1.0,  1.0, -1.0,  1.0, -1.0,     0.0,  0.0,  0.0,  0.0 }, // 16
        {   0.0,  0.0,  0.0,  0.0,  0.0,      0.0,  0.0,  0.0,  0.0,  0.0,     0.0, -1.0, -1.0,  1.0,  1.0,     1.0,  1.0, -1.0, -1.0 }, // 17
        {   0.0,  0.0,  0.0,  0.0,  0.0,      0.0,  0.0, -1.0, -1.0,  1.0,     1.0,  0.0,  0.0,  0.0,  0.0,     1.0, -1.0,  1.0, -1.0 }  // 18
    };

	vector<vector<float>> matInv(DIRECTION_SIZE, vector<float>(DIRECTION_SIZE,0));
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
	

    //============================================================================================
    DeviceManager manager = DeviceManager::createDeviceManager();
    Device device;
    bool success = false;

    for (auto &hwDevice : manager.getDevices(TargetType::IPU, 1))
    {
        device = std::move(hwDevice);
        std::cerr << " - Trying to attatch to IPU" << device.getId() << std::endl;

        if ((success = device.attach()))
        std::cerr << " - Attached to IPU" << device.getId() << std::endl;
    }

    if (!success)
    {
        std::cerr << " - Error attaching to device" << std::endl;
        return -1;
    }

    Target target = device.getTarget();
    Graph graph(target);

    // Add codelets to the graph
    graph.addCodelets("lbm_codelets.cpp");


    //============================================================================================
    // Add Variables to the graph
    std::cout<< " - Add Variables to the graph\n";
    Tensor arF     		= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx),DIRECTION_SIZE}, "arF");
    //Tensor arFTemp 	= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx),DIRECTION_SIZE}, "arFTemp");
    Tensor arFTemp     	= graph.addVariable(FLOAT, {unsigned(INPUT.ny+2),unsigned(INPUT.nz+2),unsigned(INPUT.nx+2),DIRECTION_SIZE}, "arFTemp");

    Tensor CurrDen 		= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "CurrDen");
    Tensor PrevDen 		= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "PrevDen");

    Tensor CurrVelx		= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "CurrVelx");
    Tensor CurrVely		= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "CurrVely");
    Tensor CurrVelz		= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "CurrVelz");

    //Tensor PrevVelx 	= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "PrevVelx");
    //Tensor PrevVely 	= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "PrevVely");
    //Tensor PrevVelz 	= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "PrevVelz");

    Tensor TauEddy		= graph.addVariable(FLOAT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "TauEddy");
    Tensor NodeType		= graph.addVariable(INT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "NodeType");
    Tensor DirectionState = graph.addVariable(INT, {unsigned(INPUT.ny),unsigned(INPUT.nz),unsigned(INPUT.nx)}, "DirectionState");

    Tensor Matrix 		= graph.addVariable(FLOAT, {DIRECTION_SIZE,DIRECTION_SIZE}, "Matrix");
    Tensor MatInv 		= graph.addVariable(FLOAT, {DIRECTION_SIZE,DIRECTION_SIZE}, "MatInv");
    Tensor Direction 	= graph.addVariable(FLOAT, {DIMENSION*DIRECTION_SIZE}, "Direction");
    Tensor arW        	= graph.addVariable(FLOAT, {DIRECTION_SIZE}, "arW");
    Tensor OppIndex 	= graph.addVariable(INT, {DIRECTION_SIZE}, "OppIndex");

	// Mapping on last tile 
	graph.setTileMapping(Matrix, 	TILE_NUM);
	graph.setTileMapping(MatInv, 	TILE_NUM);
	graph.setTileMapping(Direction,	TILE_NUM);
	graph.setTileMapping(arW, 		TILE_NUM);
	graph.setTileMapping(OppIndex, 	TILE_NUM);

	/*
	// Mapping on 1st tile 
	graph.setTileMapping(Matrix, 	0);
	graph.setTileMapping(MatInv, 	0);
	graph.setTileMapping(Direction,	0);
	graph.setTileMapping(arW, 		0);
	graph.setTileMapping(OppIndex, 	0);
	*/
    //============================================================================================

    //============================================================================================
    // Setting for decomposition 
	int N_PROCS = XDECOMP*YDECOMP*ZDECOMP;
	int Slice[N_PROCS][DIMENSION][2] = {0};
	decomposition(pinput, Slice);
	printf("Number of Processes(Threads on Tiles) participating = %d\n",N_PROCS);
	int Move_decomp[N_PROCS][DIRECTION_SIZE];
    Tensor Move 		= graph.addVariable(INT, {N_PROCS,DIRECTION_SIZE}, "Move");
	set_move_decomp(N_PROCS, Slice, Move_decomp);
	

    //============================================================================================
    // Create Host write handles for variables on the Device
    std::cout << " - Create Host write(read) handles for variables on the Device\n";
    graph.createHostWrite("H2D-Matrix",		Matrix);
    graph.createHostWrite("H2D-MatInv",		MatInv);
    graph.createHostWrite("H2D-Direction", 	Direction);
    graph.createHostWrite("H2D-arW", 		arW);
    graph.createHostWrite("H2D-OppIndex", 	OppIndex);
    graph.createHostWrite("H2D-Move", 		Move);

    graph.createHostWrite("H2D-arF", 		arF);
    graph.createHostRead("D2H-arF", 		arF);
    //graph.createHostRead("D2H-arFTemp",		arFTemp);

    Sequence initProg, mainProg, iterProg;

    Program initial_step = initialize(graph, pinput, Direction, DirectionState, NodeType, arF, arFTemp, TauEddy,		\
				 				 	  CurrDen, PrevDen, CurrVelx, CurrVely, CurrVelz, arW, Move, N_PROCS, Slice);

    Program single_step = runSimulation(graph, pinput, arF, arFTemp, CurrDen, PrevDen, CurrVelx, CurrVely, CurrVelz,	\
										TauEddy, NodeType, DirectionState, Matrix, MatInv, Direction, arW, OppIndex, 	\
										Move, N_PROCS, Slice);

    initProg.add(initial_step);
	iterProg.add(initProg);

    mainProg.add(single_step);
    iterProg.add(Repeat(INPUT.maxitr, mainProg));

    Engine engine(graph, iterProg);
    engine.load(device);

    engine.writeTensor("H2D-Matrix",	temp.m_MatMRT);
    engine.writeTensor("H2D-MatInv",	temp.m_MatInv);
    engine.writeTensor("H2D-Direction", SOLVER.m_HostDirection);
    engine.writeTensor("H2D-arW", 		SOLVER.m_HostW);
    engine.writeTensor("H2D-OppIndex",	SOLVER.m_HostOppIndex); 
    engine.writeTensor("H2D-Move", 		Move_decomp);

    engine.writeTensor("H2D-arF",   	h_arF.data());

    std::cout << " - Copy Host data via the write handle to variables on the Device\n";
    //============================================================================================

    auto start_t = std::chrono::high_resolution_clock::now();
    std::cout << " - Running program\n";
    engine.run(0);
    std::cout << " - Program complete\n";
    auto end_t = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_t - start_t;
    printf(" - Elapsed time(IPU) = %fs | h=%d\n",elapsed.count(), INPUT.h);

    std::cout << " - Copy Device data via read handle to variables on the Host\n";
    engine.readTensor("D2H-arF", h_arF.data());
    //engine.readTensor("D2H-arFTemp", h_arF.data());

    //============================================================================================
    printf("---------------------------------\n");
    for(int nDir=9; nDir>8; nDir--)
    {
        for(int j=INPUT.ny-1; j>=INPUT.ny-5; j--)
        {
            for(int k=5; k<6; k++)
            {
            	for(int i=INPUT.nx-5; i<INPUT.nx; i++)
            	{
					//Different from Original E8ight CUDA version indexing!!!! 
                	int id = nDir + DIRECTION_SIZE*(i + k*INPUT.nx + j*INPUT.nx*INPUT.nz);
                	printf("%8.6f ", h_arF[id]);
				}
            	printf("\n");
            }
        }
		printf("\n");	
    }
    printf("---------------------------------\n");



	/*
    printf("---------------------------------\n");
    //for(int nDir=0; nDir<DIRECTION_SIZE; nDir++)

    for(int nDir=9; nDir>8; nDir--)
    {
        for(int j=INPUT.ny-1; j>=0; j--)
        {
            //for(int k=1; k<INPUT.nz+2; k++)
            //for(int k=4; k<5; k++)
            for(int k=2; k<3; k++)
            {
            	for(int i=0; i<INPUT.nx; i++)
            	{
					//Different from Original E8ight CUDA version indexing!!!! 
                	int id = nDir + DIRECTION_SIZE*(i + k*INPUT.nx + j*INPUT.nx*INPUT.nz);

                	//if(h_arF[id]>0) printf(RED "%1d%1d%1d%2d " RESET, i,j,k,nDir);
                	//else            printf("%1d%1d%1d%2d ", i,j,k,nDir);

                	//if(h_arF[id]>0.0) printf(RED "%8.6f " RESET, h_arF[id]);
                	//else            printf("%8.6f ", h_arF[id]);
                	printf("%8.6f ", h_arF[id]);
				}
            	printf("\n");
            }
        }
		printf("\n");	
    }
    printf("---------------------------------\n");
	*/
	delete pinput;
	delete psolver;
}

