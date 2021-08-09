// 축 정보: [y][z][x]
// ComputeSets for initilaization 
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

using namespace std;
using namespace poplar;
using namespace poplar::program;


int slice_chunk(const unsigned nprocs, const int Slice[][DIMENSION][2], int *xs, int *xe, \
			    int *ys, int *ye, int *zs, int *ze, \
                int *chunk_nx, int *chunk_ny, int *chunk_nz, int *chunk_len);

Program initialize(Graph &graph, Inputval *pinput, Tensor Direction, Tensor DirectionState, Tensor NodeType, 	\
                   Tensor arF, Tensor arFTemp, Tensor TauEddy, Tensor CurrDen, Tensor PrevDen, 					\
				   Tensor CurrVelx, Tensor CurrVely, Tensor CurrVelz, Tensor arW, Tensor Move,	\
				   int N_PROCS, int Slice[][DIMENSION][2])
{
	int xs,xe,ys,ye,zs,ze;
	int ds,de;
	int chunk_nx,chunk_ny,chunk_nz,chunk_len;
	ds = 0;
	de = DIRECTION_SIZE;

	ComputeSet Initialize = graph.addComputeSet("Initialize");
	{
		for(unsigned nprocs=0; nprocs<N_PROCS; nprocs++)
		{
			if(slice_chunk(nprocs,Slice,&xs,&xe,&ys,&ye,&zs,&ze,&chunk_nx,&chunk_ny,&chunk_nz,&chunk_len)) continue;
			auto v = graph.addVertex(Initialize,
							  	"Initialize",
							  	{{"DirectionState",	DirectionState.slice({ys,zs,xs},{ye,ze,xe}).flatten()	},   
							  	 {"nNodeType",		NodeType.slice({ys,zs,xs},{ye,ze,xe}).flatten()			},
							  	 {"arF",			arF.slice({ys,zs,xs,ds},{ye,ze,xe,de}).flatten()		},
							  	 {"CurrDen",		CurrDen.slice({ys,zs,xs},{ye,ze,xe}).flatten()			},
							  	 {"CurrVelx",		CurrVelx.slice({ys,zs,xs},{ye,ze,xe}).flatten()			},
							  	 {"CurrVely",		CurrVely.slice({ys,zs,xs},{ye,ze,xe}).flatten()			},
							  	 {"CurrVelz",		CurrVelz.slice({ys,zs,xs},{ye,ze,xe}).flatten()			},
							  	 {"arW", 			arW														},
							  	 {"Direction", 		Direction               								},
							  	 {"RefDen", 		INPUT.m_dRefDen											},
                                 {"RefVelx",     	INPUT.m_dRefVelx                                        },
                                 {"RefVely",     	INPUT.m_dRefVely                                        },
                                 {"RefVelz",     	INPUT.m_dRefVelz                                        },
							   	 {"nx",				chunk_nx												},
							   	 {"ny",				chunk_ny												},
							   	 {"nz",				chunk_nz												},
							   	 {"length", 		chunk_len												},
							   	 {"NX",				INPUT.nx												},
							   	 {"NY",				INPUT.ny												},
							   	 {"NZ",				INPUT.nz												},
							   	 {"PADX",			xs														},
							   	 {"PADY",			ys														},
							   	 {"PADZ",			zs														}});

			unsigned tile = (nprocs*TILE_NUM)/N_PROCS;
			//printf("Nprocs=%d | TileNum=%d\n",nprocs,tile);
			graph.setTileMapping(v, 	tile);


    		graph.setTileMapping(DirectionState.slice({ys,zs,xs},{ye,ze,xe}),		tile);
			graph.setTileMapping(NodeType.slice({ys,zs,xs},{ye,ze,xe}), 			tile);
			graph.setTileMapping(arF.slice({ys,zs,xs,ds},{ye,ze,xe,de}),			tile);
			graph.setTileMapping(CurrDen.slice({ys,zs,xs},{ye,ze,xe}),				tile);
			graph.setTileMapping(CurrVelx.slice({ys,zs,xs},{ye,ze,xe}),				tile);
			graph.setTileMapping(CurrVely.slice({ys,zs,xs},{ye,ze,xe}),				tile);
			graph.setTileMapping(CurrVelz.slice({ys,zs,xs},{ye,ze,xe}),				tile);

			graph.setTileMapping(arFTemp.slice({ys,zs,xs,ds},{ye+2,ze+2,xe+2,de}),	tile);
			graph.setTileMapping(PrevDen.slice({ys,zs,xs},{ye,ze,xe}),				tile);
			graph.setTileMapping(TauEddy.slice({ys,zs,xs},{ye,ze,xe}),				tile);
			graph.setTileMapping(Move.slice({nprocs,ds},{nprocs+1,de}),				tile);
		}
	}
	return Sequence(Execute(Initialize));
}

