// 축 정보: [y][z][x]
// runSimulation is handler for ComputeSets 
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


int slice_chunk(const unsigned tile, const int Slice[][DIMENSION][2], int *xs, int *xe, int *ys, int *ye, \
                int *zs, int *ze, int *chunk_nx, int *chunk_ny, int *chunk_nz, int *chunk_len);

Program runSimulation(Graph &graph, Inputval *pinput, Tensor arF, Tensor arFTemp, Tensor CurrDen,  			\
					  Tensor PrevDen, Tensor CurrVelx, Tensor CurrVely, Tensor CurrVelz, Tensor TauEddy, 	\
				      Tensor NodeType, Tensor DirectionState, Tensor Matrix, Tensor MatInv, 				\
					  Tensor Direction, Tensor arW, Tensor OppIndex, Tensor Move,							\ 
					  int NTILE, int Slice[][DIMENSION][2]) 
{
	int xs,xe,ys,ye,zs,ze;
	int ds,de;
	int chunk_nx,chunk_ny,chunk_nz,chunk_len;
	ds = 0;
	de = DIRECTION_SIZE;

    ComputeSet ComputeTurbulence_LES = graph.addComputeSet("ComputeTurbulence_LES");
    {
 		for(unsigned tile=0; tile<NTILE; tile++)
		{
			if(slice_chunk(tile,Slice,&xs,&xe,&ys,&ye,&zs,&ze,&chunk_nx,&chunk_ny,&chunk_nz,&chunk_len)) continue;
       		auto v = graph.addVertex(ComputeTurbulence_LES,
                                "ComputeTurbulence_LES",
                                {{"nNodeType",		NodeType.slice({ys,zs,xs},{ye,ze,xe}).flatten()         },
                                 {"arF",            arF.slice({ys,zs,xs,ds},{ye,ze,xe,de}).flatten()        },
                                 {"CurrDen",        CurrDen.slice({ys,zs,xs},{ye,ze,xe}).flatten()          },
                                 {"CurrVelx",       CurrVelx.slice({ys,zs,xs},{ye,ze,xe}).flatten()         },
                                 {"CurrVely",       CurrVely.slice({ys,zs,xs},{ye,ze,xe}).flatten()         },
                                 {"CurrVelz",       CurrVelz.slice({ys,zs,xs},{ye,ze,xe}).flatten()         },
                                 {"dTauEddy",      	TauEddy.slice({ys,zs,xs},{ye,ze,xe}).flatten()     		},
                                 {"arW",            arW                                          			},
                                 {"Direction",      Direction                                               },
                                 {"dTau",         	INPUT.m_dTau                                            },
                                 {"dSmagorinsky",  	INPUT.m_dSmagorinsky                         			},
                                 {"nx",     		chunk_nx                               					},
                                 {"ny",     		chunk_ny                                            	},
                                 {"nz",     		chunk_nz                                            	},
                                 {"length", 		chunk_len                                           	}});
			graph.setTileMapping(v, 	tile);
		}
	}

    ComputeSet Collision_MRT = graph.addComputeSet("Collision_MRT");
    {
 		for(unsigned tile=0; tile<NTILE; tile++)
		{
			if(slice_chunk(tile,Slice,&xs,&xe,&ys,&ye,&zs,&ze,&chunk_nx,&chunk_ny,&chunk_nz,&chunk_len)) continue;
        	auto v = graph.addVertex(Collision_MRT,
                                "Collision_MRT",
                                {{"nNodeType",   	NodeType.slice({ys,zs,xs},{ye,ze,xe}).flatten()         			},
                                 {"arF",            arF.slice({ys,zs,xs,ds},{ye,ze,xe,de}).flatten()        			},
								    /**** Only Inside points(No ghostcell) ****/
                                 {"arFTemp",        arFTemp.slice({ys+1,zs+1,xs+1,ds},{ye+1,ze+1,xe+1,de}).flatten() 	},	
                                 {"CurrDen",        CurrDen.slice({ys,zs,xs},{ye,ze,xe}).flatten()          			},
                                 {"CurrVelx",       CurrVelx.slice({ys,zs,xs},{ye,ze,xe}).flatten()         			},
                                 {"CurrVely",       CurrVely.slice({ys,zs,xs},{ye,ze,xe}).flatten()         			},
                                 {"CurrVelz",       CurrVelz.slice({ys,zs,xs},{ye,ze,xe}).flatten()         			},
                                 {"dTauEddy",      	TauEddy.slice({ys,zs,xs},{ye,ze,xe}).flatten()     					},
                                 {"RefDen",       	INPUT.m_dRefDen														},
                                 {"dTau",         	INPUT.m_dTau														},
                                 {"dWeps",       	INPUT.dWeps															},
                                 {"dWepsj",       	INPUT.dWepsj														},
                                 {"dWXX",  			INPUT.dWXX															},
                                 {"m_MatMRT",  		Matrix.flatten()                               						},
                                 {"m_MatInv",  		MatInv.flatten()                               						},
                                 {"nx",     		chunk_nx                               								},
                                 {"ny",     		chunk_ny                                            				},
                                 {"nz",     		chunk_nz                                            				},
                                 {"length", 		chunk_len                                           				}});
			graph.setTileMapping(v, 	tile);
		}
	}

    ComputeSet Streaming = graph.addComputeSet("Streaming");
    {
 		for(unsigned tile=0; tile<NTILE; tile++)
		{
			if(slice_chunk(tile,Slice,&xs,&xe,&ys,&ye,&zs,&ze,&chunk_nx,&chunk_ny,&chunk_nz,&chunk_len)) continue;
        	auto v = graph.addVertex(Streaming,
                                "Streaming",
                                {{"nNodeType",  	NodeType.slice({ys,zs,xs},{ye,ze,xe}).flatten()         	},
                                 {"arF",            arF.slice({ys,zs,xs,ds},{ye,ze,xe,de}).flatten()        	},
								    /*********  Including Ghostcells *********/
                                 {"arFTemp",        arFTemp.slice({ys,zs,xs,ds},{ye+2,ze+2,xe+2,de}).flatten()	},
							  	 {"DirectionState",	DirectionState.slice({ys,zs,xs},{ye,ze,xe}).flatten()		},
                                 {"Move",        	Move.slice({tile,ds},{tile+1,de}).flatten() 				},
                                 {"nOppIndex",      OppIndex                               						},
                                 {"nx",     		chunk_nx                               						},
                                 {"ny",     		chunk_ny                                            		},
                                 {"nz",     		chunk_nz                                            		},
                                 {"length", 		chunk_len                                           		}});
			graph.setTileMapping(v, 	tile);
		}
	}

    ComputeSet BoundaryCavity = graph.addComputeSet("BoundaryCavity");
    {
 		for(unsigned tile=0; tile<NTILE; tile++)
		{
			if(slice_chunk(tile,Slice,&xs,&xe,&ys,&ye,&zs,&ze,&chunk_nx,&chunk_ny,&chunk_nz,&chunk_len)) continue;
        	//printf("TILE=%4d | %02d-%02d | %02d-%02d | %02d-%02d | %2d %2d %2d\n",tile, xs,xe,ys,ye,zs,ze,chunk_nx,chunk_ny,chunk_nz);
        	auto v = graph.addVertex(BoundaryCavity,
                                "BoundaryCavity",
                                {{"nNodeType",  	NodeType.slice({ys,zs,xs},{ye,ze,xe}).flatten()         			},
                                 {"arF",            arF.slice({ys,zs,xs,ds},{ye,ze,xe,de}).flatten()        			},
								    /**** Only Inside points(No ghostcell) ****/
                                 {"arFTemp",        arFTemp.slice({ys+1,zs+1,xs+1,ds},{ye+1,ze+1,xe+1,de}).flatten() 	},	
                                 {"arW",            arW                                          						},
                                 {"Direction",      Direction                                               			},
							  	 {"DirectionState",	DirectionState.slice({ys,zs,xs},{ye,ze,xe}).flatten()				},
                                 {"nOppIndex",      OppIndex                                       						},
                                 {"RefDen",       	INPUT.m_dRefDen														},
                                 {"velx",       	INPUT.m_dRefVelx													},
                                 {"vely",       	INPUT.m_dRefVely													},
                                 {"velz",       	INPUT.m_dRefVelz													},
                                 {"nx",     		chunk_nx                               								},
                                 {"ny",     		chunk_ny                                            				},
                                 {"nz",     		chunk_nz                                            				},
                                 {"length", 		chunk_len                                           				}});
			graph.setTileMapping(v, 	tile);
		}
	}

    ComputeSet Macroscopic = graph.addComputeSet("Macroscopic");
    {
 		for(unsigned tile=0; tile<NTILE; tile++)
		{
			if(slice_chunk(tile,Slice,&xs,&xe,&ys,&ye,&zs,&ze,&chunk_nx,&chunk_ny,&chunk_nz,&chunk_len)) continue;
        	auto v = graph.addVertex(Macroscopic,
                                "Macroscopic",
                                {{"nNodeType",  	NodeType.slice({ys,zs,xs},{ye,ze,xe}).flatten()         },
                                 {"CurrDen",        CurrDen.slice({ys,zs,xs},{ye,ze,xe}).flatten()          },
                                 {"PrevDen",        PrevDen.slice({ys,zs,xs},{ye,ze,xe}).flatten()          },
                                 {"CurrVelx",       CurrVelx.slice({ys,zs,xs},{ye,ze,xe}).flatten()         },
                                 {"CurrVely",       CurrVely.slice({ys,zs,xs},{ye,ze,xe}).flatten()         },
                                 {"CurrVelz",       CurrVelz.slice({ys,zs,xs},{ye,ze,xe}).flatten()         },
                                 {"arF",            arF.slice({ys,zs,xs,ds},{ye,ze,xe,de}).flatten()        },
                                 {"Direction",      Direction                                               },
                                 {"nx",     		chunk_nx                               					},
                                 {"ny",     		chunk_ny                                            	},
                                 {"nz",     		chunk_nz                                            	},
                                 {"length", 		chunk_len                                           	}});
			graph.setTileMapping(v, 	tile);
		}
	}

	return Sequence(Execute(ComputeTurbulence_LES), Execute(Collision_MRT), Execute(Streaming),	\
				    Execute(BoundaryCavity), Execute(Macroscopic));

	//return Sequence(Execute(ComputeTurbulence_LES), Execute(Collision_MRT), Execute(BoundaryCavity), Execute(Macroscopic));
}

