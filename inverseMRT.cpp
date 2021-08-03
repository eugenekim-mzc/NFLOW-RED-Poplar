#include <iostream>
#include <vector>
#include <cmath>
#include "struct.h"

using namespace std;

void inverseMRT(std::vector<std::vector<float>> &matori, std::vector<std::vector<float>> &pInvMat)
{
	std::vector<std::vector<float>> matInv;
	matInv = matori;

	std::vector<float> tmpa(DIRECTION_SIZE, 0.0);
	std::vector<float> tmpa2(DIRECTION_SIZE, 0.0);

	for (int i = 0; i <DIRECTION_SIZE; ++i)
		pInvMat[i][i] = 1.0;

	for (int i = 0; i <DIRECTION_SIZE - 1; ++i)
	{
		while (abs(matInv[i][i]) < pow(10, -9))
		{
			tmpa = matInv[i];
			tmpa2 = pInvMat[i];

			for (int j = i; j <DIRECTION_SIZE - 1; ++j)
			{
				matInv[j] = matInv[j + 1];
				pInvMat[j] = pInvMat[j + 1];
			}

			matInv[DIRECTION_SIZE - 1] = tmpa;
			pInvMat[DIRECTION_SIZE - 1] = tmpa2;
		}

		float inv_ii = matInv[i][i];
		for (int j = 0; j <DIRECTION_SIZE; ++j)
		{
			pInvMat[i][j] = pInvMat[i][j] / inv_ii;
			matInv[i][j] = matInv[i][j] / inv_ii;
		}

		tmpa = matInv[i];
		tmpa2 = pInvMat[i];

		for (int j = i + 1; j <DIRECTION_SIZE; ++j)
		{
			float inv_ji = matInv[j][i];
			for (int k = 0; k <DIRECTION_SIZE; ++k)
			{
				pInvMat[j][k] = pInvMat[j][k] - tmpa2[k] * inv_ji;
				matInv[j][k] = matInv[j][k] - tmpa[k] * inv_ji;
			}
		}
	}

	float inv_ee = matInv[DIRECTION_SIZE - 1][DIRECTION_SIZE - 1];
	for (int i = 0; i <DIRECTION_SIZE; ++i)
	{
		pInvMat[DIRECTION_SIZE - 1][i] = pInvMat[DIRECTION_SIZE - 1][i] / inv_ee;
		matInv[DIRECTION_SIZE - 1][i] = matInv[DIRECTION_SIZE - 1][i] / inv_ee;
	}

	for (int i = DIRECTION_SIZE - 1; i > 0; --i)
	{
		tmpa = matInv[i];
		tmpa2 = pInvMat[i];
		for (int j = i - 1; j >= 0; --j)
		{
			float inv_ji = matInv[j][i];
			for (int k = 0; k <DIRECTION_SIZE; ++k)
			{
				pInvMat[j][k] = pInvMat[j][k] - tmpa2[k] * inv_ji;
				matInv[j][k] = matInv[j][k] - tmpa[k] * inv_ji;
			}
		}
	}

	float sum_inv = 0.0;
	for (int i = 0; i <DIRECTION_SIZE; ++i)
	{
		sum_inv = 0.0;
		for (int j = 0; j <DIRECTION_SIZE; ++j)
			sum_inv += abs(matInv[i][j]);

		if (sum_inv < pow(10, -10))
		{
			printf("singular matrix");
			exit(0);
		}
	}
}
