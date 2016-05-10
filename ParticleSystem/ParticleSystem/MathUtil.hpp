#pragma once

#include "Defines.hpp"
#include <tuple>
typedef std::tuple<p_type, p_type, p_type> point;


inline float fInvSqrt(const float& in)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = in * 0.5F;
	y = in;
	i = *(long *)&y;
	i = 0x5f3759df - (i >> 1);
	y = *(float *)&i;
	y = y * (threehalfs - (x2 * y * y));
	y = y * (threehalfs - (x2 * y * y));	//extra precision
	return y;
}

inline point getNormal(const p_type* pos, const int& p1, const int &p2)	//returns normalize vector toward point2
{
	p_type diffx = (pos[p2] - pos[p1]);			//calculating difference between points
	p_type diffy = (pos[p2 + 1] - pos[p1 + 1]);
	p_type diffz = (pos[p2 + 2] - pos[p1 + 2]);

	p_type distsqr = abs(diffx*diffx + diffy*diffy + diffz*diffz);

	p_type invsqrt = fInvSqrt((float)distsqr);
	p_type normx = invsqrt*diffx;
	p_type normy = invsqrt*diffy;
	p_type normz = invsqrt*diffz;

	return std::make_tuple(normx, normy, normz);
}
inline point getTangent(const p_type* pos, const int& p1, const int& p2)
{
	point t = getNormal(pos, p1, p2);
	return std::make_tuple(std::get<1>(t), -std::get<0>(t), std::get<2>(t));	//flip 90 degrees in some direction ... :(
}

inline p_type getMag(const p_type* pos, const int & p1, const int& p2)
{
	p_type diffx = (pos[p2] - pos[p1]);			//calculating difference between points
	p_type diffy = (pos[p2 + 1] - pos[p1 + 1]);
	p_type diffz = (pos[p2 + 2] - pos[p1 + 2]);
	return  abs(diffx*diffx + diffy*diffy + diffz*diffz);
}

//from Origin
inline point getNormalO(const p_type* pos, const int& p1)	//returns normalize vector toward point2
{
	p_type diffx = (- pos[p1]);			//calculating difference between points
	p_type diffy = (- pos[p1 + 1]);
	p_type diffz = (- pos[p1 + 2]);

	p_type distsqr = abs(diffx*diffx + diffy*diffy + diffz*diffz);

	p_type invsqrt = fInvSqrt((float)distsqr);
	p_type normx = invsqrt*diffx;
	p_type normy = invsqrt*diffy;
	p_type normz = invsqrt*diffz;

	return std::make_tuple(normx, normy, normz);
}
inline point getTangentO(const p_type* pos, const int& p1)
{
	point t = getNormalO(pos, p1);
	return std::make_tuple(std::get<1>(t), -std::get<0>(t), std::get<2>(t));	//flip 90 degrees in some direction ... :(
}

inline p_type getMagO(const p_type* pos, const int & p1)
{
	p_type diffx = (- pos[p1]);			//calculating difference between points
	p_type diffy = (- pos[p1 + 1]);
	p_type diffz = (- pos[p1 + 2]);
	return  abs(diffx*diffx + diffy*diffy + diffz*diffz);
}

