#define TINYBVH_IMPLEMENTATION
#define _CRT_SECURE_NO_WARNINGS
#include "tiny_bvh.h"

#ifndef _MSC_VER // testing code for watertight below.

// Minimal example for tiny_bvh.h

#include <cstdlib>
#include <cstdio>

#define TRIANGLE_COUNT	8192

tinybvh::bvhvec4 triangles[TRIANGLE_COUNT * 3]; // must be 16 byte!

float uniform_rand() { return (float)rand() / (float)RAND_MAX; }

int main()
{
	// create a scene consisting of some random small triangles
	for (int i = 0; i < TRIANGLE_COUNT; i++)
	{
		// create a random triangle
		tinybvh::bvhvec4& v0 = triangles[i * 3 + 0];
		tinybvh::bvhvec4& v1 = triangles[i * 3 + 1];
		tinybvh::bvhvec4& v2 = triangles[i * 3 + 2];
		// triangle position, x/y/z = 0..1
		float x = uniform_rand();
		float y = uniform_rand();
		float z = uniform_rand();
		// set first vertex
		v0.x = x + 0.1f * uniform_rand();
		v0.y = y + 0.1f * uniform_rand();
		v0.z = z + 0.1f * uniform_rand();
		// set second vertex
		v1.x = x + 0.1f * uniform_rand();
		v1.y = y + 0.1f * uniform_rand();
		v1.z = z + 0.1f * uniform_rand();
		// set third vertex
		v2.x = x + 0.1f * uniform_rand();
		v2.y = y + 0.1f * uniform_rand();
		v2.z = z + 0.1f * uniform_rand();
	}

	tinybvh::bvhvec3 O( 0.5f, 0.5f, -1 );
	tinybvh::bvhvec3 D( 0.1f, 0, 2 );
	tinybvh::Ray ray( O, D );

	// build a BVH over the scene
	tinybvh::BVH bvh;
	bvh.Build( triangles, TRIANGLE_COUNT );

	// from here: play with the BVH!
	int steps = bvh.Intersect( ray );
	printf( "std: nearest intersection: %f (found in %i traversal steps).\n", ray.hit.t, steps );

	// all done.
	return 0;
}

#else

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <vector>
using BVHT = tinybvh::BVH;

int main()
{
	FILE* obj = fopen( "dragon.obj", "r" );
	std::vector<tinybvh::bvhvec4> bvhv;
	std::vector<unsigned int> bvhi;
	char line[1024];
	while (fgets( line, sizeof( line ), obj ))
	{
		if (line[0] == 'v')
		{
			float x, y, z;
			sscanf( line + 1, "%f %f %f", &x, &y, &z );
			bvhv.push_back( { x, y, z, 0.f } );
		}
		else if (line[0] == 'f')
		{
			int a, b, c;
			sscanf( line + 1, "%d %d %d", &a, &b, &c );
			bvhi.push_back( a - 1 );
			bvhi.push_back( b - 1 );
			bvhi.push_back( c - 1 );
		}
	}
	printf( "Loaded obj: %d vertices, %d triangles\n", int( bvhv.size() ), int( bvhi.size() / 3 ) );
	BVHT bvh;
	bvh.Build( bvhv.data(), bvhi.data(), (uint32_t)bvhi.size() / 3 );
	float min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
	float max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	for (size_t i = 0; i < bvhv.size(); ++i)
		min[0] = fminf( min[0], bvhv[i].x ), min[1] = fminf( min[1], bvhv[i].y ),
		min[2] = fminf( min[2], bvhv[i].z ), max[0] = fmaxf( max[0], bvhv[i].x ),
		max[1] = fmaxf( max[1], bvhv[i].y ), max[2] = fmaxf( max[2], bvhv[i].z );
	const int N = 1024;
	const float bias = (max[1] - min[1]) * 1e-5f;
	FILE* ppm = fopen( "_test.ppm", "wb" );
	fprintf( ppm, "P6\n%d %d\n255\n", N, N );
	for (int y = 0; y < N; ++y)	for (int x = 0; x < N; ++x)
	{
		tinybvh::bvhvec3 origin = 
		{
			min[0] + (float( x ) / float( N )) * (max[0] - min[0]), max[1] + bias,
			min[2] + (float( y ) / float( N )) * (max[2] - min[2]),
		};
		tinybvh::bvhvec3 direction = { 0, -1, 0 };
		tinybvh::Ray ray( origin, direction );
		bvh.Intersect( ray );
		float hit = ray.hit.t;
		float dt = hit < BVH_FAR ? hit / (max[1] - min[0] + bias * 2) : 0.f;
		fputc( (int)(dt * 255), ppm );
		fputc( (hit < BVH_FAR) * 255, ppm );
		fputc( 0, ppm );
	}
	fclose( ppm );
}

#endif