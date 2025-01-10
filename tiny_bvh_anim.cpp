#define FENSTER_APP_IMPLEMENTATION
#define SCRWIDTH 800
#define SCRHEIGHT 600
#include "external/fenster.h" // https://github.com/zserge/fenster

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include <fstream>

using namespace tinybvh;

BVH bvh;
int frameIdx = 0, verts = 0;
bvhvec4* triangles = 0;

// setup view pyramid for a pinhole camera
static bvhvec3 eye( -15.24f, 21.5f, 2.54f ), p1, p2, p3;
static bvhvec3 view = normalize( bvhvec3( 0.826f, -0.438f, -0.356f ) );

void Init()
{
	// load raw vertex data for Crytek's Sponza
	std::fstream s{ "./testdata/cryteksponza.bin", s.binary | s.in };
	s.read( (char*)&verts, 4 );
	printf( "Loading triangle data (%i tris).\n", verts );
	verts *= 3, triangles = (bvhvec4*)malloc64( verts * 16 );
	s.read( (char*)triangles, verts * 16 );
	bvh.Build( triangles, verts / 3 );
}

bool UpdateCamera( float delta_time_s, fenster& f )
{
	bvhvec3 right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) ), up = 0.8f * cross( view, right );
	float moved = 0, spd = 10.0f * delta_time_s;
	if (f.keys['A'] || f.keys['D']) eye += right * (f.keys['D'] ? spd : -spd), moved = 1;
	if (f.keys['W'] || f.keys['S']) eye += view * (f.keys['W'] ? spd : -spd), moved = 1;
	if (f.keys['R'] || f.keys['F']) eye += up * 2.0f * (f.keys['R'] ? spd : -spd), moved = 1;
	if (f.keys[20]) view = normalize( view + right * -0.1f * spd ), moved = 1;
	if (f.keys[19]) view = normalize( view + right * 0.1f * spd ), moved = 1;
	if (f.keys[17]) view = normalize( view + up * -0.1f * spd ), moved = 1;
	if (f.keys[18]) view = normalize( view + up * 0.1f * spd ), moved = 1;
	// recalculate right, up
	right = normalize( cross( bvhvec3( 0, 1, 0 ), view ) ), up = 0.8f * cross( view, right );
	bvhvec3 C = eye + 1.2f * view;
	p1 = C - right + up, p2 = C + right + up, p3 = C - right - up;
	return moved > 0;
}

void Tick( float delta_time_s, fenster& f, uint32_t* buf )
{
	// handle user input and update camera
	bool moved = UpdateCamera( delta_time_s, f ) || frameIdx++ == 0;

	// clear the screen with a debug-friendly color
	for (int i = 0; i < SCRWIDTH * SCRHEIGHT; i++) buf[i] = 0xaaaaff;

	// trace rays
	const bvhvec3 L = normalize( bvhvec3( 1, 2, 3 ) );
	for (int ty = 0; ty < SCRHEIGHT / 4; ty++) for (int tx = 0; tx < SCRWIDTH / 4; tx++)
	{
		for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++)
		{
			float u = (float)(tx * 4 + x) / SCRWIDTH, v = (float)(ty * 4 + y) / SCRHEIGHT;
			bvhvec3 D = normalize( p1 + u * (p2 - p1) + v * (p3 - p1) - eye );
			Ray ray( eye, D, 1e30f );
			bvh.Intersect( ray );
			if (ray.hit.t < 10000)
			{
				int pixel_x = tx * 4 + x, pixel_y = ty * 4 + y, primIdx = ray.hit.prim;
				bvhvec3 v0 = triangles[primIdx * 3];
				bvhvec3 v1 = triangles[primIdx * 3 + 1];
				bvhvec3 v2 = triangles[primIdx * 3 + 2];
				bvhvec3 N = normalize( cross( v1 - v0, v2 - v0 ) );
				int c = (int)(255.9f * fabs( dot( N, L ) ));
				buf[pixel_x + pixel_y * SCRWIDTH] = c + (c << 8) + (c << 16);
			}
		}
	}
}

void Shutdown() { /* nothing here. */ }