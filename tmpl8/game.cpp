// Template, 2024 IGAD Edition
// Get the latest version from: https://github.com/jbikker/tmpl8
// IGAD/NHTV/BUAS/UU - Jacco Bikker - 2006-2024

#include "precomp.h"
using namespace tinybvh;
using namespace tinyocl;
#include "game.h"

#define AUTOCAM

// -----------------------------------------------------------
// Initialize the application
// -----------------------------------------------------------
void Game::Init()
{
	// create OpenCL kernels
	init = new Kernel( "wavefront2.cl", "SetRenderData" );
	clear = new Kernel( "wavefront2.cl", "Clear" );
	rayGen = new Kernel( "wavefront2.cl", "Generate" );
	extend = new Kernel( "wavefront2.cl", "Extend" );
	shade = new Kernel( "wavefront2.cl", "Shade" );
	updateCounters1 = new Kernel( "wavefront2.cl", "UpdateCounters1" );
	updateCounters2 = new Kernel( "wavefront2.cl", "UpdateCounters2" );
	traceShadows = new Kernel( "wavefront2.cl", "Connect" );
	finalize = new Kernel( "wavefront2.cl", "Finalize" );

	// we need the 'compute unit' or 'SM' count for wavefront rendering; ask OpenCL for it.
	clGetDeviceInfo( init->GetDeviceID(), CL_DEVICE_MAX_COMPUTE_UNITS, sizeof( size_t ), &computeUnits, NULL );

	// create OpenCL buffers for wavefront path tracing
	int N = SCRWIDTH * SCRHEIGHT;
	pixels = new Buffer( N * sizeof( uint32_t ) );
	raysIn = new Buffer( N * sizeof( bvhvec4 ) * 4 );
	raysOut = new Buffer( N * sizeof( bvhvec4 ) * 4 );
	connections = new Buffer( N * 3 * sizeof( bvhvec4 ) * 3 );
	accumulator = new Buffer( N * sizeof( bvhvec4 ) );
	pixels = new Buffer( N * sizeof( uint32_t ) );
	LoadBlueNoise();
	noise = new Buffer( 128 * 128 * 8 * sizeof( uint32_t ), blueNoise );
	noise->CopyToDevice();

	// load dragon mesh
	AddMesh( "./testdata/dragon.bin", 1, bvhvec3( 0 ) );
	swap( verts, dragonVerts );
	swap( triCount, dragonTriCount );
	dragon.Build( dragonVerts, dragonTriCount );

	// create dragon instances
	for (int d = 0; d < DRAGONS; d++)
	{
		instance[d + 1] = BLASInstance( 1 /* dragon */ );
		BLASInstance& i = instance[d + 1];
		float t = (float)d * 0.17f + 1.0f;
		float size = 0.1f + 0.075f * RandomFloat();
		bvhvec3 pos = splinePos( t );
		bvhvec3 D = -size * normalize( splinePos( t + 0.01f ) - pos );
		bvhvec3 U( 0, size, 0 );
		bvhvec3 R( -D.z, 0, D.x );
		pos += R * 20.0f * (RandomFloat() - 0.5f);
		i.transform[0] = R.x, i.transform[1] = R.y, i.transform[2] = R.z;
		i.transform[4] = U.x, i.transform[5] = U.y, i.transform[6] = U.z;
		i.transform[8] = D.x, i.transform[9] = D.y, i.transform[10] = D.z;
		i.transform[3] = pos.x;
		i.transform[7] = -9.2f;
		i.transform[11] = pos.z;
	}
	// load vertex data for static scenery
	AddQuad( bvhvec3( -22, 12, 2 ), 9, 5, 0x1ffffff ); // hard-coded light source
	AddMesh( "./testdata/bistro_ext_part1.bin", 1, bvhvec3( 0 ) );
	AddMesh( "./testdata/bistro_ext_part2.bin", 1, bvhvec3( 0 ) );

	// build bvh (here: 'compressed wide bvh', for efficient GPU rendering)
	bistro.Build( verts, triCount );
	instance[0] = BLASInstance( 0 /* static geometry */ );
	tlas.Build( instance, DRAGONS + 1, blasList, 2 );

	// create OpenCL buffers for BVH data
	tlasNodes = new Buffer( tlas.allocatedNodes /* could change! */ * sizeof( BVH_GPU::BVHNode ), tlas.bvhNode );
	tlasIndices = new Buffer( tlas.bvh.idxCount * sizeof( uint32_t ), tlas.bvh.triIdx );
	tlasNodes->CopyToDevice();
	tlasIndices->CopyToDevice();
	blasInstances = new Buffer( (DRAGONS + 1) * sizeof( BLASInstance ), instance );
	blasInstances->CopyToDevice();
	bistroNodes = new Buffer( bistro.usedBlocks * sizeof( bvhvec4 ), bistro.bvh8Data );
	bistroTris = new Buffer( bistro.idxCount * 3 * sizeof( bvhvec4 ), bistro.bvh8Tris );
	bistroVerts = new Buffer( triCount * 3 * sizeof( bvhvec4 ), verts );
	dragonNodes = new Buffer( dragon.usedBlocks * sizeof( bvhvec4 ), dragon.bvh8Data );
	dragonTris = new Buffer( dragon.idxCount * 3 * sizeof( bvhvec4 ), dragon.bvh8Tris );
	drVerts = new Buffer( dragonTriCount * 3 * sizeof( bvhvec4 ), dragonVerts );
	dragonNodes->CopyToDevice();
	dragonTris->CopyToDevice();
	drVerts->CopyToDevice();
	bistroNodes->CopyToDevice();
	bistroTris->CopyToDevice();
	bistroVerts->CopyToDevice();

	// load camera position / direction from file
	std::fstream t = std::fstream{ "camera_gpu.bin", t.binary | t.in };
	if (!t.is_open()) return;
	t.read( (char*)&rd, sizeof( rd ) );
}

// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Game::Tick( float /* deltaTime */ )
{
}