
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

//------------------------------------------------------------------------------
//
//  swimmingShark.cpp -- Renders marine life OBJ models with a time-varying warping transform.
//
//------------------------------------------------------------------------------

// Models and textures copyright Toru Miyazawa, Toucan Corporation. Used by permission.
// http://toucan.web.infoseek.co.jp/3DCG/3ds/FishModelsE.html

#include "fishMonger.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

#include <sutil.h>
#include <GLUTDisplay.h>
#include <ImageLoader.h>

#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace optix;

float S = 0;
const float SPAN = 50.0f; // meter radius
const optix::Aabb SCENE_BOX( optix::make_float3( -SPAN, 0, -SPAN ), optix::make_float3( SPAN, 12.0f, SPAN ) );
optix::Aabb TargetBox( optix::make_float3( SCENE_BOX.m_min.x * 0.4f, SCENE_BOX.m_min.y, SCENE_BOX.m_max.z * 0.85f ),
                      optix::make_float3( SCENE_BOX.m_max.x * 0.4f, SCENE_BOX.m_max.y, SCENE_BOX.m_max.z ) );

//------------------------------------------------------------------------------
//
// Bubbles definition
//
//------------------------------------------------------------------------------

class Bubbles_t
{
public:
    Bubbles_t( const float3 &bubble_origin, Material TankMat, TextureSampler BlankTS, Context context )
        : m_bubble_origin( bubble_origin ),
        m_sphere_rad( SCENE_BOX.extent( 1 )*0.01f ),
        m_num_bubbles( 128 )
    {
        BB = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, m_num_bubbles );
        Buffer MB = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, m_num_bubbles );

        unsigned int *sphere_mats = reinterpret_cast<unsigned int *>( MB->map( ) );
        float4 *spheres = reinterpret_cast<float4 *>( BB->map( ) );

        for( size_t i=0; i<m_num_bubbles; i++ ) {
            float3 B = m_bubble_origin;
            B.y += SCENE_BOX.extent( 1 ) * i / float( m_num_bubbles );
            spheres[i] = make_float4( B, m_sphere_rad );
            sphere_mats[i] = 0;
        }

        MB->unmap( );
        BB->unmap( );

        std::string ptx_path = SampleScene::ptxpath( "swimmingShark", "sphere_list.cu" );
        G = context->createGeometry( );
        G->setBoundingBoxProgram( context->createProgramFromPTXFile( ptx_path, "bounds" ) );
        G->setIntersectionProgram( context->createProgramFromPTXFile( ptx_path, "intersect" ) );
        G->setPrimitiveCount( static_cast<unsigned int>(m_num_bubbles) );

        G["sphere_buffer"]->setBuffer( BB );
        G["material_buffer"]->setBuffer( MB );

        GeometryInstance GI = context->createGeometryInstance( G, &TankMat, &TankMat+1 );
        GI["caustic_map"]->setTextureSampler( BlankTS );
        GI["diffuse_map"]->setTextureSampler( BlankTS );
        GI["diffuse_map_scale"]->setFloat( 1.0f );
        GI["emission_color"]->setFloat( 0 );
        GI["Kr"]->setFloat( 1.0f );

        GG = context->createGeometryGroup( );
        GG->setChildCount( 1u );
        GG->setChild( 0, GI );
        GG->setAcceleration( context->createAcceleration( "MedianBvh", "Bvh" ) );
    }

    void updateGeometry( )
    {
        float4 *spheres = reinterpret_cast<float4 *>( BB->map( ) );

        for( size_t i=0; i<m_num_bubbles; i++ ) {
            const float speed = SCENE_BOX.extent( 1 )*0.01f;
            float3 P = make_float3( spheres[i] );
            P += MakeDRand( make_float3( -speed, speed, -speed ), make_float3( speed, speed*0.9f, speed ) );
            if( P.y > SCENE_BOX.m_max.y ) P.y = m_bubble_origin.y;
            spheres[i] = make_float4( P, spheres[i].w );
        }

        BB->unmap( );

        // Mark the accel structure and geometry as dirty so they will be rebuilt.
        //G->markDirty( );
        GG->getAcceleration( )->markDirty( );
    }

    GeometryGroup GG;

private:
    Geometry G;
    Buffer BB;

    float3 m_bubble_origin;
    float  m_sphere_rad;
    size_t m_num_bubbles;

	#define CROSSPROD(p1,p2,p3) \
		p3.x = p1.y*p2.z - p1.z*p2.y; \
		p3.y = p1.z*p2.x - p1.x*p2.z; \
		p3.z = p1.x*p2.y - p1.y*p2.x
};

//------------------------------------------------------------------------------
//
// TankScene definition
//
//------------------------------------------------------------------------------

class TankScene : public SampleScene
{
public:
  TankScene( const std::string& objfilename, const std::string& objpath, const std::string& texturepath,
    int num_species_to_load, int fish_per_species )
  : m_objfilename( objfilename ), m_objpath( objpath ), m_texturepath( texturepath ),
    m_num_species_to_load( num_species_to_load ), m_fish_per_species( fish_per_species )
  { }

  ~TankScene( )
  { }

  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual Buffer getOutputBuffer( );
  virtual bool   keyPressed( unsigned char key, int x, int y );

private:
  void updateGeometry( );
  void createGeometry( );
  Geometry createHeightField( );
  Geometry createParallelogram( Program pbounds, Program pisect, float3 anchor, float3 v1, float3 v2 );
  void createGroundData( );

  Group         TopGroup;
  Buffer        GroundBuf;
  Material      TankMat;
  TextureSampler CausticTS;

  std::vector<Fish_t *> Fish;
  Bubbles_t *   Bubbles;

  std::string   m_objfilename, m_objpath, m_texturepath;
  float         m_scene_epsilon;
  float         m_ground_ymin, m_ground_ymax;
  int           m_num_species_to_load, m_fish_per_species;
  bool          m_animate;
  int           m_anaglyphic = 0;
  float		    m_dist;

  const static int         WIDTH;
  const static int         HEIGHT;
  const static int         GROUND_WID;
  const static int         GROUND_HGT;
  const static float3      WATER_BLUE;
};

const int TankScene::WIDTH  = 800;
const int TankScene::HEIGHT = 600;
const int TankScene::GROUND_WID = 15;
const int TankScene::GROUND_HGT = 15;
const float3 TankScene::WATER_BLUE = make_float3( 0.192f, 0.498f, 0.792f );

void TankScene::createGeometry( )
{
  // Make fish models
  if( !m_objfilename.empty( ) ) {
    for( int f=0; f<m_fish_per_species; f++ ) {
      Fish.push_back( new Fish_t( m_objfilename, TankMat, CausticTS, NULL, m_context, f ? Fish.back( )->m_geom_group : static_cast<GeometryGroup>(NULL) ) );
    }
  }

  // We want a jittered sampling of fish if the array is sorted by size.  Divide
  // num_species by m_num_species_to_load to get the bins to sample from, then sample
  // within that bin.
  float species_per_bin = static_cast<float>(FishMonger_t::num_species)/m_num_species_to_load;
  
  for( int s=0; s<m_num_species_to_load; s++ ) {
    int sp = static_cast<int>(species_per_bin * ( s + FRand()));
    if (sp >= FishMonger_t::num_species) sp = FishMonger_t::num_species-1;
    std::string fname = m_objpath + FishMonger_t::SpeciesInfo[sp].name;

    for( int f=0; f<m_fish_per_species; f++ ) {
      Fish.push_back( new Fish_t( fname, TankMat, CausticTS, NULL, m_context, f ? Fish.back( )->m_geom_group : static_cast<GeometryGroup>(NULL) ) );
    }
  }

  // Make tank
  std::cerr << "Initializing tank ...";

  // Geometry
  std::string ptx_path = SampleScene::ptxpath( "swimmingShark", "parallelogram.cu" );
  Program pbounds = m_context->createProgramFromPTXFile( ptx_path, "bounds" );
  Program pisect = m_context->createProgramFromPTXFile( ptx_path, "intersect" );

  Geometry WaterSurfaceG = createParallelogram( pbounds, pisect,
    make_float3( SCENE_BOX.m_min.x, SCENE_BOX.m_max.y, SCENE_BOX.m_min.z ),
    make_float3( SCENE_BOX.extent( 0 ), 0, 0 ),
    make_float3( 0, 0, SCENE_BOX.extent( 2 ) ) );

  TextureSampler ConstTS = loadTexture( m_context, "", WATER_BLUE );

  // Water Surface
  GeometryInstance WaterSurfaceGI = m_context->createGeometryInstance( WaterSurfaceG, &TankMat, &TankMat+1 );
  WaterSurfaceGI["caustic_map"]->setTextureSampler( ConstTS );
  WaterSurfaceGI["diffuse_map"]->setTextureSampler( CausticTS );
  WaterSurfaceGI["diffuse_map_scale"]->setFloat( 12.0f );
  WaterSurfaceGI["emission_color"]->setFloat( WATER_BLUE*0.7f );
  WaterSurfaceGI["Kr"]->setFloat( 1.0f );

  Geometry GroundG = createHeightField( );

  GeometryInstance GroundGI = m_context->createGeometryInstance( GroundG, &TankMat, &TankMat+1 );
  GroundGI["caustic_map"]->setTextureSampler( ConstTS );
  GroundGI["diffuse_map"]->setTextureSampler( loadTexture( m_context, m_texturepath + "/sand.ppm", make_float3( 1, 1, 0 ) ) );
  GroundGI["diffuse_map_scale"]->setFloat( 18.0f );
  GroundGI["emission_color"]->setFloat( WATER_BLUE*0.4f );
  GroundGI["Kr"]->setFloat( 0 );

  GeometryGroup SceneGG = m_context->createGeometryGroup( );
  SceneGG->setAcceleration( m_context->createAcceleration( "Sbvh","Bvh" ) );
  SceneGG->setChildCount( static_cast<unsigned int>( 2 ) );

  SceneGG->setChild( 0, GroundGI );
  SceneGG->setChild( 1, WaterSurfaceGI );

  // Make bubbles
  //Bubbles = new Bubbles_t( make_float3( SCENE_BOX.m_max.x*0.3f, SCENE_BOX.m_min.y, SCENE_BOX.m_max.z*0.5f ), TankMat, ConstTS, m_context );
  Bubbles = new Bubbles_t(make_float3(0, 0, 50.0f), TankMat, ConstTS, m_context);

  unsigned int numFish = static_cast<unsigned int>(Fish.size());

  // Make overall group
  TopGroup = m_context->createGroup( );
  TopGroup->setChildCount( numFish + 1 + ( Bubbles ? 1 : 0 ) ); // Each fish plus the tank plus the bubbles

  for( unsigned int i=0; i<numFish; i++ )
    TopGroup->setChild( i, Fish[i]->m_tran );
  TopGroup->setChild( numFish, SceneGG );
  if( Bubbles ) TopGroup->setChild( numFish + 1, Bubbles->GG );
  TopGroup->setAcceleration( m_context->createAcceleration( "MedianBvh","Bvh" ) );

  m_context["top_object"]->set( TopGroup );
  m_context["top_shadower"]->set( TopGroup );

  std::cerr << "finished." << std::endl;
}

void TankScene::initScene( InitialCameraData& camera_data )
{
  // Setup context
  m_context->setRayTypeCount( 2 );
  m_context->setEntryPointCount( 1 );
  m_context->setStackSize( 1350 );
  m_context->setPrintEnabled( false );
  m_context->setPrintBufferSize( 1024 );
  m_context->setPrintLaunchIndex( 400,300 );

  m_context[ "radiance_ray_type" ]->setUint( 0u );
  m_context[ "shadow_ray_type" ]->setUint( 1u );
  m_scene_epsilon = 1.e-3f;
  m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
  m_context[ "max_depth" ]->setInt( 3 );

  // Output buffer
  m_context["output_buffer"]->set( createOutputBuffer( RT_FORMAT_UNSIGNED_BYTE4, WIDTH, HEIGHT ) );

  // Ray generation program
  std::string ptx_path = ptxpath( "swimmingShark", "pinhole_camera.cu" );
  Program ray_gen_program = m_context->createProgramFromPTXFile( ptx_path, "pinhole_camera" );
  m_context->setRayGenerationProgram( 0, ray_gen_program );

  // Exception / miss programs
  m_context->setExceptionProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "exception" ) );
  m_context[ "bad_color" ]->setFloat( 0, 1.0f, 0 );

  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "constantbg.cu" ), "miss" ) );
  m_context[ "bg_color" ]->setFloat( WATER_BLUE );

  // Set up the material. Some uses will override certain parameters.
  TankMat = m_context->createMaterial( );
  TankMat->setClosestHitProgram( 0, m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "tank_material.cu" ), "closest_hit_radiance" ) );
  TankMat->setAnyHitProgram( 1, m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "tank_material.cu" ), "any_hit_shadow" ) );

  TankMat["ambient_light_color"]->setFloat( WATER_BLUE );
  TankMat["ambient_light_color"]->setFloat(make_float3(1.0f, 1.0f, 1.0f));
  TankMat["attenuation_color"]->setFloat( WATER_BLUE );
  //TankMat["attenuation_density"]->setFloat( 0.0f ); // Must be < 0.
  TankMat["attenuation_density"]->setFloat(-0.045f); // Must be < 0.

  TankMat["caustic_light_color"]->setFloat( 1.6f, 1.6f, 1.6f );
  TankMat["caustic_map_scale"]->setFloat( 0.3f );
  TankMat["light_dir"]->setFloat( 0, 1.0f, 0 );

  CausticTS = loadTexture( m_context, m_texturepath + "/caustic.ppm", make_float3( 1, 0, 0 ) );

  // Set up geometry
  createGeometry( );

  // Set up camera
  float3 eye = make_float3( 0.0001f, SCENE_BOX.center( 1 ), SCENE_BOX.m_max.z + SCENE_BOX.m_max.y * 0.9f );
  //float3 lookat = SCENE_BOX.center( );
  float3 lookat = make_float3(SCENE_BOX.center(0), SCENE_BOX.center(1), 50.0f);

  camera_data = InitialCameraData(
    eye,                       // eye
    lookat,                    // lookat
    make_float3( 0, 1.0f, 0 ), // up
    50.0f );                   // vfov

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context["eye"]->setFloat( make_float3( 0, 0, 0 ) );
  m_context["U"]->setFloat( make_float3( 0, 0, 0 ) );
  m_context["V"]->setFloat( make_float3( 0, 0, 0 ) );
  m_context["W"]->setFloat( make_float3( 0, 0, 0 ) );

  for( size_t i=0; i<Fish.size( ); i++ )
    Fish[i]->initAnimation( TargetBox, -1.0f, false );
  m_animate = true;

  // Prepare to run
  m_context->validate( );
  m_context->compile( );
}

void TankScene::createGroundData( )
{
  int N = ( GROUND_WID+1 ) * ( GROUND_HGT+1 );
  GroundBuf = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT, GROUND_WID+1, GROUND_HGT+1 );
  float* p = reinterpret_cast<float*>( GroundBuf->map( ) );

  // Compute data range
  m_ground_ymin = 1e9;
  m_ground_ymax = -1e9;

  const float ground_scale = 0.8f; // meters

  for( int i = 0; i<N; i++ ) {
    p[i] = FRand( ground_scale );
    m_ground_ymin = fminf( m_ground_ymin, p[i] );
    m_ground_ymax = fmaxf( m_ground_ymax, p[i] );
  }

  m_ground_ymin -= 1.e-6f;
  m_ground_ymax += 1.e-6f;
  GroundBuf->unmap( );
}

Geometry TankScene::createHeightField( )
{
  createGroundData( );

  Geometry HgtFldG = m_context->createGeometry( );
  HgtFldG->setPrimitiveCount( 1u );

  HgtFldG->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "heightfield.cu" ), "bounds" ) );
  HgtFldG->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "swimmingShark", "heightfield.cu" ), "intersect" ) );
  float3 min_corner = make_float3( SCENE_BOX.m_min.x, m_ground_ymin, SCENE_BOX.m_min.z );
  float3 max_corner = make_float3( SCENE_BOX.m_max.x, m_ground_ymax, SCENE_BOX.m_max.z );
  RTsize nx, nz;
  GroundBuf->getSize( nx, nz );

  // If buffer is nx by nz, we have nx-1 by nz-1 cells;
  float3 cellsize = ( max_corner - min_corner ) / ( make_float3( static_cast<float>( nx-1 ), 1.0f, static_cast<float>( nz-1 ) ) );
  cellsize.y = 1;
  float3 inv_cellsize = make_float3( 1 )/cellsize;
  HgtFldG["boxmin"]->setFloat( min_corner );
  HgtFldG["boxmax"]->setFloat( max_corner );
  HgtFldG["cellsize"]->setFloat( cellsize );
  HgtFldG["inv_cellsize"]->setFloat( inv_cellsize );
  HgtFldG["data"]->setBuffer( GroundBuf );

  return HgtFldG;
}

Geometry TankScene::createParallelogram( Program pbounds, Program pisect, float3 anchor, float3 v1, float3 v2 )
{
  Geometry parallelogram = m_context->createGeometry( );
  parallelogram->setPrimitiveCount( 1u );
  parallelogram->setBoundingBoxProgram( pbounds );
  parallelogram->setIntersectionProgram( pisect );

  float3 normal = cross( v1, v2 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );

  float4 plane = make_float4( normal, d );

  parallelogram["plane"]->setFloat( plane );
  parallelogram["v1"]->setFloat( v1 );
  parallelogram["v2"]->setFloat( v2 );
  parallelogram["anchor"]->setFloat( anchor );

  return parallelogram;
}

void TankScene::updateGeometry( )
{
  if( !m_animate )
    return;

  for( size_t i=0; i<Fish.size( ); i++ ) {
    Fish[i]->updateGeometry( TargetBox );

    // Implement schooling by copying the target position of another fish of the same species
    if( i>0 && !Fish[i]->owns_geom_group() && ( rand( ) % 3 == 0 ) )
      Fish[i]->target_pos( Fish[i-1]->target_pos() );
  }

  if( Bubbles ) Bubbles->updateGeometry( );

  TopGroup->getAcceleration( )->markDirty( );
}


/*
Normalise a vector
*/
void Normalise(float3 *p){
	double length;

	length = sqrt(p->x * p->x + p->y * p->y + p->z * p->z);
	if (length != 0) {
		p->x /= length;
		p->y /= length;
		p->z /= length;
	}
	else {
		p->x = 0;
		p->y = 0;
		p->z = 0;
	}
}

/*
Calculate the unit normal at p given two other points
p1,p2 on the surface. The normal points in the direction
of p1 crossproduct p2
*/
float3 CalcNormal(float3 p, float3 p1, float3 p2)
{
	float3 n, pa, pb;

	pa.x = p1.x - p.x;
	pa.y = p1.y - p.y;
	pa.z = p1.z - p.z;
	pb.x = p2.x - p.x;
	pb.y = p2.y - p.y;
	pb.z = p2.z - p.z;
	Normalise(&pa);
	Normalise(&pb);

	n.x = pa.y * pb.z - pa.z * pb.y;
	n.y = pa.z * pb.x - pa.x * pb.z;
	n.z = pa.x * pb.y - pa.y * pb.x;
	Normalise(&n);

	return(n);
}

void TankScene::trace(const RayGenCameraData& camera_data){
	float3 pos = camera_data.eye;
	float3 look = camera_data.W;

	float alfa = atan2(look.z - pos.z, look.x - pos.x);

	float3 posA, posB;

	posA.x = pos.x - sin(alfa) * S;
	posA.y = pos.y;
	posA.z = pos.z + cos(alfa) * S;

	posB.x = pos.x + sin(alfa) * S;
	posB.y = pos.y;
	posB.z = pos.z - cos(alfa) * S;

	m_context["eye"]->setFloat(camera_data.eye);
	m_context["U"]->setFloat(camera_data.U);
	m_context["V"]->setFloat(camera_data.V);
	m_context["W"]->setFloat(camera_data.W);

	m_context["posA"]->setFloat(posA);
	m_context["posB"]->setFloat(posB);
	m_context["lookA"]->setFloat(look);
	m_context["lookB"]->setFloat(look);
	m_context["anaglyphic"]->setInt(m_anaglyphic);

	Buffer buffer = m_context["output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize(buffer_width, buffer_height);

	//Update the vertex positions before rendering
	updateGeometry();

	m_context->launch(0, buffer_width, buffer_height);
}

//void TankScene::trace( const RayGenCameraData& camera_data )
//{
//	float3 right, focus;
//
//	float3 vp, vd, vu;
//	vp = camera_data.eye;
//	vd = camera_data.W;
//	vu = camera_data.up;
//	m_dist = length(camera_data.W);
//
//	//float focallength = 6;
//	float focallength = m_dist/4;
//	float eyesep = focallength / 30.0;
//
//	/* Determine the focal point */
//	Normalise(&vd);
//	focus.x = vp.x + focallength * vd.x;
//	focus.y = vp.y + focallength * vd.y;
//	focus.z = vp.z + focallength * vd.z;
//
//	printf("focal: %f %f %f\n", focus.x, focus.y, focus.z);
//	printf("W: %f %f %f\n", camera_data.W.x, camera_data.W.y, camera_data.W.z);
//
//	/* Derive the the "right" vector */
//	CROSSPROD(vd, vu, right);
//	Normalise(&right);
//	right.x *= eyesep / 2.0;
//	right.y *= eyesep / 2.0;
//	right.z *= eyesep / 2.0;
//
//	  m_context["eye"]->setFloat( camera_data.eye );
//	  m_context["U"]->setFloat( camera_data.U );
//	  m_context["V"]->setFloat( camera_data.V );
//	  m_context["W"]->setFloat( camera_data.W );
//
//	  m_context["vp"]->setFloat(vp);
//	  m_context["vd"]->setFloat(vd);
//	  m_context["vu"]->setFloat(vu);
//	  m_context["focus"]->setFloat(focus);
//	  m_context["right"]->setFloat(right);
//	  m_context["anaglyphic"]->setInt(m_anaglyphic);
//
//  Buffer buffer = m_context["output_buffer"]->getBuffer( );
//  RTsize buffer_width, buffer_height;
//  buffer->getSize( buffer_width, buffer_height );
//
//  //Update the vertex positions before rendering
//  updateGeometry( );
//
//  m_context->launch( 0, buffer_width, buffer_height );
//}

Buffer TankScene::getOutputBuffer( )
{
  return m_context["output_buffer"]->getBuffer( );
}

bool TankScene::keyPressed( unsigned char key, int x, int y )
{
  switch ( key ) {
  case 'e':
    m_scene_epsilon /= 10.0f;
    std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
    m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
    return true;
  case 'E':
    m_scene_epsilon *= 10.0f;
    std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
    m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
    return true;
  case 'a':
    m_animate = !m_animate;
    return true;
  case ' ':
	  m_anaglyphic = !m_anaglyphic;
	  return true;
  case 'l':
	  printf("%f\n", m_dist);
	  return true;
  case 'p':
	  for (size_t i = 0; i < Fish.size(); i++) {
		  float3 pos = Fish[i]->target_pos();
		  printf("%d - %f %f %f\n", i, pos.x, pos.y, pos.z);
	  }
	  return true;
  case '+':
	  S += 0.01;
	  printf("S = %f\n", S);
	  return true;
  case '-':
	  S -= 0.01;
	  printf("S = %f\n", S);
	  return true;
  }
  

  return false;
}

//------------------------------------------------------------------------------
//
//  main driver
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj <obj_file>                     Specify .OBJ model to be rendered\n"
    << "  -f  | --species <num>                      Specify the number of species to load\n"
    << "  -n  | --school-size <num>                  Specify the number of fish of each species to load\n"
    << "  -P  | --objpath <obj_path>                 Specify path to the OBJ models\n"
    << "  -t  | --texpath <tex_path>                 Specify path to the sand, water and caustic textures\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  e Decrease scene epsilon size (used for shadow ray offset)\n"
    << "  E Increase scene epsilon size (used for shadow ray offset)\n"
    << "  a Toggle animation\n"
    << std::endl;

  if ( doExit ) exit(1);
}

int main( int argc, char** argv )
{
  GLUTDisplay::init( argc, argv );

  // Process command line options
  std::string objfilename, objpath, texturepath;
  int num_species_to_load = 0, fish_per_species = 7;
  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if ( arg == "--help" || arg == "-h" ) {
      printUsageAndExit( argv[0] );
    } else if ( arg == "--species" || arg == "-f" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      num_species_to_load = atoi( argv[++i] );
    } else if ( arg == "--school-size" || arg == "-n" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      fish_per_species = atoi( argv[++i] );
    } else if ( arg == "--objpath" || arg == "-P" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      objpath = argv[++i];
    } else if ( arg == "--obj" || arg == "-o" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      objfilename = argv[++i];
    } else if ( arg == "--texpath" || arg == "-t" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      texturepath = argv[++i];
    } else {
      std::cerr << "Unknown option: '" << arg << "'\n";
      printUsageAndExit( argv[0] );
    }
  }

  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  if( objfilename.empty( ) && num_species_to_load==0 )
    num_species_to_load = 7;
  if( objpath.empty( ) ) {
    objpath = std::string( sutilSamplesDir() ) + "/swimmingShark/Fish_OBJ_PPM/";
  }
  if( texturepath.empty( ) ) {
    texturepath = std::string( sutilSamplesDir() ) + "/swimmingShark/";
  }

  if( !GLUTDisplay::isBenchmark( ) ) {
    // With Unix rand( ), a small magnitude change in input seed yields a small change in the first random number. Duh!
    unsigned int tim = static_cast<unsigned int>( time( 0 ) );
    unsigned int tim2 = ( ( tim & 0xff ) << 24 ) | ( ( tim & 0xff00 ) << 8 ) | ( ( tim & 0xff0000 ) >> 8 ) | ( ( tim & 0xff000000 ) >> 24 );
    srand( tim2 );
  }

  try {
    TankScene scene( objfilename, objpath, texturepath, num_species_to_load, fish_per_species );
    GLUTDisplay::run( "Swimming Shark", &scene, GLUTDisplay::CDAnimated );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString( ).c_str( ) );
    exit( 1 );
  }

  return 0;
}
