#include "../swimmingShark/fishMonger.h"

#include <optixu/optixu_math_stream_namespace.h>

#include <OptixMesh.h>

#include <iostream>
#include <cstdlib>

#include <memory.h>

using namespace optix;

// Note that I sorted these fish based on lengthMeters (column 3) instead of file name using:
// column -t speciesInfo.h | sort -k 3 -n > speciesInfo.h
const Species_t FishMonger_t::SpeciesInfo[] = {
#include "../swimmingShark/speciesInfo.h"
};

const int FishMonger_t::num_species = sizeof( SpeciesInfo ) / sizeof( Species_t );

const Species_t * FishMonger_t::FindSpecies(std::string &name)
{
  for( int i=0; i<num_species; i++ )
    if( name.find( SpeciesInfo[i].name ) != std::string::npos )
      return SpeciesInfo + i;

  std::cerr << "Couldn't find " << name << ". Using " << SpeciesInfo[0].name << ".\n";

  return SpeciesInfo;
}

Fish_t::Fish_t(std::string &objfilename, Material matl, TextureSampler projectedTexSamp, Program intersectProgram,
               Context context, GeometryGroup inGG /*= GeometryGroup( 0 ) */) : m_geom_group( inGG ), m_this_owns_geom_group( false )
{
  static bool printedPermissions = false;
  if( !printedPermissions ) {
    if( objfilename.find( "Fish_OBJ_PPM" ) != std::string::npos ) {
      std::cout << "\nModels and textures copyright Toru Miyazawa, Toucan Corporation. Used by permission.\n";
      std::cout << "http://toucan.web.infoseek.co.jp/3DCG/3ds/FishModelsE.html\n\n";
      printedPermissions = true;
    }
  }

  m_species = FishMonger_t::FindSpecies( objfilename );

  // std::cerr << "Found name: " << m_species->name << '\n';
  GeometryInstance GI;
  if( m_geom_group.get() == 0 ) {
    std::cerr << "Loading " << objfilename << '\n';
    m_this_owns_geom_group = true;

    m_geom_group = context->createGeometryGroup();
    OptixMesh FishLoader( context, m_geom_group, matl );

    float m[16] = {
      0,-1,0,0,
      0,0,1,0,
      -1,0,0,0,
      0,0,0,1
    };
    Matrix4x4 Rot( m );
    Matrix4x4 XForm = Rot;
    XForm = Matrix4x4::scale( make_float3( 1.0f/m_species->sizeInObj ) ) * XForm;
    XForm = Matrix4x4::translate( make_float3( 0, 0, m_species->centerOffset ) ) * XForm;
    XForm = Matrix4x4::scale( make_float3( m_species->lengthMeters ) ) * XForm;
    
    FishLoader.setLoadingTransform( XForm );
    if( intersectProgram )
      FishLoader.setDefaultIntersectionProgram( intersectProgram );

    FishLoader.loadBegin_Geometry( objfilename );
    FishLoader.loadFinish_Materials();
    
    m_aabb = FishLoader.getSceneBBox();

    // Set the material properties that differ between the fish and the other scene elements
    for (unsigned int i = 0; i < m_geom_group->getChildCount(); ++i) {
      GI = m_geom_group->getChild( i );
      if ( projectedTexSamp )
        GI["caustic_map"]->setTextureSampler( projectedTexSamp );
      GI["diffuse_map_scale"]->setFloat( 1.0f );
      GI["emission_color"]->setFloat( 0 );
      GI["Kr"]->setFloat( 0 );
    }

    // Select an AS builder that allows refit, unlike the OptixMesh default.
    Acceleration AS = m_geom_group->getAcceleration();
    AS->setBuilder( "Bvh" );
    AS->setProperty( "refit", "1" );
  } else {
    GI = m_geom_group->getChild( 0 );
  }

  m_geom = GI->getGeometry();
  m_vert_buff = m_geom["vertex_buffer"]->getBuffer();
  m_vert_buff->getSize( m_num_verts ); // Query number of vertices in the buffer

  m_norm_buff = m_geom["normal_buffer"]->getBuffer();
  m_norm_buff->getSize( m_num_norms ); // Query number of normals in the buffer, which doesn't match the number of vertices

  m_vindices_buff = m_geom["vindex_buffer"]->getBuffer();
  m_vindices_buff->getSize( m_num_tris );

  m_nindices_buff = m_geom["nindex_buffer"]->getBuffer();

  m_tran = context->createTransform();
  m_tran->setChild( m_geom_group );
}

void Fish_t::initAnimation( optix::Aabb TargetBox, float fishFrac, bool deterministic )
{
  m_swim_deterministically = deterministic;

  if( m_this_owns_geom_group ) {
    // Get a pointer to the initial, non-deformed buffer data
    float3* Verts = ( float3* )m_vert_buff->map();
    float3* Norms = ( float3* )m_norm_buff->map();
    int* Vinds = ( int* )m_vindices_buff->map();
    int* Ninds = ( int* )m_nindices_buff->map();

    m_animated_points.resize( ANIM_STEPS );
    m_animated_normals.resize( ANIM_STEPS );

    for ( int ph = 0; ph < ANIM_STEPS; ph++ ) {
      // Compute this frame of the animation
      float phaseDeg = 360.0f * ph / float( ANIM_STEPS - 1 ); // The phase in degrees

      m_animated_points[ph].resize( m_num_verts );
      m_animated_normals[ph].resize( m_num_norms );

      // Initialize vertices and normals to dirty
      for ( size_t t = 0; t < m_num_verts; t++ )
        m_animated_points[ph][t] = make_float3(-1024.0f,0.0f,0.0f);
      for ( size_t t = 0; t < m_num_norms; t++ )
        m_animated_normals[ph][t] = make_float3(-1024.0f,0.0f,0.0f);

      // Loop over each index (3 per triangle)
      for ( size_t t = 0; t < 3u * m_num_tris; t++ ) {
        if( m_animated_points[ph][Vinds[t]].x == -1024.0f || m_animated_normals[ph][Ninds[t]].x == -1024.0f ) { // If not dirty, avoid a lot of computation
          float3 vert(Verts[Vinds[t]]);
          float3 norm(Norms[Ninds[t]]); // Norms and Verts are indexed separately, so have to use indices to find corresponding normals
          norm = normalize( norm );

          swimVertex( phaseDeg, vert, norm );
          
          m_animated_points[ph][Vinds[t]] = vert;
          m_animated_normals[ph][Ninds[t]] = norm;

          float l = dot( m_animated_normals[ph][Ninds[t]], m_animated_normals[ph][Ninds[t]] );
          if (l < 0.99f || l > 1.01f) {
            // Compute a facet normal for the bad normal
            size_t t0 = (t / 3u) * 3u;
            float3 leg0 = m_animated_points[ph][Vinds[t0+1]] - m_animated_points[ph][Vinds[t0]];
            float3 leg1 = m_animated_points[ph][Vinds[t0+2]] - m_animated_points[ph][Vinds[t0]];
            float3 newn = normalize(cross(leg0, leg1));
            m_animated_normals[ph][Ninds[t]] = newn;
          }
        }
      } // t
    }   // ph

    // Unmap buffer
    m_vert_buff->unmap();
    m_norm_buff->unmap();
    m_vindices_buff->unmap();
    m_nindices_buff->unmap();
  }

  if( !m_swim_deterministically ) {
    m_pos = MakeDRand( TargetBox.m_min, TargetBox.m_max );
    m_vel = make_float3( 0,0,1 );
    m_target_pos = MakeDRand( TargetBox.m_min, TargetBox.m_max );
  }
  else { // start deterministically in a circle
    m_pos = TargetBox.center() + make_float3(cosf(2.0f*M_PIf*fishFrac), fishFrac*0.7f, sinf(2.0f*M_PIf*fishFrac)) * (TargetBox.maxExtent() * 0.5f);
    m_vel = make_float3( 0,0,1 );
    m_target_pos = m_pos + m_vel; // will be ignored
  }

  m_phase_num = rand() % ANIM_STEPS;
  m_frames_this_target = 10000;
}

void Fish_t::swimPoint(float3 &P, const float phaseDeg)
{
  const float DtoR = float( M_PI / 180.0 );

  // z >= 0 is the front of the model, and the movement of the head can be reduced.
  // angle = phase factor * damping * full amplitude.
  float wave_scale = ( P.z >= 0 ) ? P.z / m_aabb.m_max.z : -P.z / m_aabb.m_min.z;
  float wave_len = ( P.z >= 0 ) ? m_species->headWaveLen : m_species->tailWaveLen;
  float damping = wave_scale; // Scales the wave by distance from origin;
  float rotAngRad = sin( ( -phaseDeg - wave_len * wave_scale ) * DtoR ) * damping * m_species->maxAmplRad;

  // Rotate about +Y
  float3 Pnew(make_float3( cos( rotAngRad ) * P.x + sin( rotAngRad ) * P.z, P.y, -sin( rotAngRad ) * P.x + cos( rotAngRad ) * P.z ));
  P = Pnew;
}

// Calculate the rotation angle of the current model point depending on the z-coordinate of the point
inline void Fish_t::swimVertex( const float phaseDeg, float3 &P, float3& N )
{
  float delta = 0.001f * m_species->lengthMeters;

  float3 Y = make_float3(0,1,0);
  float3 B = cross(N, Y);
  float3 T = cross(B, N);
  float3 Btip = P + B * delta;
  float3 Ttip = P + T * delta;

  float3 Pnew = P;
  swimPoint(Pnew, phaseDeg);
  float3 Btipnew = Btip;
  swimPoint(Btipnew, phaseDeg);
  float3 Ttipnew = Ttip;
  swimPoint(Ttipnew, phaseDeg);

  float3 Bnew = normalize(Btipnew - Pnew);
  float3 Tnew = normalize(Ttipnew - Pnew);
  float3 Nnew = normalize(cross(Tnew, Bnew));

  P = Pnew;
  N = Nnew;
}

void Fish_t::updateGeometry(optix::Aabb TargetBox)
{
  if( m_this_owns_geom_group ) {
    // We have precomputed every animation pose. Here we copy the relevant one into place.
    assert( m_animated_points[m_phase_num].size() == m_num_verts );
    memcpy( ( float3* )m_vert_buff->map(), &( m_animated_points[m_phase_num][0] ), m_num_verts * sizeof( float3 ) );
    m_vert_buff->unmap();

    // Copy the transformed normals for this frame
    assert( m_animated_normals[m_phase_num].size() == m_num_norms );
    memcpy( ( float3* )m_norm_buff->map(), &( m_animated_normals[m_phase_num][0] ), m_num_norms * sizeof( float3 ) );
    m_norm_buff->unmap();

    m_phase_num++;
    if( m_phase_num >= ANIM_STEPS ) m_phase_num = 0;

    // Mark the accel structure and geometry as dirty so they will be rebuilt.
    m_geom_group->getAcceleration()->markDirty();
  }

  if (m_swim_deterministically)
    updatePosCircle( TargetBox );
  else
    updatePos( TargetBox );

  // Transform the fish's position and orientation
  float3 forward = normalize( m_vel );
  float3 side = make_float3( forward.z, 0, -forward.x );
  side = normalize( side );
  float3 up = cross( forward, side );

  Matrix<4,4> Rotate;
  Rotate.setCol( 0u, make_float4( side, 0 ) );
  Rotate.setCol( 1u, make_float4( up, 0 ) );
  Rotate.setCol( 2u, make_float4( forward, 0 ) );
  Rotate.setCol( 3u, make_float4( 0,0,0,1 ) );

  Matrix<4,4> Translate = Matrix<4,4>::translate( m_pos );
  Matrix<4,4> Comp = Translate * Rotate;

  m_tran->setMatrix( false, Comp.getData(), 0 );
}

void Fish_t::updatePosCircle( optix::Aabb TargetBox )
{
  const float rot_vel = 0.08f; // radians per frame
  Matrix4x4 Rot = Matrix4x4::rotate( rot_vel, make_float3(0,1,0) );

  float3 old_pos = m_pos;
  m_pos = make_float3( Rot * make_float4( m_pos,1 ) );
  m_vel = normalize( m_pos - old_pos );
}

void Fish_t::updatePos( optix::Aabb TargetBox )
{
  // Update position
#if 0
  m_pos = make_float3( 0,0,0 );
  m_vel = make_float3( 1,0,0 );
  m_target_pos = m_pos + m_vel * 300.0f;
  return;
#endif
  m_pos += m_vel;

  // Update velocity
  float range = length( m_target_pos - m_pos );
  float3 TargetDir = normalize( m_target_pos - m_pos );
  m_vel = normalize( m_vel );
  float oldYVel = m_vel.y;

  float ang = acos( dot( m_vel, TargetDir ) );
  if( ang > 1e-5 ) {
    float3 Axis = normalize( cross( m_vel, TargetDir ) );
    const float max_ang = 0.08f;
    ang = fminf( ang, max_ang );
    Matrix4x4 Rot = Matrix4x4::rotate( ang, Axis );
    m_vel = make_float3( Rot * make_float4( m_vel,1 ) );
  }

  // Prevent them from pitching too quickly
  const float max_dpitch = 0.01f;
  float new_pitch = 0;
  if( m_vel.y > oldYVel+max_dpitch ) new_pitch = oldYVel+max_dpitch;
  if( m_vel.y < oldYVel-max_dpitch ) new_pitch = oldYVel-max_dpitch;

  if( new_pitch ) {
    m_vel.y = 0;
    m_vel = normalize( m_vel );
    m_vel.y = new_pitch;
    m_vel = normalize( m_vel );
  }

  m_vel *= m_species->speed * m_species->lengthMeters;

  // Update target
  // meters
  if( range > 1.0f && m_frames_this_target++ < 40 )
    return;

  // Choose new random target
  m_target_pos = MakeDRand( TargetBox.m_min, TargetBox.m_max );
  m_frames_this_target = 0;
}

float FRand()
{
  return float( rand() ) / ( float( RAND_MAX ) );
}

float FRand(const float high)
{
  return FRand() * high;
}

float FRand(const float low, const float high)
{
  return low + FRand() * ( high - low );
}

float3 MakeDRand(const float3 low, const float3 high)
{
  return make_float3( FRand( low.x, high.x ), FRand( low.y, high.y ), FRand( low.z, high.z ) );
}
