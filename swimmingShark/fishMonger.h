#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

float FRand(); // A random number on 0.0 to 1.0.
float FRand( const float high ); // A random number on 0.0 to high.
float FRand( const float low, const float high ); // A random number on low to high.
optix::float3 MakeDRand( const optix::float3 low, const optix::float3 high );

struct Species_t
{
  std::string name;   // String name of OBJ file
  float sizeInObj;    // Size in OBJ, for scaling to max( abs( z ) )==1.0.
  float lengthMeters; // Length in meters
  float centerOffset; // Amount to offset center in -1..1 space
  float maxAmplRad;   // Max angle the head and tail can swing
  float headWaveLen;  // Number of degrees of sine wave from nose to origin
  float tailWaveLen;  // Number of degrees of sine wave from origin to tail
  float speed;        // Speed as a percentage of size
};

class FishMonger_t
{
public:
  const static Species_t *FindSpecies( std::string &name );
  const static int num_species;
  const static Species_t SpeciesInfo[];
};

//------------------------------------------------------------------------------
//
// Fish_t definition
//
//------------------------------------------------------------------------------

class Fish_t
{
public:
  Fish_t( std::string &objfilename, optix::Material matl, optix::TextureSampler projectedTexSamp, optix::Program intersectProgram,
          optix::Context context, optix::GeometryGroup inGG );

  void initAnimation( optix::Aabb TargetBox, float fishFrac, bool deterministic );
  void updateGeometry( optix::Aabb TargetBox );

  optix::float3 target_pos() const { return m_target_pos; }
  void target_pos(optix::float3 val) { m_target_pos = val; }
  bool owns_geom_group() const { return m_this_owns_geom_group; }

  optix::Transform     m_tran;
  optix::GeometryGroup m_geom_group;

private:
  void swimVertex( const float phaseDeg, optix::float3& P, optix::float3& N );

  void swimPoint(optix::float3 &P, const float phaseDeg);

  void updatePos( optix::Aabb TargetBox );
  void updatePosCircle( optix::Aabb TargetBox );

  std::vector<std::vector<optix::float3> > m_animated_points;
  std::vector<std::vector<optix::float3> > m_animated_normals;

  optix::Geometry      m_geom;
  optix::Buffer        m_vert_buff;
  optix::Buffer        m_norm_buff;
  optix::Buffer        m_vindices_buff;
  optix::Buffer        m_nindices_buff;
  optix::Aabb          m_aabb;
  optix::float3        m_pos, m_vel, m_target_pos;
  int                  m_phase_num;
  int                  m_frames_this_target; // Frames since m_target_pos was chosen
  RTsize               m_num_verts;
  RTsize               m_num_norms;
  RTsize               m_num_tris;
  bool                 m_this_owns_geom_group; // True if this Fish is responsible for updating the shared GG for this species each frame
  bool                 m_swim_deterministically;
  const Species_t *    m_species;

  const static int ANIM_STEPS = 31;
};

