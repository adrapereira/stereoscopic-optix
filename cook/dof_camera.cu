
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#include "helpers.h"

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

rtDeclareVariable(float3,        eye, , ) = { 1.0f, 0.0f, 0.0f };
rtDeclareVariable(float3,        U, , )   = { 0.0f, 1.0f, 0.0f };
rtDeclareVariable(float3,        V, , )   = { 0.0f, 0.0f, 1.0f };
rtDeclareVariable(float3,        W, , )   = { -1.0f, 0.0f, 0.0f };

rtDeclareVariable(float3,		 posA, , );
rtDeclareVariable(float3,		 lookA, , );
rtDeclareVariable(float3,		 posB, , );
rtDeclareVariable(float3,		 lookB, , );
rtDeclareVariable(int,			 anaglyphic, , );
rtDeclareVariable(int,			 stereo, , );

rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , ) = 0.1f;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtBuffer<float3, 2>              output_buffer_f3;
rtBuffer<float4, 2>              output_buffer_f4;
rtDeclareVariable(int,  output_format, , ) = RT_FORMAT_FLOAT4;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

rtDeclareVariable(float, aperture_radius, , );
rtDeclareVariable(float, focal_scale, , );

rtDeclareVariable(unsigned int, frame_number, , );
rtDeclareVariable(float4, jitter, , );

__device__ __forceinline__ void write_output( float3 c )
{
  if ( output_format == RT_FORMAT_FLOAT4 ) {
    output_buffer_f4[launch_index] = make_float4(c, 1.f);
  }
  else {
    output_buffer_f3[launch_index] = c;
  }
}

__device__ __forceinline__ float3 read_output()
{
  if ( output_format == RT_FORMAT_FLOAT4 ) {
    return make_float3( output_buffer_f4[launch_index] );
  }
  else {
    return output_buffer_f3[launch_index];
  }
}

static __device__ __inline__ float3 trace_one(float3 pos, float3 look)
{
  size_t2 screen = output_format == RT_FORMAT_FLOAT4 ? output_buffer_f4.size() : output_buffer_f3.size();

  // pixel sampling
  float2 pixel_sample = make_float2(launch_index) + make_float2(jitter.x, jitter.y);
  float2 d = pixel_sample / make_float2(screen) * 2.f - 1.f;

  // Calculate ray-viewplane intersection point
  float3 ray_origin = pos;
  float3 ray_direction = d.x*U + d.y*V + look;
  float3 ray_target = ray_origin + focal_scale * ray_direction;

  // lens sampling
  float2 sample = optix::square_to_disk(make_float2(jitter.z, jitter.w));
  ray_origin = ray_origin + aperture_radius * ( sample.x * normalize( U ) +  sample.y * normalize( V ) );
  ray_direction = normalize(ray_target - ray_origin);

  // shoot ray
  optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

  PerRayData_radiance prd;
  prd.importance = 1.f;
  prd.depth = 0;

  rtTrace(top_object, ray, prd);

  return prd.result;
}

RT_PROGRAM void dof_camera(){
	float3 new_color;
	if (anaglyphic){
		float3 olhoEsq = trace_one(posB, lookB);
		float3 olhoDir = trace_one(posA, lookA);

		float3 cyan, red;
		cyan.x = 0; cyan.y = 1; cyan.z = 1;
		red.x = 1; red.y = 0; red.z = 0;
		olhoEsq = olhoEsq * red;
		olhoDir = olhoDir * cyan;

		new_color.x = (olhoEsq.x + olhoDir.x);
		new_color.y = (olhoEsq.y + olhoDir.y);
		new_color.z = (olhoEsq.z + olhoDir.z);
	}
	else{
		new_color = trace_one(eye, W);
	}
	if (frame_number > 1 && !stereo)
	{
		float a = 1.0f / (float)frame_number;
		float b = ((float)frame_number - 1.0f) * a;
		const float3 old_color = read_output();
		write_output(a * new_color + b * old_color);
	}else write_output(new_color);
}

RT_PROGRAM void exception()
{
  write_output(bad_color);
}
