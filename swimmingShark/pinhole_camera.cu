
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

#include <optix_world.h>
#include "helpers.h"

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth; 
};

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );

rtDeclareVariable(float3, posA, , );
rtDeclareVariable(float3, lookA, , );
rtDeclareVariable(float3, posB, , );
rtDeclareVariable(float3, lookB, , );

rtDeclareVariable(int,			 anaglyphic, , );

rtDeclareVariable(float3,        bad_color, , );
rtDeclareVariable(float,         scene_epsilon, , );
rtBuffer<uchar4, 2>              output_buffer;
rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );
rtDeclareVariable(float, time_view_scale, , ) = 1e-6f;

//#define TIME_VIEW

static __device__ __inline__ float3 trace(float3 pos, float3 look)
{
	float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f;
	float3 ray_origin = pos;
	float3 ray_direction = normalize(d.x*U + d.y*V + look);

	optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX);

	PerRayData_radiance prd;
	prd.importance = 1.f;
	prd.depth = 0;

	rtTrace(top_object, ray, prd);
	return prd.result;
}

RT_PROGRAM void pinhole_camera(){
	if (anaglyphic){
		float3 olhoEsq = trace(posB, lookB);
		float3 olhoDir = trace(posA, lookA);

		float3 cyan, red;
		cyan.x = 0; cyan.y = 1; cyan.z = 1;
		red.x = 1; red.y = 0; red.z = 0;
		olhoEsq = olhoEsq * red;
		olhoDir = olhoDir * cyan;

		float3 new_color;
		new_color.x = (olhoEsq.x + olhoDir.x);
		new_color.y = (olhoEsq.y + olhoDir.y);
		new_color.z = (olhoEsq.z + olhoDir.z);
		output_buffer[launch_index] = make_color(new_color);
	}else{
		float3 res = trace(eye, W);
		output_buffer[launch_index] = make_color(res);
	}
}

RT_PROGRAM void exception()
{
  const unsigned int code = rtGetExceptionCode();
  rtPrintf( "Caught exception 0x%X at launch index (%d,%d)\n", code, launch_index.x, launch_index.y );
  output_buffer[launch_index] = make_color( bad_color );
}
