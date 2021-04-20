Shader "KelvinvanHoorn/SMBH_Finished"
{
    Properties
    {
		_DiscTex ("Disc texture", 2D) = "white" {}
		_DiscWidth ("Width of the accretion disc", float) = 0.1
		_DiscOuterRadius ("Object relative disc outer radius", Range(0,1)) = 1
		_DiscInnerRadius ("Object relative disc inner radius", Range(0,1)) = 0.25
		_DiscSpeed ("Disc rotation speed", float) = 2
		[HDR]_DiscColor ("Disc main color", Color) = (1,0,0,1)
		_DopplerBeamingFactor ("Doppler beaming effect factor", float) = 66
		_HueRadius ("Hue shift start radius", Range(0,1)) = 0.75
		_HueShiftFactor ("Hue shifting factor", float) = -0.03
		_Steps ("Amount of steps", int) = 256
		_StepSize ("Step size", Range(0.001, 1)) = 0.1
		_SSRadius ("Object relative Schwarzschild radius", Range(0,1)) = 0.1
		_GConst ("Gravitational constant", float) = 0.3
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "RenderPipeline" = "UniversalRenderPipeline" "Queue" = "Transparent" }
		Cull Front

        Pass
        {
            HLSLPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
			#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareOpaqueTexture.hlsl"       

            static const float maxFloat = 3.402823466e+38;

			struct Attributes
			{
				float4 posOS	: POSITION;
			};

			struct v2f
			{
				float4 posCS		: SV_POSITION;
				float3 posWS		: TEXCOORD0;

				float3 centre		: TEXCOORD1;
				float3 objectScale	: TEXCOORD2;
			};

            v2f vert(Attributes IN)
			{
				v2f OUT = (v2f)0;

				VertexPositionInputs vertexInput = GetVertexPositionInputs(IN.posOS.xyz);

				OUT.posCS = vertexInput.positionCS;
				OUT.posWS = vertexInput.positionWS;

				// Object information, based upon Unity's shadergraph library functions
				OUT.centre = UNITY_MATRIX_M._m03_m13_m23;
				OUT.objectScale = float3(length(float3(UNITY_MATRIX_M[0].x, UNITY_MATRIX_M[1].x, UNITY_MATRIX_M[2].x)),
                             length(float3(UNITY_MATRIX_M[0].y, UNITY_MATRIX_M[1].y, UNITY_MATRIX_M[2].y)),
                             length(float3(UNITY_MATRIX_M[0].z, UNITY_MATRIX_M[1].z, UNITY_MATRIX_M[2].z)));

				return OUT;
			}

			float _DiscWidth;
			float _DiscOuterRadius;
			float _DiscInnerRadius;

			Texture2D<float4> _DiscTex;
			SamplerState sampler_DiscTex;
			float4 _DiscTex_ST;

			float _DiscSpeed;

			float4 _DiscColor;
			float _DopplerBeamingFactor;
			float _HueRadius;
			float _HueShiftFactor;

			int _Steps;
			float _StepSize;

			float _SSRadius;
			float _GConst;

			// Based upon https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/#:~:text=When%20the%20ray%20and%20sphere,equations%20and%20solving%20for%20t.
			// Returns dstToSphere, dstThroughSphere
			// If inside sphere, dstToSphere will be 0
			// If ray misses sphere, dstToSphere = max float value, dstThroughSphere = 0
			// Given rayDir must be normalized
			float2 intersectSphere(float3 rayOrigin, float3 rayDir, float3 centre, float radius) {

				float3 offset = rayOrigin - centre;
				const float a = 1;
				float b = 2 * dot(offset, rayDir);
				float c = dot(offset, offset) - radius * radius;

				float discriminant = b * b - 4 * a*c;
				// No intersections: discriminant < 0
				// 1 intersection: discriminant == 0
				// 2 intersections: discriminant > 0
				if (discriminant > 0) {
					float s = sqrt(discriminant);
					float dstToSphereNear = max(0, (-b - s) / (2 * a));
					float dstToSphereFar = (-b + s) / (2 * a);

					if (dstToSphereFar >= 0) {
						return float2(dstToSphereNear, dstToSphereFar - dstToSphereNear);
					}
				}
				// Ray did not intersect sphere
				return float2(maxFloat, 0);
			}

			// Based upon https://mrl.cs.nyu.edu/~dzorin/rend05/lecture2.pdf
			float2 intersectInfiniteCylinder(float3 rayOrigin, float3 rayDir, float3 cylinderOrigin, float3 cylinderDir, float cylinderRadius)
			{
				float3 a0 = rayDir - dot(rayDir, cylinderDir) * cylinderDir;
				float a = dot(a0,a0);

				float3 dP = rayOrigin - cylinderOrigin;
				float3 c0 = dP - dot(dP, cylinderDir) * cylinderDir;
				float c = dot(c0,c0) - cylinderRadius * cylinderRadius;

				float b = 2 * dot(a0, c0);

				float discriminant = b * b - 4 * a * c;

				if (discriminant > 0) {
					float s = sqrt(discriminant);
					float dstToNear = max(0, (-b - s) / (2 * a));
					float dstToFar = (-b + s) / (2 * a);

					if (dstToFar >= 0) {
						return float2(dstToNear, dstToFar - dstToNear);
					}
				}
				return float2(maxFloat, 0);
			}

			// Based upon https://mrl.cs.nyu.edu/~dzorin/rend05/lecture2.pdf
			float intersectInfinitePlane(float3 rayOrigin, float3 rayDir, float3 planeOrigin, float3 planeDir)
			{
				float a = 0;
				float b = dot(rayDir, planeDir);
				float c = dot(rayOrigin, planeDir) - dot(planeDir, planeOrigin);

				float discriminant = b * b - 4 * a*c;

				return -c/b;
			}

			// Based upon https://mrl.cs.nyu.edu/~dzorin/rend05/lecture2.pdf
			float intersectDisc(float3 rayOrigin, float3 rayDir, float3 p1, float3 p2, float3 discDir, float discRadius, float innerRadius)
			{
				float discDst = maxFloat;
				float2 cylinderIntersection = intersectInfiniteCylinder(rayOrigin, rayDir, p1, discDir, discRadius);
				float cylinderDst = cylinderIntersection.x;

				if(cylinderDst < maxFloat)
				{
					float finiteC1 = dot(discDir, rayOrigin + rayDir * cylinderDst - p1);
					float finiteC2 = dot(discDir, rayOrigin + rayDir * cylinderDst - p2);

					// Ray intersects with edges of the cylinder/disc
					if(finiteC1 > 0 && finiteC2 < 0 && cylinderDst > 0)
					{
						discDst = cylinderDst;
					}
					else
					{
						float radiusSqr = discRadius * discRadius;
						float innerRadiusSqr = innerRadius * innerRadius;

						float p1Dst = max(intersectInfinitePlane(rayOrigin, rayDir, p1, discDir), 0);
						float3 q1 = rayOrigin + rayDir * p1Dst;
						float p1q1DstSqr = dot(q1 - p1, q1 - p1);

						// Ray intersects with lower plane of cylinder/disc
						if(p1Dst > 0 && p1q1DstSqr < radiusSqr && p1q1DstSqr > innerRadiusSqr)
						{
							if(p1Dst < discDst)
							{
								discDst = p1Dst;
							}
						}
							
						float p2Dst = max(intersectInfinitePlane(rayOrigin, rayDir, p2, discDir), 0);
						float3 q2 = rayOrigin + rayDir * p2Dst;
						float p2q2DstSqr = dot(q2 - p2, q2 - p2);

						// Ray intersects with upper plane of cylinder/disc
						if(p2Dst > 0 && p2q2DstSqr < radiusSqr && p2q2DstSqr > innerRadiusSqr)
						{
							if(p2Dst < discDst)
							{
								discDst = p2Dst;
							}
						}
					}
				}
				
				return discDst;
			}

			float remap(float v, float minOld, float maxOld, float minNew, float maxNew) {
				return minNew + (v - minOld) * (maxNew - minNew) / (maxOld - minOld);
			}

			float2 discUV(float3 planarDiscPos, float3 discDir, float3 centre, float radius)
			{
				float3 planarDiscPosNorm = normalize(planarDiscPos);
				float sampleDist01 = length(planarDiscPos) / radius;

				float3 tangentTestVector = float3(1,0,0);
				if(abs(dot(discDir, tangentTestVector)) >= 1)
					tangentTestVector = float3(0,1,0);

				float3 tangent = normalize(cross(discDir, tangentTestVector));
				float3 biTangent = cross(tangent, discDir);
				float phi = atan2(dot(planarDiscPosNorm, tangent), dot(planarDiscPosNorm, biTangent)) / PI;
				phi = remap(phi, -1, 1, 0, 1);

				// Radial distance
				float u = sampleDist01;
				// Angular distance
				float v = phi;

				return float2(u,v);
			}

			// Based upon UnityCG.cginc, used in hdrIntensity 
			float3 LinearToGammaSpace (float3 linRGB)
			{
				linRGB = max(linRGB, float3(0.f, 0.f, 0.f));
				// An almost-perfect approximation from http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html?m=1
				return max(1.055h * pow(linRGB, 0.416666667h) - 0.055h, 0.h);
			}

			// Based upon UnityCG.cginc, used in hdrIntensity 
			float3 GammaToLinearSpace (float3 sRGB)
			{
				// Approximate version from http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html?m=1
				return sRGB * (sRGB * (sRGB * 0.305306011f + 0.682171111f) + 0.012522878f);
			}

			// Based upon https://forum.unity.com/threads/how-to-change-hdr-colors-intensity-via-shader.531861/
			float3 hdrIntensity(float3 emissiveColor, float intensity)
            {
                // if not using gamma color space, convert from linear to gamma
                #ifndef UNITY_COLORSPACE_GAMMA
                emissiveColor.rgb = LinearToGammaSpace(emissiveColor.rgb);
                #endif
                // apply intensity exposure
                emissiveColor.rgb *= pow(2.0, intensity);
                // if not using gamma color space, convert back to linear
                #ifndef UNITY_COLORSPACE_GAMMA
                emissiveColor.rgb = GammaToLinearSpace(emissiveColor.rgb);
                #endif

                return emissiveColor;
            }

			// Based upon Unity's shadergraph library functions
            float3 RGBToHSV(float3 c)
			{
				float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
				float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
				float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
				float d = q.x - min(q.w, q.y);
				float e = 1.0e-10;
				return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
			}

			// Based upon Unity's shadergraph library functions
			float3 HSVToRGB(float3 c)
			{
				float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
				float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
				return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
			}

			// Based upon Unity's shadergraph library functions
            float3 RotateAboutAxis(float3 In, float3 Axis, float Rotation)
            {
                float s = sin(Rotation);
                float c = cos(Rotation);
                float one_minus_c = 1.0 - c;

                Axis = normalize(Axis);
                float3x3 rot_mat = 
                {   one_minus_c * Axis.x * Axis.x + c, one_minus_c * Axis.x * Axis.y - Axis.z * s, one_minus_c * Axis.z * Axis.x + Axis.y * s,
                    one_minus_c * Axis.x * Axis.y + Axis.z * s, one_minus_c * Axis.y * Axis.y + c, one_minus_c * Axis.y * Axis.z - Axis.x * s,
                    one_minus_c * Axis.z * Axis.x - Axis.y * s, one_minus_c * Axis.y * Axis.z + Axis.x * s, one_minus_c * Axis.z * Axis.z + c
                };
                return mul(rot_mat,  In);
            }

			float3 discColor(float3 baseColor, float3 planarDiscPos, float3 discDir, float3 cameraPos, float u, float radius)
			{
				float3 newColor = baseColor;

				// Distance intensity fall-off
				float intensity = remap(u, 0, 1, 0.5, -1.2);
				intensity *= abs(intensity);

				// Doppler beaming intensity change
				float3 rotatePos = RotateAboutAxis(planarDiscPos, discDir, 0.01);
				float dopplerDistance = (length(rotatePos - cameraPos) - length(planarDiscPos - cameraPos)) / radius;
				intensity += dopplerDistance * _DiscSpeed * _DopplerBeamingFactor;
				
				newColor = hdrIntensity(baseColor, intensity);

				// Distance hue shift
				float3 hueColor = RGBToHSV(newColor);
				float hueShift = saturate(remap(u, _HueRadius, 1, 0, 1));
				hueColor.r += hueShift * _HueShiftFactor;
				newColor = HSVToRGB(hueColor);

				return newColor;
			}

			float4 frag (v2f IN) : SV_Target
			{
				// Initial ray information
				float3 rayOrigin = _WorldSpaceCameraPos;
				float3 rayDir = normalize(IN.posWS - _WorldSpaceCameraPos);

				float sphereRadius = 0.5 * min(min(IN.objectScale.x, IN.objectScale.y), IN.objectScale.z);
				float2 outerSphereIntersection = intersectSphere(rayOrigin, rayDir, IN.centre, sphereRadius);

				// Disc information, direction is objects rotation
				float3 discDir = normalize(mul(unity_ObjectToWorld, float4(0,1,0,0)).xyz);
				float3 p1 = IN.centre - 0.5 * _DiscWidth * discDir;
				float3 p2 = IN.centre + 0.5 * _DiscWidth * discDir;
				float discRadius = sphereRadius * _DiscOuterRadius;
				float innerRadius = sphereRadius * _DiscInnerRadius;

				// Ray information
				float transmittance = 0;
				float blackHoleMask = 0;
				float3 samplePos = float3(maxFloat, 0, 0);
				float3 currentRayPos = rayOrigin + rayDir * outerSphereIntersection.x;
				float3 currentRayDir = rayDir;

				// Ray intersects with the outer sphere
				if(outerSphereIntersection.x < maxFloat)
				{
					for (int i = 0; i < _Steps; i++)
					{
						float3 dirToCentre = IN.centre-currentRayPos;
						float dstToCentre = length(dirToCentre);
						dirToCentre /= dstToCentre;

						if(dstToCentre > sphereRadius + _StepSize)
						{
							break;
						}

						float force = _GConst/(dstToCentre*dstToCentre);
						currentRayDir = normalize(currentRayDir + dirToCentre * force * _StepSize);

						// Move ray forward
						currentRayPos += currentRayDir * _StepSize;

						float blackHoleDistance = intersectSphere(currentRayPos, currentRayDir, IN.centre, _SSRadius * sphereRadius).x;
						if(blackHoleDistance <= _StepSize)
						{
							blackHoleMask = 1;
							break;
						}

						// Check for disc intersection nearby
						float discDst = intersectDisc(currentRayPos, currentRayDir, p1, p2, discDir, discRadius, innerRadius);
						if(transmittance < 1 && discDst < _StepSize)
						{
							transmittance = 1;
							samplePos = currentRayPos + currentRayDir * discDst;
						}
					}
				}

				float2 uv = float2(0,0);
				float3 planarDiscPos = float3(0,0,0);
				if(samplePos.x < maxFloat)
				{
					planarDiscPos = samplePos - dot(samplePos - IN.centre, discDir) * discDir - IN.centre;
					uv = discUV(planarDiscPos, discDir, IN.centre, discRadius);
					uv.y += _Time.x * _DiscSpeed;
				}
				float texCol = _DiscTex.SampleLevel(sampler_DiscTex, uv * _DiscTex_ST.xy, 0).r;

				float2 screenUV = IN.posCS.xy / _ScreenParams.xy;

				// Ray direction projection
				float3 distortedRayDir = normalize(currentRayPos - rayOrigin);
				float4 rayCameraSpace = mul(unity_WorldToCamera, float4(distortedRayDir,0));
				float4 rayUVProjection = mul(unity_CameraProjection, float4(rayCameraSpace));
				float2 distortedScreenUV = rayUVProjection.xy + 1 * 0.5;

				// Screen and object edge transitions
				float edgeFadex = smoothstep(0, 0.25, 1 - abs(remap(screenUV.x, 0, 1, -1, 1)));
				float edgeFadey = smoothstep(0, 0.25, 1 - abs(remap(screenUV.y, 0, 1, -1, 1)));
				float t = saturate(remap(outerSphereIntersection.y, sphereRadius, 2 * sphereRadius, 0, 1)) * edgeFadex * edgeFadey;
				distortedScreenUV = lerp(screenUV, distortedScreenUV, t);

				float3 backgroundCol = SampleSceneColor(distortedScreenUV) * (1 - blackHoleMask);

				float3 discCol = discColor(_DiscColor.rgb, planarDiscPos, discDir, _WorldSpaceCameraPos, uv.x, discRadius);

				transmittance *= texCol * _DiscColor.a;
				float3 col = lerp(backgroundCol, discCol, transmittance);
				return float4(col,1);
			}
            ENDHLSL
        }
    }
}