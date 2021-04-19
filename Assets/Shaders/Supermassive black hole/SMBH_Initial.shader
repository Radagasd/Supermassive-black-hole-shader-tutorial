Shader "KelvinvanHoorn/SMBH"
{
    Properties
    {
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

            float4 frag (v2f IN) : SV_Target
            {
                return float4(1,0,0,1);
            }
            ENDHLSL
        }
    }
}