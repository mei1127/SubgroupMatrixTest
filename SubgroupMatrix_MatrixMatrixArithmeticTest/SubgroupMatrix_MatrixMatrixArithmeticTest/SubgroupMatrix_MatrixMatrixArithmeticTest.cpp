#include <cstdio>
#include <cstring>

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "dxcapi.h"

#include <string>

#ifndef D3D_SHADER_MODEL_6_10
#define D3D_SHADER_MODEL_6_10 ((D3D_SHADER_MODEL)0x6A)
#endif

using Microsoft::WRL::ComPtr;

// This repro is pinned to Agility SDK 1.721.0.
// The project's local d3d12.h and the runtime binaries loaded from .\D3D12\
// (D3D12Core.dll and d3d12SDKLayers.dll, copied under x64\Debug\D3D12\)
// are all taken from the Agility SDK 1.721.0 package.
extern "C" { __declspec(dllexport) extern const UINT D3D12SDKVersion = 721; }
extern "C" { __declspec(dllexport) extern const char* D3D12SDKPath = u8".\\D3D12\\"; }

#define CHECK_HR(hr, msg) \
    do { \
        HRESULT _hr = (hr); \
        if (FAILED(_hr)) { \
            printf("FAILED: %s (HRESULT=0x%08X)\n", msg, (unsigned)_hr); \
            return 1; \
        } \
    } while(0)

// Config: i8 -> i32, 8x32x16, column-major layout.
static const char* kShaderSource = R"hlsl(
#include <dx/linalg.h>
using namespace dx::linalg;
using Matrix_left_i8_8x32 = Matrix<ComponentType::I8, 8, 32, MatrixUse::A, MatrixScope::Wave>;
using Matrix_right_i8_32x16 = Matrix<ComponentType::I8, 32, 16, MatrixUse::B, MatrixScope::Wave>;
using Matrix_result_i32_8x16 = Matrix<ComponentType::I32, 8, 16, MatrixUse::Accumulator, MatrixScope::Wave>;

ByteAddressBuffer v : register(t0);
RWByteAddressBuffer v_1 : register(u1);
[numthreads(32, 1, 1)]
void main() {
    Matrix_left_i8_8x32 v_3 = Matrix_left_i8_8x32::Load(v, 0u, 8u, MatrixLayout::ColMajor);
    Matrix_right_i8_32x16 v_4 = Matrix_right_i8_32x16::Load(v, 256u, 32u, MatrixLayout::ColMajor);
    Matrix_result_i32_8x16 v_5 = Multiply<ComponentType::I32>(v_3, v_4);
    v_5.Store(v_1, 0u, 32u, MatrixLayout::ColMajor);
}
)hlsl";

bool EnableExperimentalShaderModels()
{
    const UUID features[] = { D3D12ExperimentalShaderModels };
    return SUCCEEDED(D3D12EnableExperimentalFeatures(
        _countof(features), features, nullptr, nullptr));
}

int main() {
    printf("D3D12 subgroup-matrix repro\n");

    if (!EnableExperimentalShaderModels())
    {
        printf("FAILED: D3D12EnableExperimentalFeatures\n");
        return 1;
    }

    ComPtr<IDXGIFactory6> factory;
    CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<IDXGIAdapter1> selectedAdapter;
    DXGI_ADAPTER_DESC1 selectedDesc = {};

    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            adapter.Reset();
            continue;
        }
        if (desc.VendorId == 0x8086) {
            selectedAdapter = adapter;
            selectedDesc = desc;
            break;
        }
        if (!selectedAdapter) {
            selectedAdapter = adapter;
            selectedDesc = desc;
        }
        adapter.Reset();
    }

    if (!selectedAdapter) {
        printf("ERROR: No suitable GPU adapter found.\n");
        return 1;
    }

    printf("Adapter: %ls (0x%04X:0x%04X)\n", selectedDesc.Description, selectedDesc.VendorId, selectedDesc.DeviceId);

    ComPtr<ID3D12Device> device;
    CHECK_HR(D3D12CreateDevice(selectedAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)),
        "D3D12CreateDevice");

    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_10 };
    HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel));
    if (FAILED(hr) || shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_10) {
        printf("ERROR: Shader Model 6.10 not supported. Highest=0x%X\n", (unsigned)shaderModel.HighestShaderModel);
        return 1;
    }

    ComPtr<IDxcCompiler3> compiler;
    ComPtr<IDxcUtils> utils;

    CHECK_HR(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)), "DxcCreateInstance(Compiler)");
    CHECK_HR(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils)), "DxcCreateInstance(Utils)");

    ComPtr<IDxcBlobEncoding> sourceBlob;
    CHECK_HR(utils->CreateBlob(kShaderSource, (UINT32)strlen(kShaderSource), CP_UTF8, &sourceBlob),
        "CreateBlob");

    ComPtr<IDxcIncludeHandler> includeHandler;
    CHECK_HR(utils->CreateDefaultIncludeHandler(&includeHandler), "CreateDefaultIncludeHandler");

    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring exeDir(exePath);
    exeDir = exeDir.substr(0, exeDir.find_last_of(L"\\/"));

    const wchar_t* args[] = {
        L"-T", L"cs_6_10",
        L"-E", L"main",
        L"-HV", L"2021",
        L"-Zpr",
        L"-Gis",
        L"-O3",
        L"-opt-disable", L"structurize-loop-exits-for-unroll",
        L"/enable-16bit-types",
        L"-I", exeDir.c_str(),
    };

    DxcBuffer sourceBuffer = {};
    sourceBuffer.Ptr = sourceBlob->GetBufferPointer();
    sourceBuffer.Size = sourceBlob->GetBufferSize();
    sourceBuffer.Encoding = CP_UTF8;

    ComPtr<IDxcResult> result;
    CHECK_HR(compiler->Compile(&sourceBuffer, args, _countof(args), includeHandler.Get(), IID_PPV_ARGS(&result)),
        "DXC Compile");

    HRESULT compileStatus;
    result->GetStatus(&compileStatus);

    ComPtr<IDxcBlobUtf8> errors;
    result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
    if (errors && errors->GetStringLength() > 0) {
        printf("DXC messages:\n%s\n", errors->GetStringPointer());
    }

    if (FAILED(compileStatus)) {
        printf("ERROR: Shader compilation failed.\n");
        return 1;
    }

    ComPtr<IDxcBlob> shaderBlob;
    CHECK_HR(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr),
        "GetOutput(OBJECT)");

    D3D12_DESCRIPTOR_RANGE1 ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;

    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 1;
    ranges[1].RegisterSpace = 0;
    ranges[1].Flags = D3D12_DESCRIPTOR_RANGE_FLAG_NONE;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;

    D3D12_ROOT_PARAMETER1 rootParam = {};
    rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParam.DescriptorTable.NumDescriptorRanges = 2;
    rootParam.DescriptorTable.pDescriptorRanges = ranges;
    rootParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc = {};
    rootSigDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rootSigDesc.Desc_1_1.NumParameters = 1;
    rootSigDesc.Desc_1_1.pParameters = &rootParam;
    rootSigDesc.Desc_1_1.NumStaticSamplers = 0;
    rootSigDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> serializedRootSig;
    ComPtr<ID3DBlob> rootSigError;
    CHECK_HR(D3D12SerializeVersionedRootSignature(&rootSigDesc, &serializedRootSig, &rootSigError),
        "D3D12SerializeVersionedRootSignature");

    ComPtr<ID3D12RootSignature> rootSignature;
    CHECK_HR(device->CreateRootSignature(0, serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(&rootSignature)),
        "CreateRootSignature");

    printf("\nCreating Compute Pipeline State...\n");
    printf(">>> This is where the Intel driver crashes with DXGI_ERROR_DRIVER_INTERNAL_ERROR <<<\n\n");


    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = rootSignature.Get();
    psoDesc.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
    psoDesc.CS.BytecodeLength = shaderBlob->GetBufferSize();

    
    ComPtr<ID3D12PipelineState> pipelineState;
    hr = device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState));

    if (SUCCEEDED(hr)) {
        printf("SUCCESS: CreateComputePipelineState succeeded.\n");
    }
    else {
        printf("FAILED: CreateComputePipelineState HRESULT=0x%08X\n", (unsigned)hr);

        if (hr == DXGI_ERROR_DEVICE_REMOVED) {
            HRESULT reason = device->GetDeviceRemovedReason();
            printf("Device removed reason: 0x%08X", (unsigned)reason);
            if (reason == DXGI_ERROR_DRIVER_INTERNAL_ERROR) {
                printf(" (DXGI_ERROR_DRIVER_INTERNAL_ERROR)");
            }
            else if (reason == DXGI_ERROR_DEVICE_HUNG) {
                printf(" (DXGI_ERROR_DEVICE_HUNG)");
            }
            else if (reason == DXGI_ERROR_DEVICE_RESET) {
                printf(" (DXGI_ERROR_DEVICE_RESET)");
            }
            printf("\n");
        }
    }

    return FAILED(hr) ? 1 : 0;
}
