// Standalone D3D12 repro for Intel IGC subgroup matrix multiply bug.
// Runs multiple (TileDim, WorkgroupSize) combinations sequentially to reproduce
// the failure that only appears when tests run together (not in isolation).

#ifndef UNICODE
#define UNICODE
#endif

#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <windows.h>
#include <wrl/client.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

extern "C" {
    __declspec(dllexport) extern const UINT D3D12SDKVersion = 721;
    __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}

#define CHECK_HR(hr, msg) do { if (FAILED(hr)) { printf("FAILED: %s (0x%08X)\n", msg, (unsigned)(hr)); return 1; } } while(0)

// Hardware config: 8x16x16 f16 -> f16
static constexpr uint32_t M = 8, N = 16, K = 16;

struct TestParams { uint32_t tileDim, workgroupSize; };
static const TestParams kTestCases[] = {
    {32, 128}
};

static uint16_t F32ToF16(float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FFu;
    if ((bits & 0x7FFFFFFFu) == 0) return sign;
    if (exp <= 0) return sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float F16ToF32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if      (exp == 0)  bits = sign | (mant << 13);
    else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
    else                bits = sign | (uint32_t)((exp - 15 + 127) << 23) | (mant << 13);
    float r; memcpy(&r, &bits, 4);
    return r;
}

static const float kValsF[] = { 0.0f, 1.0f, 2.0f };

static void FillMatrix(uint16_t* data, uint32_t cols, uint32_t rows, uint32_t offset) {
    for (uint32_t i = 0; i < rows * cols; i++)
        data[i] = F32ToF16(kValsF[(offset + i) % 3]);
}

static void ReferenceMatMul(uint16_t* out, const uint16_t* lhs, const uint16_t* rhs,
    uint32_t rows, uint32_t cols, uint32_t k) {
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < k; i++)
                acc += F16ToF32(lhs[r * k + i]) * F16ToF32(rhs[i * cols + c]);
            out[r * cols + c] = F32ToF16(acc);
        }
}

static std::string GenerateShader(uint32_t tileDim, uint32_t workgroupSize) {
    uint32_t matRows = M * tileDim, matCols = N * tileDim, matK = K * tileDim;
    uint32_t inputElems  = matRows * matK + matK * matCols;
    uint32_t outputElems = matRows * matCols;
    uint32_t lhsElems    = matRows * matK;
    uint32_t strideLhs   = matK   * 2;
    uint32_t strideRhs   = matCols * 2;
    uint32_t strideOut   = matCols * 2;

    std::ostringstream s;

    s << "#include <dx/linalg.h>\n"
         "using namespace dx::linalg;\n\n"
      << "typedef Matrix<ComponentType::F16, " << M << ", " << N << ", MatrixUse::Accumulator, MatrixScope::Wave> AccTy;\n"
      << "typedef Matrix<ComponentType::F16, " << M << ", " << K << ", MatrixUse::A,           MatrixScope::Wave> LhsTy;\n"
      << "typedef Matrix<ComponentType::F16, " << K << ", " << N << ", MatrixUse::B,           MatrixScope::Wave> RhsTy;\n\n"
         "ByteAddressBuffer   input  : register(t0);\n"
         "RWByteAddressBuffer output : register(u1);\n\n"
         "void compute(uint sgid, uint num_sg) {\n";

    // acc array init
    s << "    AccTy zero = AccTy::Splat(float16_t(0.0h));\n"
      << "    AccTy acc_row[" << tileDim << "] = {";
    for (uint32_t i = 0; i < tileDim; i++) { if (i) s << ", "; s << "zero"; }
    s << "};\n"
      << "    AccTy acc[" << tileDim << "][" << tileDim << "] = {";
    for (uint32_t i = 0; i < tileDim; i++) { if (i) s << ", "; s << "acc_row"; }
    s << "};\n\n";

    // compute loops
    s << "    for (uint kk = 0u; kk < " << matK << "u; kk += " << K << "u) {\n"
      << "        for (uint tr = sgid; tr < " << tileDim << "u; tr += num_sg) {\n"
      << "            for (uint tc = 0u; tc < " << tileDim << "u; tc++) {\n"
      << "                uint lhs_off = kk + tr * " << (M * matK) << "u;\n"
         "                LhsTy lhs = LhsTy::Splat(float16_t(0.0h));\n"
      << "                if (lhs_off + " << ((M - 1) * matK + K) << "u <= " << inputElems << "u)\n"
      << "                    lhs = LhsTy::Load(input, lhs_off * 2u, " << strideLhs << "u, MatrixLayout::RowMajor);\n"
      << "                uint rhs_off = tc * " << N << "u + kk * " << matCols << "u + " << lhsElems << "u;\n"
         "                RhsTy rhs = RhsTy::Splat(float16_t(0.0h));\n"
      << "                if (rhs_off + " << ((K - 1) * matCols + N) << "u <= " << inputElems << "u)\n"
      << "                    rhs = RhsTy::Load(input, rhs_off * 2u, " << strideRhs << "u, MatrixLayout::RowMajor);\n"
         "                acc[tr][tc].MultiplyAccumulate(lhs, rhs);\n"
         "            }\n"
         "        }\n"
         "    }\n\n";

    // store loops
    s << "    for (uint tr = sgid; tr < " << tileDim << "u; tr += num_sg) {\n"
      << "        for (uint tc = 0u; tc < " << tileDim << "u; tc++) {\n"
      << "            uint out_off = tc * " << N << "u + tr * " << (M * matCols) << "u;\n"
      << "            if (out_off + " << ((M - 1) * matCols + N) << "u <= " << outputElems << "u)\n"
      << "                acc[tr][tc].Store(output, out_off * 2u, " << strideOut << "u, MatrixLayout::RowMajor);\n"
         "        }\n"
         "    }\n"
         "}\n\n"
         "struct EntryInput { uint gtid : SV_GroupIndex; };\n\n"
      << "[numthreads(" << workgroupSize << ", 1, 1)]\n"
         "void main_cs(EntryInput ei) {\n"
         "    uint waveSize = WaveGetLaneCount();\n"
      << "    compute(ei.gtid / waveSize, (" << workgroupSize << "u + waveSize - 1u) / waveSize);\n"
         "}\n";

    return s.str();
}

// DXC globals
static ComPtr<IDxcCompiler3> g_compiler;
static ComPtr<IDxcUtils> g_utils;
static ComPtr<IDxcIncludeHandler> g_includeHandler;
static std::wstring g_exeDir;

static bool InitDXC() {
    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&g_compiler)))) return false;
    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&g_utils)))) return false;
    if (FAILED(g_utils->CreateDefaultIncludeHandler(&g_includeHandler))) return false;
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    g_exeDir = std::wstring(path);
    g_exeDir = g_exeDir.substr(0, g_exeDir.find_last_of(L"\\/"));
    return true;
}

static ComPtr<IDxcBlob> CompileShader(const std::string& src) {
    ComPtr<IDxcBlobEncoding> blob;
    g_utils->CreateBlob(src.c_str(), (UINT)src.size(), CP_UTF8, &blob);
    DxcBuffer buf{ blob->GetBufferPointer(), blob->GetBufferSize(), CP_UTF8 };
    const wchar_t* args[] = { L"-T", L"cs_6_10", L"-E", L"main_cs", L"-HV", L"2021",
                             L"-enable-16bit-types", L"-I", g_exeDir.c_str() };
    ComPtr<IDxcResult> result;
    g_compiler->Compile(&buf, args, _countof(args), g_includeHandler.Get(), IID_PPV_ARGS(&result));
    HRESULT status; result->GetStatus(&status);
    if (FAILED(status)) {
        ComPtr<IDxcBlobUtf8> err;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&err), nullptr);
        if (err) printf("Compile error:\n%s\n", err->GetStringPointer());
        return nullptr;
    }
    ComPtr<IDxcBlob> out;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&out), nullptr);
    return out;
}

static ComPtr<ID3D12Resource> CreateBuffer(ComPtr<ID3D12Device>& dev, uint32_t size,
    D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    ComPtr<ID3D12Resource> buf;
    D3D12_HEAP_PROPERTIES hp{ heap };
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size; desc.Height = 1; desc.DepthOrArraySize = 1;
    desc.MipLevels = 1; desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; desc.Flags = flags;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&buf));
    return buf;
}

int main() {
    printf("=== Subgroup Matrix Repro: %ux%ux%u f16->f16 ===\n\n", M, N, K);

    // Setup D3D12
    { ComPtr<ID3D12Debug> dbg; if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer(); }
    { UUID f[] = { D3D12ExperimentalShaderModels }; D3D12EnableExperimentalFeatures(1, f, nullptr, nullptr); }

    ComPtr<IDXGIFactory6> factory;
    CHECK_HR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

    ComPtr<IDXGIAdapter1> intelAdapter;
    for (UINT i = 0; ; i++) {
        ComPtr<IDXGIAdapter1> a;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
        if (d.VendorId == 0x8086) { intelAdapter = a; printf("GPU: %ls (VendorId=0x%04X DeviceId=0x%04X)\n", d.Description, d.VendorId, d.DeviceId); break; }
    }
    if (!intelAdapter) { printf("No Intel GPU found\n"); return 1; }

    ComPtr<ID3D12Device> device;
    CHECK_HR(D3D12CreateDevice(intelAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "CreateDevice");

    D3D12_FEATURE_DATA_D3D12_OPTIONS1 opts{};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &opts, sizeof(opts));
    uint32_t subgroupMax = opts.WaveLaneCountMax;
    printf("Wave lane max: %u\n\n", subgroupMax);

    if (!InitDXC()) { printf("DXC init failed\n"); return 1; }

    // Shared objects
    ComPtr<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{ D3D12_COMMAND_LIST_TYPE_COMPUTE };
    CHECK_HR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateQueue");

    ComPtr<ID3D12Fence> fence;
    CHECK_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    HANDLE evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    uint64_t fv = 0;
    auto WaitGPU = [&]() { queue->Signal(fence.Get(), ++fv); fence->SetEventOnCompletion(fv, evt); WaitForSingleObject(evt, INFINITE); };

    // Root signature: t0 (SRV), u1 (UAV)
    D3D12_DESCRIPTOR_RANGE1 ranges[2]{};
    ranges[0] = { D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC, 0 };
    ranges[1] = { D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE, 0 };
    D3D12_ROOT_PARAMETER1 rp[2]{};
    rp[0] = { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, {}, D3D12_SHADER_VISIBILITY_ALL };
    rp[0].DescriptorTable = { 1, &ranges[0] };
    rp[1] = { D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, {}, D3D12_SHADER_VISIBILITY_ALL };
    rp[1].DescriptorTable = { 1, &ranges[1] };
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC rsd{}; rsd.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    rsd.Desc_1_1 = { 2, rp };
    ComPtr<ID3DBlob> rsBlob, rsErr;
    CHECK_HR(D3D12SerializeVersionedRootSignature(&rsd, &rsBlob, &rsErr), "SerializeRootSig");
    ComPtr<ID3D12RootSignature> rootSig;
    CHECK_HR(device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig)), "CreateRootSig");

    // Run tests
    uint32_t pass = 0, fail = 0, skip = 0;
    for (const auto& tc : kTestCases) {
        uint32_t tileDim = tc.tileDim, wgSize = tc.workgroupSize;
        uint32_t matRows = M * tileDim, matCols = N * tileDim, matK = K * tileDim;
        printf("[TileDim=%2u Workgroup=%4u] %ux%u * %ux%u -> %ux%u ... ",
            tileDim, wgSize, matRows, matK, matK, matCols, matRows, matCols);
        fflush(stdout);

        if (wgSize < subgroupMax) { printf("SKIP\n"); skip++; continue; }

        auto blob = CompileShader(GenerateShader(tileDim, wgSize));
        if (!blob) { printf("SKIP (compile)\n"); skip++; continue; }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = rootSig.Get();
        pd.CS = { blob->GetBufferPointer(), blob->GetBufferSize() };
        ComPtr<ID3D12PipelineState> pso;
        if (FAILED(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)))) { printf("SKIP (PSO)\n"); skip++; continue; }

        // Data
        uint32_t inSize = (matRows * matK + matK * matCols) * 2;
        uint32_t outElems = matRows * matCols, outSize = outElems * 2;
        std::vector<uint16_t> lhs(matRows * matK), rhs(matK * matCols);
        FillMatrix(lhs.data(), matK, matRows, 0);
        FillMatrix(rhs.data(), matCols, matK, 1);

        // Buffers
        auto inBuf = CreateBuffer(device, inSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
        auto outBuf = CreateBuffer(device, outSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        auto upload = CreateBuffer(device, inSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback = CreateBuffer(device, outSize, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        { void* p; upload->Map(0, nullptr, &p); uint32_t lhsBytes = (uint32_t)(lhs.size() * sizeof(uint16_t)); memcpy(p, lhs.data(), lhsBytes); memcpy((uint8_t*)p + lhsBytes, rhs.data(), rhs.size() * sizeof(uint16_t)); upload->Unmap(0, nullptr); }

        // Descriptors
        D3D12_DESCRIPTOR_HEAP_DESC hd{ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE };
        ComPtr<ID3D12DescriptorHeap> heap; device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
        UINT dsize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();

        D3D12_SHADER_RESOURCE_VIEW_DESC sv{}; sv.Format = DXGI_FORMAT_R32_TYPELESS;
        sv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sv.Buffer = { 0, (inSize + 3) / 4, 0, D3D12_BUFFER_SRV_FLAG_RAW };
        device->CreateShaderResourceView(inBuf.Get(), &sv, cpu);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uv{}; uv.Format = DXGI_FORMAT_R32_TYPELESS;
        uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; uv.Buffer = { 0, outSize / 4, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW };
        auto cpu2 = cpu; cpu2.ptr += dsize;
        device->CreateUnorderedAccessView(outBuf.Get(), nullptr, &uv, cpu2);

        // Execute
        ComPtr<ID3D12CommandAllocator> alloc;
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        ComPtr<ID3D12GraphicsCommandList> cmd;
        device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc.Get(), nullptr, IID_PPV_ARGS(&cmd));

        cmd->CopyBufferRegion(inBuf.Get(), 0, upload.Get(), 0, inSize);
        D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = { inBuf.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE };
        cmd->ResourceBarrier(1, &b);

        cmd->SetComputeRootSignature(rootSig.Get());
        ID3D12DescriptorHeap* hs[] = { heap.Get() }; cmd->SetDescriptorHeaps(1, hs);
        cmd->SetComputeRootDescriptorTable(0, gpu);
        auto gpu2 = gpu; gpu2.ptr += dsize;
        cmd->SetComputeRootDescriptorTable(1, gpu2);
        cmd->SetPipelineState(pso.Get());
        cmd->Dispatch(1, 1, 1);

        D3D12_RESOURCE_BARRIER b2{}; b2.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; b2.UAV.pResource = outBuf.Get();
        cmd->ResourceBarrier(1, &b2);
        D3D12_RESOURCE_BARRIER b3{}; b3.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b3.Transition = { outBuf.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE };
        cmd->ResourceBarrier(1, &b3);
        cmd->CopyBufferRegion(readback.Get(), 0, outBuf.Get(), 0, outSize);
        cmd->Close();

        ID3D12CommandList* ls[] = { cmd.Get() };
        queue->ExecuteCommandLists(1, ls);
        WaitGPU();

        // Verify
        void* mapped; D3D12_RANGE rr{ 0, outSize };
        readback->Map(0, &rr, &mapped);
        auto* gpu_out = (const uint16_t*)mapped;

        std::vector<uint16_t> ref(outElems);
        ReferenceMatMul(ref.data(), lhs.data(), rhs.data(), matRows, matCols, matK);

        uint32_t diffs = 0, firstIdx = 0;
        for (uint32_t i = 0; i < outElems; i++) {
            if (gpu_out[i] != ref[i]) { if (!diffs) firstIdx = i; diffs++; }
        }

        if (!diffs) { printf("PASS\n"); pass++; }
        else {
            printf("FAIL (%u diffs, first[%u] r=%u c=%u: exp %.4g got %.4g)\n",
                diffs, firstIdx, firstIdx / matCols, firstIdx % matCols, F16ToF32(ref[firstIdx]), F16ToF32(gpu_out[firstIdx]));
            fail++;
        }
        D3D12_RANGE empty{ 0, 0 }; readback->Unmap(0, &empty);
    }

    CloseHandle(evt);
    printf("\nResult: %u pass, %u fail, %u skip\n", pass, fail, skip);
    if (fail) printf("BUG REPRODUCED: failures only when dispatches run sequentially.\n");
    return fail ? 1 : 0;
}
