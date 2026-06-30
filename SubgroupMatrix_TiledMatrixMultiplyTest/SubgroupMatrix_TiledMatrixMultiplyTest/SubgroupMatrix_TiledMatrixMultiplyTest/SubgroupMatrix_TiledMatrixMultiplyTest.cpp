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

// Hardware config: 8x16x32 u8 -> i32
static constexpr uint32_t M = 8, N = 16, K = 32;

struct TestParams { uint32_t tileDim, workgroupSize; };
static const TestParams kTestCases[] = {
    {16, 128}, {16, 256}, {16, 512}, {16, 1024},
};

static constexpr uint8_t kValues[] = { 0, 1, 2, 3, 11, 23, 37, 71, 101 };

static void FillMatrix(uint8_t* data, uint32_t cols, uint32_t rows, uint32_t offset) {
    for (uint32_t i = 0; i < rows * cols; i++)
        data[i] = kValues[(offset + i) % 9];
}

static void ReferenceMatMul(int32_t* out, const uint8_t* lhs, const uint8_t* rhs,
    uint32_t rows, uint32_t cols, uint32_t k) {
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < cols; c++) {
            int32_t acc = 0;
            for (uint32_t i = 0; i < k; i++)
                acc += (int32_t)lhs[r * k + i] * (int32_t)rhs[i * cols + c];
            out[r * cols + c] = acc;
        }
}

static std::string GenerateShader(uint32_t tileDim, uint32_t workgroupSize) {
    uint32_t matRows = M * tileDim, matCols = N * tileDim, matK = K * tileDim;
    uint32_t inputElems = matRows * matK + matK * matCols;
    uint32_t outputElems = matRows * matCols;

    std::ostringstream s;
    s << R"(#include <dx/linalg.h>
using namespace dx::linalg;

typedef Matrix<ComponentType::I32, )" << M << ", " << N << R"(, MatrixUse::Accumulator, MatrixScope::Wave> AccTy;
typedef Matrix<ComponentType::U8,  )" << M << ", " << K << R"(, MatrixUse::A, MatrixScope::Wave> LhsTy;
typedef Matrix<ComponentType::U8,  )" << K << ", " << N << R"(, MatrixUse::B, MatrixScope::Wave> RhsTy;

ByteAddressBuffer input : register(t0);
RWByteAddressBuffer output : register(u1);

AccTy mul_acc(LhsTy lhs, RhsTy rhs, AccTy acc) {
    acc.MultiplyAccumulate(lhs, rhs);
    return acc;
}

void compute(uint sgid, uint num_sg) {
    AccTy zero = AccTy::Splat(int(0));
)";

    // Declare acc tile arrays (mimics Tint's 1D-then-2D init pattern)
    s << "    AccTy row_init[" << tileDim << "] = {";
    for (uint32_t i = 0; i < tileDim; i++) { if (i) s << ", "; s << "zero"; }
    s << "};\n";
    s << "    AccTy acc[" << tileDim << "][" << tileDim << "] = {";
    for (uint32_t i = 0; i < tileDim; i++) { if (i) s << ", "; s << "row_init"; }
    s << "};\n";

    // Compute loop (Tint-style while with overflow counter)
    s << R"(
    uint2 outer_ctr = (4294967295u).xx;
    uint kk = 0u;
    while (true) {
        if (all((outer_ctr == (0u).xx))) break;
        if (kk >= )" << matK << R"(u) break;
        {
            uint2 mid_ctr = (4294967295u).xx;
            uint tr = sgid;
            while (true) {
                if (all((mid_ctr == (0u).xx))) break;
                if (tr >= )" << tileDim << R"(u) break;
                {
                    uint tc = 0u;
                    while (true) {
                        if (tc >= )" << tileDim << R"(u) break;
                        uint lhs_off = (kk + ((tr * )" << M << "u) * " << matK << R"(u));
                        LhsTy lhs = LhsTy::Splat(0u);
                        if (((lhs_off + ()" << matK << "u * " << (M - 1) << "u)) + " << K << "u) <= " << inputElems << R"(u)
                            lhs = LhsTy::Load(input, lhs_off, )" << matK << R"(u, MatrixLayout::RowMajor);
                        uint rhs_off = ((tc * )" << N << "u) + (kk * " << matCols << "u)) + " << (matRows * matK) << R"(u;
                        RhsTy rhs = RhsTy::Splat(0u);
                        if (((rhs_off + ()" << matCols << "u * " << (K - 1) << "u)) + " << N << "u) <= " << inputElems << R"(u)
                            rhs = RhsTy::Load(input, rhs_off, )" << matCols << R"(u, MatrixLayout::RowMajor);
                        acc[min(tr, )" << (tileDim - 1) << R"(u)][tc] = mul_acc(lhs, rhs, acc[min(tr, )" << (tileDim - 1) << R"(u)][tc]);
                        tc = (tc + 1u);
                    }
                }
                { uint t = (mid_ctr.x - 1u); mid_ctr.x = t; mid_ctr.y = (mid_ctr.y - uint((t == 4294967295u))); tr = (tr + num_sg); }
            }
        }
        { uint t = (outer_ctr.x - 1u); outer_ctr.x = t; outer_ctr.y = (outer_ctr.y - uint((t == 4294967295u))); kk = (kk + )" << K << R"(u); }
    }
)";

    // Store loop
    s << R"(
    uint2 store_ctr = (4294967295u).xx;
    uint str = sgid;
    while (true) {
        if (all((store_ctr == (0u).xx))) break;
        if (str >= )" << tileDim << R"(u) break;
        {
            uint stc = 0u;
            while (true) {
                if (stc >= )" << tileDim << R"(u) break;
                uint out_off = ((stc * )" << N << "u) + ((str * " << M << "u) * " << matCols << R"(u));
                AccTy tile = acc[min(str, )" << (tileDim - 1) << R"(u)][stc];
                if (((out_off + ()" << matCols << "u * " << (M - 1) << "u)) + " << N << "u) <= " << outputElems << R"(u)
                    tile.Store(output, (out_off * 4u), )" << (matCols * 4) << R"(u, MatrixLayout::RowMajor);
                stc = (stc + 1u);
            }
        }
        { uint t = (store_ctr.x - 1u); store_ctr.x = t; store_ctr.y = (store_ctr.y - uint((t == 4294967295u))); str = (str + num_sg); }
    }
}

struct EntryInput { uint gtid : SV_GroupIndex; };

[numthreads()" << workgroupSize << R"(, 1, 1)]
void dawn_entry(EntryInput ei) {
    uint waveSize = WaveGetLaneCount();
    compute((ei.gtid / waveSize), (()" << (workgroupSize - 1) << R"(u + waveSize) / waveSize));
}
)";
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
    const wchar_t* args[] = { L"-T", L"cs_6_10", L"-E", L"dawn_entry", L"-HV", L"2021",
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
    printf("=== Subgroup Matrix Repro: %ux%ux%u u8->i32 ===\n\n", M, N, K);

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
        if (d.VendorId == 0x8086) { intelAdapter = a; printf("GPU: %ls\n", d.Description); break; }
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
        uint32_t inSize = matRows * matK + matK * matCols;
        uint32_t outElems = matRows * matCols, outSize = outElems * 4;
        std::vector<uint8_t> lhs(matRows * matK), rhs(matK * matCols);
        FillMatrix(lhs.data(), matK, matRows, 0);
        FillMatrix(rhs.data(), matCols, matK, 1);

        // Buffers
        auto inBuf = CreateBuffer(device, inSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
        auto outBuf = CreateBuffer(device, outSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        auto upload = CreateBuffer(device, inSize, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback = CreateBuffer(device, outSize, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        { void* p; upload->Map(0, nullptr, &p); memcpy(p, lhs.data(), lhs.size()); memcpy((uint8_t*)p + lhs.size(), rhs.data(), rhs.size()); upload->Unmap(0, nullptr); }

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
        auto* gpu_out = (const int32_t*)mapped;

        std::vector<int32_t> ref(outElems);
        ReferenceMatMul(ref.data(), lhs.data(), rhs.data(), matRows, matCols, matK);

        uint32_t diffs = 0, firstIdx = 0;
        for (uint32_t i = 0; i < outElems; i++) {
            if (gpu_out[i] != ref[i]) { if (!diffs) firstIdx = i; diffs++; }
        }

        if (!diffs) { printf("PASS\n"); pass++; }
        else {
            printf("FAIL (%u diffs, first[%u] r=%u c=%u: exp %d got %d)\n",
                diffs, firstIdx, firstIdx / matCols, firstIdx % matCols, ref[firstIdx], gpu_out[firstIdx]);
            fail++;
        }
        D3D12_RANGE empty{ 0, 0 }; readback->Unmap(0, &empty);
    }

    CloseHandle(evt);
    printf("\nResult: %u pass, %u fail, %u skip\n", pass, fail, skip);
    if (fail) printf("BUG REPRODUCED: failures only when dispatches run sequentially.\n");
    return fail ? 1 : 0;
}
