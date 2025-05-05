#include "viz.hpp"

#undef FATAL
#include "gas/gas.hpp"
#include "gas/gas_ui.hpp"
#include "gas/gas_imgui.hpp"
#include "mpenv_shaders.hpp"

#include "types.hpp"
#include "sim.hpp"
#include "map_importer.hpp"
#include "mgr.hpp"
#include "utils.hpp"
#include <array>

#ifdef DB_SUPPORT
#include "db.hpp"
#endif

#include "trajectory_db.hpp"

#include "viz_shader_common.hpp"

#include <madrona/navmesh.hpp>

#ifndef MADRONA_MW_MODE
#error "Need to link against multi-world madrona lib"
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <random>
#include <stdlib.h>

#ifdef EMSCRIPTEN
#include <emscripten.h>
#endif

static constexpr float PI = 3.14159265358979323846f;

namespace NavUtils
{
	using madrona::math::Vector3;
	using madrona::Navmesh;
	int NearestNavTri(const Navmesh& navmesh, Vector3 pos)
	{
        // Keep track of the nearest while looking for containment.
        float closest = FLT_MAX;
        int closestIdx = -1;
        for (int tri = 0; tri < (int)navmesh.numTris; tri++)
        {
            bool contained = true;
            bool gtz = false;
            for (int i = 0; i < 3; i++)
            {
                Vector3 v1 = navmesh.vertices[navmesh.triIndices[tri * 3 + i]];
                Vector3 v2 = navmesh.vertices[navmesh.triIndices[tri * 3 + ((i + 1) % 3)]];
                Vector3 v3 = v2 - v1;
                Vector3 vp = pos - v1;
                Vector3 c = cross(v3, vp);
                if ((c.z > 0.0f) != gtz && i > 0)
                    contained = false;
                gtz = c.z > 0.0f;
                float distsq = v1.distance2(pos);
                if (distsq < closest)
                {
                    float dir = v3.dot(vp);
                    Vector3 perp = vp * (-dir / v3.dot(v3)) + v3;
                    distsq = perp.dot(perp);
                    if (distsq < closest)
                    {
                        closest = fabs(c.z);
                        closestIdx = tri;
                    }
                }
            }
            if (contained)
                return tri;
        }
		return closestIdx;
	}

    Vector3 CenterOfTri(const Navmesh& navmesh, int tri)
    {
        Vector3 center = { 0.0f, 0.0f, 0.0f };
        for (int i = 0; i < 3; i++)
        {
            center += navmesh.vertices[navmesh.triIndices[tri * 3 + i]] / 3.0f;
        }
        return center;
    }

    bool VisitCell(const Navmesh& navmesh, int cell, int targetCell, std::vector<int> &path, std::vector<bool> &visited)
    {
        if (visited[cell])
            return false;
path.push_back(cell);
        visited[cell] = true;

        if (cell == targetCell)
            return true;
        
        for (int i = 0; i < 3; i++)
        {
            int neighbor = navmesh.triAdjacency[cell * 3 + i];
            if (neighbor < 0 )
                continue;
            if (VisitCell(navmesh, neighbor, targetCell, path, visited))
                return true;
        }
        path.pop_back();
        return false;
    }

    Vector3 BruteForcePathfindToPoint(const Navmesh& navmesh, const Vector3& start, const Vector3& pos)
    {
        // Brute force navigation.
        Vector3 stop = { 0.0f, 0.0f, 0.0f };
        int startTri = NearestNavTri(navmesh, start);
        int posTri = NearestNavTri(navmesh, pos);
        if (startTri < 0 || posTri < 0)
            return stop;
        std::vector<int> path;
        path.reserve(navmesh.numTris);
        std::vector<bool> visited(navmesh.numTris, false);
        if (VisitCell(navmesh, startTri, posTri, path, visited))
        {
            if (path.size() > 1)
                return CenterOfTri(navmesh, path[1]);
            return pos;
        }
        return stop;
    }

    struct Node
    {
        int idx;
        int cameFrom;
        float startDist;
        float score;

        void Clear(int index)
        {
            idx = index;
            cameFrom = -1;
            startDist = FLT_MAX;
            score = FLT_MAX;
        }
    };
    Vector3 AStarPathfindToPoint(const Navmesh& navmesh, const Vector3& start, const Vector3& pos)
    {
		Vector3 stop = { 0.0f, 0.0f, 0.0f };
		int startTri = NearestNavTri(navmesh, start);
		int posTri = NearestNavTri(navmesh, pos);
		if (startTri < 0 || posTri < 0)
			return stop;

        static std::vector<Node> state(navmesh.numTris);
		for (int tri = 0; tri < (int)navmesh.numTris; tri++)
			state[tri].Clear(tri);
        state[startTri].startDist = 0.0f;

        auto sortHeap = [](const int &lhs, const int &rhs)
        {
            return state[lhs].score < state[rhs].score;
        };
        std::set<int, decltype(sortHeap)> heap(sortHeap);
        heap.insert(startTri);

        while (!heap.empty())
        {
            int thisTri = *heap.begin();
            if (thisTri == posTri)
            {
                int goal = thisTri;
                while (goal != startTri && goal != -1)
                {
                    if (state[goal].cameFrom == startTri)
                        return CenterOfTri(navmesh, goal);
                    goal = state[goal].cameFrom;
                }
                return pos;
            }

            heap.erase(heap.begin());
            Vector3 center = CenterOfTri(navmesh, thisTri);
            if (thisTri == startTri)
                center = start;
            for (int i = 0; i < 3; i++)
            {
                int neighbor = navmesh.triAdjacency[thisTri * 3 + i];
                if (neighbor == -1)
                    continue;
                Vector3 neighpos = CenterOfTri(navmesh, neighbor);
                float score = state[thisTri].startDist + center.distance(neighpos);
                if (score < state[neighbor].startDist)
                {
                    state[neighbor].cameFrom = thisTri;
                    state[neighbor].startDist = score;
                    state[neighbor].score = score + neighpos.distance(pos);
                    heap.insert(neighbor);
                }
            }
        }
        return stop;
    }

    Vector3 PathfindToPoint(const Navmesh& navmesh, const Vector3& start, const Vector3& pos)
    {
        //return BruteForcePathfindToPoint(navmesh, start, pos);    
		return AStarPathfindToPoint(navmesh, start, pos);
	}
};

namespace madronaMPEnv {

using namespace gas;

using madrona::math::Mat3x4;
using madrona::math::Diag3x3;

struct MeshMaterial {
  Vector3 color;
};

struct Mesh {
  Buffer buffer;
  u32 vertexOffset;
  u32 indexOffset;
  u32 numTriangles;
  i32 materialIndex;
};

struct Object {
  i32 meshOffset;
  i32 numMeshes;
};

struct AssetGroup {
  Buffer geometryBuffer;
  i32 objectsOffset;
  i32 numObjects;
};

struct Camera {
  Vector3 position;
  Vector3 fwd;
  Vector3 up;
  Vector3 right;

  Vector3 mapMin;
  Vector3 mapMax;
  Vector3 target;

  float fov = 60.f;
  float orthoHeight = 5.f;

  float heading = 0.0f;
  float azimuth = 45.0f;
  float zoom = 1.0f;
  bool perspective = true;
  bool fine_aim = false;
  bool orbit = true;
};

enum class AnalyticsThreadCtrl : u32 {
  Idle,
  Filter,
  Exit,
};

struct FilterResult {
  i64 matchID;
  i64 teamID;
  i64 windowStart;
  i64 windowEnd;
};

struct AnalyticsStepSnapshot {
  PackedMatchState matchData;
  PackedPlayerSnapshot playerData[consts::maxTeamSize * 2];
  u32 captureEventsOffset;
  u32 numCaptureEvents;
  u32 reloadEventsOffset;
  u32 numReloadEvents;
  u32 killEventsOffset;
  u32 numKillEvents;
  u32 playerShotEventsOffset;
  u32 numPlayerShotEvents;
};

struct AnalyticsMatchData {
  DynArray<AnalyticsStepSnapshot> steps;
  DynArray<GameEvent::Capture> captureEvents;
  DynArray<GameEvent::Reload> reloadEvents;
  DynArray<GameEvent::Kill> killEvents;
  DynArray<GameEvent::PlayerShot> playerShotEvents;
};

static Camera initCam(Vector3 pos, Vector3 fwd, Vector3 up)
{
  fwd = normalize(fwd);
  up = normalize(up);
  Vector3 right = normalize(cross(fwd, up));
  up = normalize(cross(right, fwd));

  return Camera{
    .position = pos,
    .fwd = fwd,
    .up = up,
    .right = right,
    .target = Vector3(0.0f, 0.0f, 0.0f),
  };
}

struct AnalyticsDB {
#ifdef DB_SUPPORT
  sqlite3 *hdl = nullptr;
  sqlite3_stmt *loadStepPlayerStates = nullptr;
  sqlite3_stmt *loadMatchZoneState = nullptr;
  sqlite3_stmt *loadMatchStepIDs = nullptr;
  sqlite3_stmt *loadTeamStates = nullptr;

  sqlite3_stmt *loadMatchStepPlayerSnapshots = nullptr;
  sqlite3_stmt *loadMatchStepMatchDataSnapshots = nullptr;
  sqlite3_stmt *loadMatchCaptureEvents = nullptr;
  sqlite3_stmt *loadMatchReloadEvents = nullptr;
  sqlite3_stmt *loadMatchKillEvents = nullptr;
  sqlite3_stmt *loadMatchPlayerShotEvents = nullptr;

  AnalyticsFilterType addFilterType = AnalyticsFilterType::CaptureEvent;

  DynArray<AnalyticsFilter> currentFilters =
      DynArray<AnalyticsFilter>(0);

  AABB2D16 *visualSelectRegion = nullptr;
  bool visualSelectRegionSelectMin = true;

  int filterTimeWindow = 0;

  alignas(MADRONA_CACHE_LINE) AtomicU32 threadCtrl =
      (u32)AnalyticsThreadCtrl::Idle;

  AtomicU32 resultsStatus = 0;

  Optional<DynArray<FilterResult>> filteredResults =
      Optional<DynArray<FilterResult>>::none();

  Optional<DynArray<FilterResult>> displayResults =
      Optional<DynArray<FilterResult>>::none();

  int currentSelectedResult = -1;
  int currentVizMatch = -1;
  int currentVizMatchTimestep = -1;

  int numMatches = 0;

  std::array<TeamConvexHull, 2> eventTeamConvexHulls = {};

  StepSnapshot curSnapshot = {};

  bool playLoggedStep = false;
  std::chrono::time_point<std::chrono::steady_clock> lastMatchReplayTick = {};
  int matchReplayHz = 100;

  std::ofstream trajectoriesOutFile {};

  std::thread bgThread = {};
#endif
};

static RasterShader loadShader(VizState* viz, const char* path, RasterShaderInit init);

class PostEffectPass
{
public:
  PostEffectPass();
  void Prepare(struct VizState* _viz, ShaderID _shaderID, float resXMult, float resYMult, int colorOutputs, bool outputDepth);
  void AddTextureInput(gas::Texture& texture, bool volumeTexture = false );
  void AddDepthInput(gas::Texture& depth);
  void SetParams(const Vector4 &shaderParams);
  gas::RasterPassEncoder Execute(bool final);
  void Destroy();
  gas::Texture& Output(int i);
  gas::Texture& Depth();
private:
  struct VizState* viz = nullptr;
  RasterPassInterface interface;
  RasterPass pass;
  std::vector<Texture> targets;
  std::vector<Sampler> samplers;
  std::vector<TextureBindingConfig> bindings;
  std::vector<SamplerBindingConfig> samplerBindings;
  std::vector<Texture> inputs;
  ParamBlockType paramType;
  ParamBlock params;
  ParamBlock params2;
  RasterShader shader;
  ShaderID shaderID;
  u16 resX;
  u16 resY;
  bool initialized;
};

const int DownsamplePasses = 3;

struct AgentRecentTrajectory {
  std::array<Vector3, 256> points;
  i32 curOffset = 0;
};

struct VizState {
  UISystem *ui;
  Window *window;

  GPULib *gpuAPI;
  GPUDevice *gpu;

  Swapchain swapchain;
  GPUQueue mainQueue;
  TextureFormat swapchainFormat;

  brt::StackAlloc persistentAlloc;
  VizShaders vizShaders;

  Texture depthAttachment;
  Texture sceneColor;
  Texture sceneDepth;
  Sampler sceneSampler;
  Sampler depthSampler;
  Texture heatmapTexture;

  RasterPassInterface offscreenPassInterface;
  RasterPass offscreenPass;
  RasterPass mainmenuPass;

  PostEffectPass ssaoPass;
  PostEffectPass ssaoDownsamplePass;
  PostEffectPass downsamplePasses[DownsamplePasses];
  PostEffectPass bloomHorizontalPasses[DownsamplePasses];
  PostEffectPass bloomVerticalPasses[DownsamplePasses];
  PostEffectPass finalPass;

  ParamBlockType globalParamBlockType;
  ParamBlockType mapGeoParamBlockType;
  ParamBlockType postEffectParamBlockType;
  Buffer globalPassDataBuffer;
  Buffer postEffectDataBuffer;
  ParamBlock globalParamBlock;

  RasterShader mapShader;
  RasterShader renderableObjectsShader;
  RasterShader agentShader;
  RasterShader goalRegionsShader;
  RasterShader goalRegionsShaderWireframe;
  RasterShader goalRegionsShaderWireframeNoDepth;
  RasterShader analyticsTeamHullShader;

  ParamBlockType agentPathsParamBlockType;
  RasterShader agentPathsShader;

  ParamBlockType shotVizParamBlockType;

  RasterShader shotVizShader;

  CommandEncoder enc;

  i32 curWorld;
  i32 curView;
  i32 curControl;

  i32 numWorlds;
  i32 numViews;
  i32 teamSize;

  i32 simTickRate;
  bool doAI[2];

  float mouseSensitivity = 30.f;

  /*Camera flyCam = {
    .position = {202.869324, 211.050766, 716.584534},
    .fwd = {0.592786, 0.093471, -0.799917},
    .up = {0.790154, 0.124592, 0.600111},
    .right = {0.155756, -0.987796, -0.000000},
  };*/

	  //.position = {},
	  //.fwd = {}

  Camera flyCam = initCam({79, 143, 4307},
	                           {0.0f,-0.05f, -1.00f},
                             {0, 1, -0.02f});
  
  //Camera flyCam = initCam({43000, 8500, 4500},
	//                           {0.2f,0.0f, -0.9f},
  //                           {0, 1, -0.02f});

#if 0
    initCam(
    { 1500, 0, 2400.f },
    ( Quat::angleAxis(math::pi / 2.f, math::up) *
      Quat::angleAxis(-0.35f * math::pi, math::right)
    ).normalize()
  );
#endif
  bool linkViewControl = true;
  float cameraMoveSpeed = 2000.f;

  bool showUnmaskedMinimaps = false;

  UserInputEvents simEventsState {};

  std::vector<MeshMaterial> meshMaterials = { { Vector3(1, 1, 1) } };
  std::vector<Mesh> meshes = {};
  std::vector<Object> objects = {};
  std::vector<AssetGroup> objectAssetGroups = {};

  Buffer mapBuffer = {};
  ParamBlock mapGeoParamBlock = {};
  std::vector<MapGeoMesh> mapMeshes = {};

  bool mainMenu = true;
  bool gameRunning = false;
  bool debugMenus = false;

  AnalyticsDB db = {};

  AgentRecentTrajectory agentTrajectories[consts::maxTeamSize * 2] = {};

  const char *recordedDataPath = nullptr;
  
  TrajectoryDB *trajectoryDB = nullptr;
  std::vector<i64> curWorkingTrajectories = {};

  std::vector<AgentTrajectoryStep> humanTrace = {};

  i64 curVizTrajectoryID = -1;
  i32 curVizTrajectoryIndex = -1;

  std::chrono::steady_clock::time_point last_sim_tick_time = {};
  std::chrono::steady_clock::time_point last_frontend_tick_time = {};

  float mouse_yaw_delta = 0.f;
  float mouse_pitch_delta = 0.f;
};

PostEffectPass::PostEffectPass()
{
  initialized = false;
}

void PostEffectPass::Prepare(struct VizState* _viz, ShaderID _shaderID, float resXMult, float resYMult, int colorOutputs, bool finalOutput)
{
  if (initialized)
    return;

  viz = _viz;
  shaderID = _shaderID;
 
  static UUID broken = { 123,0 };
  broken[1]++;

  std::vector<gas::ColorAttachmentConfig> colorConfigList;
  resX = (u16)(viz->window->pixelWidth * resXMult);
  resY = (u16)(viz->window->pixelHeight * resYMult);
  for (int i = 0; i < colorOutputs; i++)
  {
    targets.push_back(viz->gpu->createTexture({
        .format = viz->swapchainFormat,
        .width = resX,
        .height = resY,
        .usage = TextureUsage::ColorAttachment | TextureUsage::ShaderSampled,
      }));

    colorConfigList.push_back({
      .format = viz->swapchainFormat,
      .loadMode = AttachmentLoadMode::Clear,
    });
  }

  madrona::Span<gas::ColorAttachmentConfig> colorConfigs = madrona::Span<gas::ColorAttachmentConfig>(colorConfigList.data(), colorConfigList.size());

  if (finalOutput)
  {
    targets.push_back(viz->gpu->createTexture({
        .format = TextureFormat::Depth32_Float,
        .width = resX,
        .height = resY,
        .usage = TextureUsage::ColorAttachment | TextureUsage::ShaderSampled,
      }));

    DepthAttachmentConfig depthConfig = {
      .format = TextureFormat::Depth32_Float,
      .loadMode = AttachmentLoadMode::Clear,
    };

    interface = viz->gpu->createRasterPassInterface({
      .uuid = broken,//UUID::randomFromSeedString(name.c_str(), name.size()),
      .depthAttachment = depthConfig,
      .colorAttachments = colorConfigs,
      });

    pass = viz->gpu->createRasterPass({
    .interface = interface,
    .depthAttachment = targets[1],
    .colorAttachments = { viz->swapchain.proxyAttachment() },
      });
  }
  else
  {
    interface = viz->gpu->createRasterPassInterface({
      .uuid = broken,//UUID::randomFromSeedString(name.c_str(), name.size()),
      .colorAttachments = colorConfigs,
      });

    pass = viz->gpu->createRasterPass({
    .interface = interface,
    .colorAttachments = { targets[0] },
      });
  }
}

void PostEffectPass::AddTextureInput(gas::Texture& texture, bool volumeTexture /* = false*/)
{
  if (initialized)
    return;
  inputs.push_back(texture);
  samplers.push_back(viz->sceneSampler);
  if( volumeTexture )
    bindings.push_back({
      .type = TextureBindingType::Texture3D,
      .shaderUsage = ShaderStage::Fragment
      });
  else
    bindings.push_back({ .shaderUsage = ShaderStage::Fragment });
  samplerBindings.push_back({ .shaderUsage = ShaderStage::Fragment });
}

void PostEffectPass::AddDepthInput(gas::Texture& texture)
{
  if (initialized)
    return;
  inputs.push_back(texture);
  samplers.push_back(viz->depthSampler);
  bindings.push_back({ .type = TextureBindingType::UnfilterableTexture2D,.shaderUsage = ShaderStage::Fragment });
  samplerBindings.push_back({ .type = SamplerBindingType::NonFiltering,.shaderUsage = ShaderStage::Fragment });
}

void PostEffectPass::SetParams(const Vector4 &shaderParams)
{
  CopyPassEncoder copy_enc = viz->enc.beginCopyPass();
  MappedTmpBuffer param_staging =
    copy_enc.tmpBuffer(sizeof(PostEffectData));

  PostEffectData* param_staging_ptr =
    (PostEffectData*)param_staging.ptr;

  *param_staging_ptr = PostEffectData {
    .params1 = shaderParams,
    .params2 = { resX, resY, 0, 0 },
    .mapBBMin = Vector4(viz->flyCam.mapMin.x, viz->flyCam.mapMin.y, viz->flyCam.mapMin.z, 0.f),
    .mapBBMax = Vector4(viz->flyCam.mapMax.x, viz->flyCam.mapMax.y, viz->flyCam.mapMax.z + 65.0f * 2.0f, 0.f),
  };

  copy_enc.copyBufferToBuffer(
    param_staging.buffer, viz->postEffectDataBuffer,
    param_staging.offset, 0, sizeof(PostEffectData));

  viz->enc.endCopyPass(copy_enc);
}

gas::RasterPassEncoder PostEffectPass::Execute( bool final )
{
  // Set up the inputs based on what was called between Prepare and Execute.
  if (!initialized)
  {
    // Convert collected vectors into spans to input into constructor.
    madrona::Span<gas::Texture> textureInputs = madrona::Span<gas::Texture>(inputs.data(), inputs.size());
    madrona::Span<gas::Sampler> samplerInputs = madrona::Span<gas::Sampler>(samplers.data(), samplers.size());
    madrona::Span<gas::TextureBindingConfig> textureBindingsSpan = madrona::Span<gas::TextureBindingConfig>(bindings.data(), bindings.size());
    madrona::Span<gas::SamplerBindingConfig> samplerBindingsSpan = madrona::Span<gas::SamplerBindingConfig>(samplerBindings.data(), samplerBindings.size());

    static UUID broken = { 0,0 };
    broken[0]++;
    broken[1]++;
    paramType = viz->gpu->createParamBlockType({
    .uuid = broken,//UUID::randomFromSeedString(name.c_str(), name.size()),
    .textures = textureBindingsSpan,
    .samplers = samplerBindingsSpan,
      });

    params = viz->gpu->createParamBlock({
      .typeID = paramType,
      .textures = textureInputs,
      .samplers = samplerInputs,
      });

    params2 = viz->gpu->createParamBlock({
      .typeID = viz->postEffectParamBlockType,
      .buffers = {
        {.buffer = viz->postEffectDataBuffer, .numBytes = sizeof(PostEffectData) },
      }
      });

    shader = viz->gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(shaderID),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = interface,
      .paramBlockTypes = { viz->globalParamBlockType, paramType, viz->postEffectParamBlockType },
      .rasterConfig = {
        .depthCompare = DepthCompare::Disabled,
      },
      });
  }

  initialized = true;
  gas::RasterPassEncoder enc = viz->enc.beginRasterPass(pass);
  enc.setShader(shader);
  enc.setParamBlock(0, viz->globalParamBlock);
  enc.setParamBlock(1, params);
  enc.setParamBlock(2, params2);
  enc.draw(0, 1);
  if (!final)
    viz->enc.endRasterPass(enc);
  return enc;
}

gas::Texture& PostEffectPass::Output(int i)
{
  return targets[i];
}

gas::Texture& PostEffectPass::Depth()
{
  return targets[targets.size() - 1];
}

void PostEffectPass::Destroy()
{
  if (!viz) {
    return;
  }

  viz->gpu->destroyRasterShader(shader);
  viz->gpu->destroyRasterPass(pass);
  viz->gpu->destroyRasterPassInterface(interface);
  for( gas::Texture &texture : targets )
    viz->gpu->destroyTexture(texture);
}

struct VizWorld {
  VizState *viz;

  Query<Position, Rotation, Scale, ObjectID> renderableObjectsQuery;
};

static inline float srgbToLinear(float srgb)
{
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    }

    return powf((srgb + 0.055f) / 1.055f, 2.4f);
}

static inline Vector4 rgb8ToFloat(uint8_t r, uint8_t g, uint8_t b,
                                  float alpha = 1.f)
{
    return {
        srgbToLinear((float)r / 255.f),
        srgbToLinear((float)g / 255.f),
        srgbToLinear((float)b / 255.f),
        alpha,
    };
}

static void loadObjects(VizState *viz,
                        Span<const imp::SourceObject> objs,
                        Span<const imp::SourceMaterial> materials,
                        Span<const imp::SourceTexture> imported_textures)
{
  (void)imported_textures;

  i32 new_materials_start_offset = (i32)viz->meshMaterials.size();
  for (const auto &mat : materials) {
    viz->meshMaterials.push_back({
      .color = { mat.color.x, mat.color.y, mat.color.z },
    });
  }

  GPUDevice *gpu = viz->gpu;
  CommandEncoder &enc = viz->enc;

  u32 total_num_bytes;
  {
    u32 cur_num_bytes = 0;
    for (const auto &src_obj : objs) {
      for (const auto &src_mesh : src_obj.meshes) {
        assert(src_mesh.faceCounts == nullptr);

        cur_num_bytes = utils::roundUp(cur_num_bytes, (u32)sizeof(OpaqueGeoVertex));
        cur_num_bytes += sizeof(OpaqueGeoVertex) * src_mesh.numVertices;
        cur_num_bytes += sizeof(u32) * src_mesh.numFaces * 3;
      }
    }
    total_num_bytes = cur_num_bytes;
  }

  Buffer staging = gpu->createStagingBuffer(total_num_bytes);
  Buffer mesh_buffer = gpu->createBuffer({
    .numBytes = total_num_bytes,
    .usage = BufferUsage::DrawVertex | BufferUsage::DrawIndex |
        BufferUsage::CopyDst,
  });

  u8 *staging_ptr;
  gpu->prepareStagingBuffers(1, &staging, (void **)&staging_ptr);

  i32 new_objs_start = viz->objects.size();
  u32 cur_buf_offset = 0;
  for (const auto &src_obj : objs) {
    viz->objects.push_back({(i32)viz->meshes.size(), (i32)src_obj.meshes.size()});

    for (const auto &src_mesh : src_obj.meshes) {
      cur_buf_offset = utils::roundUp(cur_buf_offset, (u32)sizeof(OpaqueGeoVertex));
      u32 vertex_offset = cur_buf_offset / sizeof(OpaqueGeoVertex);

      OpaqueGeoVertex *vertex_staging =
          (OpaqueGeoVertex *)(staging_ptr + cur_buf_offset);

      for (i32 i = 0; i < (i32)src_mesh.numVertices; i++) {
        vertex_staging[i] = OpaqueGeoVertex {
          .pos = src_mesh.positions[i],
          .normal = src_mesh.normals[i],
          .uv = src_mesh.uvs[i],
        };
      }

      cur_buf_offset += sizeof(OpaqueGeoVertex) * src_mesh.numVertices;

      u32 index_offset = cur_buf_offset / sizeof(u32);
      u32 *indices_staging = (u32 *)(staging_ptr + cur_buf_offset);

      u32 num_index_bytes = sizeof(u32) * src_mesh.numFaces * 3;
      memcpy(indices_staging, src_mesh.indices, num_index_bytes);
      cur_buf_offset += num_index_bytes;

      viz->meshes.push_back({
        .buffer = mesh_buffer,
        .vertexOffset = vertex_offset,
        .indexOffset = index_offset, 
        .numTriangles = src_mesh.numFaces,
        .materialIndex = src_mesh.materialIDX == 0xFFFF'FFFF ? 0 :
             new_materials_start_offset + (i32)src_mesh.materialIDX,
      });
    }
  }
  assert(cur_buf_offset == total_num_bytes);

  gpu->flushStagingBuffers(1, &staging);

  gpu->waitUntilReady(viz->mainQueue);

  {
    enc.beginEncoding();
    CopyPassEncoder copy_enc = enc.beginCopyPass();

    copy_enc.copyBufferToBuffer(staging, mesh_buffer, 0, 0, total_num_bytes);

    enc.endCopyPass(copy_enc);
    enc.endEncoding();
  }

  gpu->submit(viz->mainQueue, enc);
  gpu->waitUntilWorkFinished(viz->mainQueue);

  gpu->destroyStagingBuffer(staging);

  viz->objectAssetGroups.push_back({
    .geometryBuffer = mesh_buffer,
    .objectsOffset = new_objs_start,
    .numObjects = (i32)objs.size(),
  });
}

static void loadAssets(VizState *viz)
{
  auto capsule_path =
      (std::filesystem::path(DATA_DIR) / "capsule.obj").string();
  auto capsule_path_cstr = capsule_path.c_str();
  
  imp::AssetImporter importer;
  std::array<char, 1024> import_err;
  auto capsule_asset = importer.importFromDisk(
      Span(&capsule_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  if (!capsule_asset.has_value()) {
      FATAL("Failed to load capsule: %s", import_err);
  }
  
  auto other_capsule_asset = importer.importFromDisk(
      Span(&capsule_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto shot_cylinder_path =
      (std::filesystem::path(DATA_DIR) / "shot_cylinder.obj").string();
  auto shot_cylinder_path_cstr = shot_cylinder_path.c_str();
  auto shot_cylinder_asset_a_miss = importer.importFromDisk(
      Span(&shot_cylinder_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  if (!shot_cylinder_asset_a_miss.has_value()) {
      FATAL("Failed to load shot cylinder: %s", import_err);
  }
  
  auto shot_cylinder_asset_a_hit = importer.importFromDisk(
      Span(&shot_cylinder_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto shot_cylinder_asset_b_miss = importer.importFromDisk(
      Span(&shot_cylinder_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto shot_cylinder_asset_b_hit = importer.importFromDisk(
      Span(&shot_cylinder_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto zone_marker_path =
      (std::filesystem::path(DATA_DIR) / "zone_marker.obj").string();
  auto zone_marker_path_cstr = zone_marker_path.c_str();
  
  auto zone_marker_asset_inactive = importer.importFromDisk(
      Span(&zone_marker_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto zone_marker_asset_contested = importer.importFromDisk(
      Span(&zone_marker_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto zone_marker_asset_team_a = importer.importFromDisk(
      Span(&zone_marker_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto zone_marker_asset_team_b = importer.importFromDisk(
      Span(&zone_marker_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto spawn_marker_path =
      (std::filesystem::path(DATA_DIR) / "spawn_marker.obj").string();
  auto spawn_marker_path_cstr = spawn_marker_path.c_str();
  auto spawn_marker_asset_respawn = importer.importFromDisk(
      Span(&spawn_marker_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  auto spawn_marker_asset_team_a = importer.importFromDisk(
      Span(&spawn_marker_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  auto spawn_marker_asset_team_b = importer.importFromDisk(
      Span(&spawn_marker_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto respawn_region_marker_path =
      (std::filesystem::path(DATA_DIR) / "respawn_region_marker.obj").string();
  auto respawn_region_marker_path_cstr = zone_marker_path.c_str();
  auto respawn_region_marker_asset = importer.importFromDisk(
      Span(&respawn_region_marker_path_cstr, 1),
      Span<char>(import_err.data(), import_err.size()));
  
  auto materials = std::to_array<imp::SourceMaterial>({
      { rgb8ToFloat(230, 230, 230), -1, 0.8f, 1.0f },
      { rgb8ToFloat(20, 20, 230),   -1, 0.8f, 1.0f },
      { rgb8ToFloat(230, 20, 20),   -1, 0.8f, 1.0f },
      { rgb8ToFloat(100, 100, 230), -1, 0.8f, 1.0f },
      { rgb8ToFloat(20, 20, 100),   -1, 0.8f, 1.0f },
      { rgb8ToFloat(230, 100, 100), -1, 0.8f, 1.0f },
      { rgb8ToFloat(100, 20, 20),   -1, 0.8f, 1.0f },
      { rgb8ToFloat(20, 100, 20),   -1, 0.8f, 1.0f },
      { rgb8ToFloat(100, 230, 100),   -1, 0.8f, 1.0f },
  });
  
  HeapArray<imp::SourceObject> common_objects(consts::numNonMapAssets);
  common_objects[0] = capsule_asset->objects[0];
  common_objects[1] = other_capsule_asset->objects[0];
  
  common_objects[2] = shot_cylinder_asset_a_hit->objects[0];
  common_objects[3] = shot_cylinder_asset_a_miss->objects[0];
  common_objects[4] = shot_cylinder_asset_b_hit->objects[0];
  common_objects[5] = shot_cylinder_asset_b_miss->objects[0];
  common_objects[6] = zone_marker_asset_inactive->objects[0];
  common_objects[7] = zone_marker_asset_contested->objects[0];
  common_objects[8] = zone_marker_asset_team_a->objects[0];
  common_objects[9] = zone_marker_asset_team_b->objects[0];
  common_objects[10] = spawn_marker_asset_respawn->objects[0];
  common_objects[11] = spawn_marker_asset_team_a->objects[0];
  common_objects[12] = spawn_marker_asset_team_b->objects[0];
  common_objects[13] = respawn_region_marker_asset->objects[0];
  
  // Capsules
  common_objects[0].meshes[0].materialIDX = 1;
  common_objects[1].meshes[0].materialIDX = 2;
  
  // ShotViz?
  common_objects[2].meshes[0].materialIDX = 3;
  common_objects[3].meshes[0].materialIDX = 4;
  common_objects[4].meshes[0].materialIDX = 5;
  common_objects[5].meshes[0].materialIDX = 6;
  
  // Zone assets
  common_objects[6].meshes[0].materialIDX = 7;
  common_objects[7].meshes[0].materialIDX = 8;
  common_objects[8].meshes[0].materialIDX = 1;
  common_objects[9].meshes[0].materialIDX = 2;
  
  // Spawn Marker
  common_objects[10].meshes[0].materialIDX = 8;
  common_objects[11].meshes[0].materialIDX = 1;
  common_objects[12].meshes[0].materialIDX = 2;
  
  // Respawn Region Marker
  common_objects[13].meshes[0].materialIDX = 2;
  
  imp::ImageImporter &tex_importer = importer.imageImporter();
  
  StackAlloc tmp_alloc;
  Span<imp::SourceTexture> imported_textures =
      tex_importer.importImages(tmp_alloc, {});
  
  loadObjects(viz, common_objects, materials, imported_textures);
}

namespace VizSystem {

static void vizStep(VizState *viz, Manager &mgr, float delta_t);

static void initMapCamera(VizState* viz, Manager& mgr)
{
  // Loop through all of the navmesh points and find the total bounding box.
  Engine &ctx = mgr.getWorldContext(viz->curWorld);
  const Navmesh& navmesh = ctx.singleton<LevelData>().navmesh;
  float unreasonablyLargeNum = 1e6f;
  viz->flyCam.mapMin = {
    unreasonablyLargeNum,
    unreasonablyLargeNum,
    unreasonablyLargeNum,
  };
  viz->flyCam.mapMax = {
    -unreasonablyLargeNum,
    -unreasonablyLargeNum,
    -unreasonablyLargeNum,
  };
  for (unsigned int i = 0; i < navmesh.numVerts; i++)
  {
    const Vector3& vert = navmesh.vertices[i];
    viz->flyCam.mapMin.x = std::min(viz->flyCam.mapMin.x, vert.x);
    viz->flyCam.mapMin.y = std::min(viz->flyCam.mapMin.y, vert.y);
    viz->flyCam.mapMin.z = std::min(viz->flyCam.mapMin.z, vert.z);
    viz->flyCam.mapMax.x = std::max(viz->flyCam.mapMax.x, vert.x);
    viz->flyCam.mapMax.y = std::max(viz->flyCam.mapMax.y, vert.y);
    viz->flyCam.mapMax.z = std::max(viz->flyCam.mapMax.z, vert.z);
  }
  viz->flyCam.target = (viz->flyCam.mapMax + viz->flyCam.mapMin) * 0.5f;
}

static void loadHeatmapData(VizState *viz)
{
  // TEMP GENERATE HEATMAP DATA!
  constexpr int heatmapWidth = 64;
  constexpr int heatmapHeight = 64;
  constexpr int heatmapDepth = 3;
  i64 * heatmapPixels = new i64[heatmapWidth * heatmapHeight * heatmapDepth];
#if 0
  // Generate random walk noise.
  float variance = 0.2f;
  for (int x = 0; x < heatmapWidth; x++)
  {
    for (int y = 0; y < heatmapHeight; y++)
    {
      for (int z = 0; z < heatmapDepth; z++)
      {
        // Average the values of all the neighbors we've already visited.
        float prev = 0.0f;
        float norm = 0.0f;
        if (x > 0)
        {
          norm++;
          prev += heatmapPixels[(z * heatmapWidth * heatmapHeight + y * heatmapWidth + (x - 1)) * 3];
        }
        if (y > 0)
        {
          norm++;
          prev += heatmapPixels[(z * heatmapWidth * heatmapHeight + (y-1) * heatmapWidth + x) * 3];
        }
        if (z > 0)
        {
          norm++;
          prev += heatmapPixels[((z-1) * heatmapWidth * heatmapHeight + y * heatmapWidth + x) * 3];
        }
        if (norm > 0)
          prev /= norm;

        // Generate a new value randomly offset.
        heatmapPixels[(z * heatmapWidth * heatmapHeight + y * heatmapWidth + x) * 3] = std::max(0.0f, std::min(1.0f, prev - variance + ((std::rand() % 1024)/1024.0f) * variance * 2.0f));
      }
    }
  }
#endif
  memset(heatmapPixels, 0,
    sizeof(i64) * heatmapWidth * heatmapHeight * heatmapDepth);

  float heatmap_rescale = 1.f;

  if (viz->recordedDataPath) {
    auto fileNumElems =
      []<typename T>
    (std::ifstream &f)
    {
      f.seekg(0, f.end);
      i64 size = f.tellg();
      f.seekg(0, f.beg);

      assert(size % sizeof(T) == 0);

      return size / sizeof(T);
    };

    std::ifstream steps_file(viz->recordedDataPath, std::ios::binary);
    assert(steps_file.is_open());

    i64 num_steps = fileNumElems.template operator()<PackedStepSnapshot>(steps_file);
    HeapArray<PackedStepSnapshot> steps(num_steps);
    steps_file.read((char *)steps.data(), sizeof(PackedStepSnapshot) * num_steps);

    Vector3 mapBBMin(viz->flyCam.mapMin.x, viz->flyCam.mapMin.y, viz->flyCam.mapMin.z);
    Vector3 mapBBMax(viz->flyCam.mapMax.x, viz->flyCam.mapMax.y, viz->flyCam.mapMax.z + 65.0f * 2.0f);

    Vector3 boxExtent = mapBBMax - mapBBMin;

    i64 max_heatmap_count = 0;
    for (i64 step_idx = 0; step_idx < num_steps; step_idx++) {
      PackedStepSnapshot &snapshot = steps[step_idx];

      for (i64 player_idx = 0; player_idx < viz->teamSize * 2; player_idx++) {
        PackedPlayerSnapshot &player = snapshot.players[player_idx];

        Vector3 pos(player.pos[0], player.pos[1], player.pos[2]);

        Vector3 uvw = (pos - mapBBMin);
        uvw.x /= boxExtent.x;
        uvw.y /= boxExtent.y;
        uvw.z /= boxExtent.z;

        int coord_x = std::clamp(int(uvw.x * heatmapWidth + 0.5f), 0, heatmapWidth - 1);
        int coord_y = std::clamp(int(uvw.y * heatmapHeight + 0.5f), 0, heatmapHeight - 1);
        int coord_z = std::clamp(int(uvw.z * heatmapDepth + 0.5f), 0, heatmapDepth - 1);
        coord_z += 1;

        int linear_idx = coord_z * heatmapWidth * heatmapHeight + coord_y * heatmapWidth + coord_x;

        heatmapPixels[linear_idx] += 1;

        if (heatmapPixels[linear_idx] > max_heatmap_count) {
          max_heatmap_count = heatmapPixels[linear_idx];
        }
      }
    }

    if (max_heatmap_count > 0) {
      heatmap_rescale = 1.f / float(max_heatmap_count);
    }
  }

  u8 *heatmapBytes = new u8[heatmapWidth * heatmapHeight * heatmapDepth * 4];
  for (int i = 0; i < heatmapWidth * heatmapHeight * heatmapDepth; i++)
  {
    float v = 10.f * heatmapPixels[i] * heatmap_rescale;

    heatmapBytes[i * 4 + 0] = (u8)(v * 255);
    heatmapBytes[i * 4 + 1] = (u8)(v * 255);
    heatmapBytes[i * 4 + 2] = (u8)(v * 255);
    heatmapBytes[i * 4 + 3] = (u8)255;
  }

  GPUDevice *gpu = viz->gpu;

  viz->heatmapTexture = gpu->createTexture({
    .format = TextureFormat::RGBA8_UNorm,
    .width = (u16)heatmapWidth,
    .height = (u16)heatmapHeight,
    .depth = (u16)heatmapDepth,
    .usage = TextureUsage::ShaderSampled,
    .initData = {
      .ptr = heatmapBytes,
    },
    }, viz->mainQueue);
  gpu->waitUntilWorkFinished(viz->mainQueue);

  delete[] heatmapPixels;
  delete[] heatmapBytes;
}

static constexpr inline f32 MOUSE_SPEED = 2.0f;// 1e-1f;
static constexpr inline f32 MOUSE_SCROLL_SPEED = 2.0f;// 1e-1f;
// FIXME

#ifdef DB_SUPPORT
static void sendAnalyticsThreadCmd(AnalyticsDB &db, AnalyticsThreadCtrl ctrl)
{
  while (true) {
    u32 cur = db.threadCtrl.load_relaxed();
    if (cur == (u32)AnalyticsThreadCtrl::Idle) {
      break;
    }

    db.threadCtrl.wait<sync::relaxed>(cur);
  }

  db.threadCtrl.store_release((u32)ctrl);
  db.threadCtrl.notify_one();
}

static void analyticsBGThread(AnalyticsDB &db);

static void loadAnalyticsDB(VizState *viz,
                            const VizConfig &cfg)
{
  AnalyticsDB &db = viz->db;

  sqlite3_config(SQLITE_CONFIG_SERIALIZED);

  REQ_SQL(db.hdl, sqlite3_open(cfg.analyticsDBPath, &db.hdl));

  db.loadStepPlayerStates = initLoadStepSnapshotStatement(db.hdl);
  db.loadMatchZoneState = initLoadMatchZoneStatement(db.hdl);

  REQ_SQL(db.hdl, sqlite3_prepare_v2(db.hdl, R"(
SELECT
  ms.id
FROM match_steps AS ms
WHERE
  ms.match_id = ?
ORDER BY ms.step_idx
)", -1, &db.loadMatchStepIDs, nullptr));

  REQ_SQL(db.hdl, sqlite3_prepare_v2(db.hdl, R"(
SELECT
  ts.centroid_x, ts.centroid_y, ts.extent_x, ts.extent_y, ts.hull_data
FROM
  team_states AS ts
WHERE
  ts.step_id = ?
ORDER BY
  ts.team_idx
)", -1, &db.loadTeamStates, nullptr));

  REQ_SQL(db.hdl, sqlite3_prepare_v2(db.hdl, R"(
SELECT
  ms.step_idx, ps.pos_x, ps.pos_y, ps.pos_z, ps.yaw, ps.pitch,
  ps.num_bullets, ps.is_reloading,
  ps.hp, ps.flags
FROM match_steps AS ms
INNER JOIN player_states AS ps ON 
  ps.step_id = ms.id
WHERE ms.match_id = ?
ORDER BY ms.step_idx, ps.player_idx
)", -1, &db.loadMatchStepPlayerSnapshots, nullptr));

  REQ_SQL(db.hdl, sqlite3_prepare_v2(db.hdl, R"(
SELECT
  ms.step_idx, ms.cur_zone, ms.cur_zone_controller,
  ms.zone_steps_remaining, ms.zone_steps_until_point
FROM match_steps AS ms
WHERE ms.match_id = ?
ORDER BY ms.step_idx
)", -1, &db.loadMatchStepMatchDataSnapshots, nullptr));

  REQ_SQL(db.hdl, sqlite3_prepare_v2(db.hdl, R"(
SELECT
  ms.step_idx, ce.zone_idx, ce.capture_team_idx, ce.in_zone_mask
FROM match_steps AS ms
INNER JOIN capture_events AS ce ON 
  ce.step_id = ms.id
WHERE ms.match_id = ?
ORDER BY ms.step_idx
)", -1, &db.loadMatchCaptureEvents, nullptr));

  REQ_SQL(db.hdl, sqlite3_prepare_v2(db.hdl, R"(
SELECT
  ms.step_idx, reloaders.player_idx, re.num_bullets
FROM match_steps AS ms
INNER JOIN reload_events AS re ON 
  re.step_id = ms.id
INNER JOIN player_states AS reloaders ON
  reloaders.id = re.player_state_id
WHERE ms.match_id = ?
ORDER BY ms.step_idx, reloaders.player_idx
)", -1, &db.loadMatchReloadEvents, nullptr));

  REQ_SQL(db.hdl, sqlite3_prepare_v2(db.hdl, R"(
SELECT
  ms.step_idx, attackers.player_idx, targets.player_idx
FROM match_steps AS ms
INNER JOIN kill_events AS ke ON 
  ke.step_id = ms.id
INNER JOIN player_states AS attackers ON
  attackers.id = ke.killer_id
INNER JOIN player_states AS targets ON
  targets.id = ke.killed_id
WHERE ms.match_id = ?
ORDER BY ms.step_idx, attackers.player_idx, targets.player_idx
)", -1, &db.loadMatchKillEvents, nullptr));

  REQ_SQL(db.hdl, sqlite3_prepare_v2(db.hdl, R"(
SELECT
  ms.step_idx, attackers.player_idx, targets.player_idx
FROM match_steps AS ms
INNER JOIN player_shot_events AS pse ON 
  pse.step_id = ms.id
INNER JOIN player_states AS attackers ON
  attackers.id = pse.attacker_id
INNER JOIN player_states AS targets ON
  targets.id = pse.target_id
WHERE ms.match_id = ?
ORDER BY ms.step_idx, attackers.player_idx, targets.player_idx
)", -1, &db.loadMatchPlayerShotEvents, nullptr));

  {
    sqlite3_stmt *num_matches_stmt;
    REQ_SQL(db.hdl, sqlite3_prepare_v2(
        db.hdl, "SELECT COUNT(id) FROM matches",
        -1, &num_matches_stmt, nullptr));

    sqlite3_step(num_matches_stmt);
    db.numMatches = sqlite3_column_int(num_matches_stmt, 0);

    REQ_SQL(db.hdl, sqlite3_reset(num_matches_stmt));
    REQ_SQL(db.hdl, sqlite3_finalize(num_matches_stmt));
  }

  db.trajectoriesOutFile = std::ofstream(
      cfg.trajectoriesDBPath, std::ios::binary);

  db.bgThread = std::thread(analyticsBGThread, std::ref(db));
}

static void unloadAnalyticsDB(AnalyticsDB &db)
{
  sendAnalyticsThreadCmd(db, AnalyticsThreadCtrl::Exit);
  db.bgThread.join();

  REQ_SQL(db.hdl, sqlite3_finalize(db.loadMatchStepPlayerSnapshots));
  REQ_SQL(db.hdl, sqlite3_finalize(db.loadMatchStepMatchDataSnapshots));
  REQ_SQL(db.hdl, sqlite3_finalize(db.loadMatchCaptureEvents));
  REQ_SQL(db.hdl, sqlite3_finalize(db.loadMatchReloadEvents));
  REQ_SQL(db.hdl, sqlite3_finalize(db.loadMatchKillEvents));
  REQ_SQL(db.hdl, sqlite3_finalize(db.loadMatchPlayerShotEvents));

  REQ_SQL(db.hdl, sqlite3_finalize(db.loadTeamStates));
  REQ_SQL(db.hdl, sqlite3_finalize(db.loadMatchStepIDs));
  REQ_SQL(db.hdl, sqlite3_finalize(db.loadMatchZoneState));
  REQ_SQL(db.hdl, sqlite3_finalize(db.loadStepPlayerStates));
  REQ_SQL(db.hdl, sqlite3_close(db.hdl));
}

static std::array<TeamConvexHull, 2> loadTeamConvexHulls(
    AnalyticsDB &db,
    i64 step_id)
{
  std::array<TeamConvexHull, 2> hulls;

  sqlite3_bind_int64(db.loadTeamStates, 1, step_id);

  auto res = sqlite3_step(db.loadTeamStates);
  assert(res == SQLITE_ROW);
  const void *t1_blob = sqlite3_column_blob(db.loadTeamStates, 4);
  memcpy(&hulls[0], t1_blob, sizeof(TeamConvexHull));

  assert(sqlite3_step(db.loadTeamStates) == SQLITE_ROW);
  const void *t2_blob = sqlite3_column_blob(db.loadTeamStates, 4);
  memcpy(&hulls[1], t2_blob, sizeof(TeamConvexHull));

  REQ_SQL(db.hdl, sqlite3_reset(db.loadTeamStates));

  return hulls;
}

static DynArray<i64> loadMatchStepIDs(AnalyticsDB &db,
                                    i64 match_id)
{
  DynArray<i64> results(5000);

  sqlite3_bind_int64(db.loadMatchStepIDs, 1, match_id);

  while (sqlite3_step(db.loadMatchStepIDs) == SQLITE_ROW) {
    i64 step_id = sqlite3_column_int(db.loadMatchStepIDs, 0);
    results.push_back(step_id);
  }

  REQ_SQL(db.hdl, sqlite3_reset(db.loadMatchStepIDs));

  return results;
}

static AnalyticsMatchData loadMatchData(AnalyticsDB &db, i64 match_id)
{
  AnalyticsMatchData match_data {
    .steps = {},
    .captureEvents = {},
    .reloadEvents = {},
    .killEvents = {},
    .playerShotEvents = {},
  };


  i32 cur_capture_step = -1;
  i32 cur_reload_step = -1;
  i32 cur_kill_step = -1;
  i32 cur_player_shot_step = -1;

  sqlite3_bind_int64(db.loadMatchStepPlayerSnapshots, 1, match_id);
  sqlite3_bind_int64(db.loadMatchStepMatchDataSnapshots, 1, match_id);
  sqlite3_bind_int64(db.loadMatchCaptureEvents, 1, match_id);
  sqlite3_bind_int64(db.loadMatchReloadEvents, 1, match_id);
  sqlite3_bind_int64(db.loadMatchKillEvents, 1, match_id);
  sqlite3_bind_int64(db.loadMatchPlayerShotEvents, 1, match_id);

  while (sqlite3_step(db.loadMatchStepMatchDataSnapshots) == SQLITE_ROW) {
    i32 step_idx = (i32)match_data.steps.size();

    AnalyticsStepSnapshot snapshot;
    if (cur_capture_step == step_idx) {
      snapshot.captureEventsOffset = (u32)match_data.captureEvents.size() - 1;
      snapshot.numCaptureEvents = 1;
    } else {
      snapshot.captureEventsOffset = (u32)match_data.captureEvents.size();
      snapshot.numCaptureEvents = 0;
    }

    if (cur_reload_step == step_idx) {
      snapshot.reloadEventsOffset = (u32)match_data.reloadEvents.size() - 1;
      snapshot.numReloadEvents = 1;
    } else {
      snapshot.reloadEventsOffset = (u32)match_data.reloadEvents.size();
      snapshot.numReloadEvents = 0;
    }

    if (cur_kill_step == step_idx) {
      snapshot.killEventsOffset = (u32)match_data.killEvents.size() - 1;
      snapshot.numKillEvents = 1;
    } else {
      snapshot.killEventsOffset = (u32)match_data.killEvents.size();
      snapshot.numKillEvents = 0;
    }

    if (cur_player_shot_step == step_idx) {
      snapshot.playerShotEventsOffset = (u32)match_data.playerShotEvents.size() - 1;
      snapshot.numPlayerShotEvents = 1;
    } else {
      snapshot.playerShotEventsOffset = (u32)match_data.playerShotEvents.size();
      snapshot.numPlayerShotEvents = 0;
    }

    {
      int db_step_idx = sqlite3_column_int(db.loadMatchStepMatchDataSnapshots, 0);

      assert(db_step_idx == step_idx);
    }

    snapshot.matchData.step = (u16)step_idx;
    snapshot.matchData.curZone =
        sqlite3_column_int(db.loadMatchStepMatchDataSnapshots, 1);

    snapshot.matchData.curZoneController =
        sqlite3_column_int(db.loadMatchStepMatchDataSnapshots, 2);

    snapshot.matchData.zoneStepsRemaining =
        sqlite3_column_int(db.loadMatchStepMatchDataSnapshots, 3);

    snapshot.matchData.stepsUntilPoint =
        sqlite3_column_int(db.loadMatchStepMatchDataSnapshots, 4);

    for (i32 player_idx = 0; player_idx < consts::maxTeamSize * 2; player_idx++) {
      assert(sqlite3_step(db.loadMatchStepPlayerSnapshots) == SQLITE_ROW);
      {
        int db_step_idx = sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 0);
        assert(db_step_idx == step_idx);
      }

      snapshot.playerData[player_idx] = {
        .pos = {
            (i16)sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 1),
            (i16)sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 2),
            (i16)sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 3),
        },
        .yaw = (i16)sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 4),
        .pitch = (i16)sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 5),
        .magNumBullets = (u8)sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 6),
        .isReloading = (u8)sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 7),
        .hp = (u8)sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 8),
        .flags = (u8)sqlite3_column_int(db.loadMatchStepPlayerSnapshots, 9),
      };
    }

    match_data.steps.push_back(snapshot);

    if (cur_capture_step <= step_idx) {
      int step_res;
      while ((step_res = sqlite3_step(db.loadMatchCaptureEvents)) == SQLITE_ROW) {
        cur_capture_step = sqlite3_column_int(db.loadMatchCaptureEvents, 0);
        match_data.captureEvents.push_back({
          .zoneIDX = 
              (u8)sqlite3_column_int(db.loadMatchCaptureEvents, 1),
          .captureTeam =
              (u8)sqlite3_column_int(db.loadMatchCaptureEvents, 2),
          .inZoneMask =
              (u8)sqlite3_column_int(db.loadMatchCaptureEvents, 3),
        });

        if (cur_capture_step == step_idx) {
          snapshot.numCaptureEvents += 1;
        } else if (cur_capture_step > step_idx) {
          break;
        } else {
          assert(false);
        }
      }

      if (step_res == SQLITE_DONE) {
        cur_capture_step = INT_MAX;
      }
    }

    if (cur_reload_step <= step_idx) {
      int step_res;
      while ((step_res = sqlite3_step(db.loadMatchReloadEvents)) == SQLITE_ROW) {
        cur_reload_step = sqlite3_column_int(db.loadMatchReloadEvents, 0);
        match_data.reloadEvents.push_back({
          .player = 
              (u8)sqlite3_column_int(db.loadMatchReloadEvents, 1),
          .numBulletsAtReloadTime =
              (u8)sqlite3_column_int(db.loadMatchReloadEvents, 2),
        });

        if (cur_reload_step == step_idx) {
          snapshot.numReloadEvents += 1;
        } else if (cur_reload_step > step_idx) {
          break;
        } else {
          assert(false);
        }
      }

      if (step_res == SQLITE_DONE) {
        cur_reload_step = INT_MAX;
      }
    }

    if (cur_kill_step <= step_idx) {
      int step_res;
      while ((step_res = sqlite3_step(db.loadMatchKillEvents)) == SQLITE_ROW) {
        cur_kill_step = sqlite3_column_int(db.loadMatchKillEvents, 0);
        match_data.killEvents.push_back({
          .killer = 
              (u8)sqlite3_column_int(db.loadMatchKillEvents, 1),
          .killed =
              (u8)sqlite3_column_int(db.loadMatchKillEvents, 2),
        });

        if (cur_kill_step == step_idx) {
          snapshot.numKillEvents += 1;
        } else if (cur_kill_step > step_idx) {
          break;
        } else {
          assert(false);
        }
      }

      if (step_res == SQLITE_DONE) {
        cur_kill_step = INT_MAX;
      }
    }

    if (cur_player_shot_step <= step_idx) {
      int step_res;
      while ((step_res = sqlite3_step(db.loadMatchPlayerShotEvents)) == SQLITE_ROW) {
        cur_player_shot_step = sqlite3_column_int(db.loadMatchPlayerShotEvents, 0);
        match_data.playerShotEvents.push_back({
          .attacker = 
              (u8)sqlite3_column_int(db.loadMatchPlayerShotEvents, 1),
          .target =
              (u8)sqlite3_column_int(db.loadMatchPlayerShotEvents, 2),
        });

        if (cur_player_shot_step == step_idx) {
          snapshot.numPlayerShotEvents += 1;
        } else if (cur_player_shot_step > step_idx) {
          break;
        } else {
          assert(false);
        }
      }

      if (step_res == SQLITE_DONE) {
        cur_player_shot_step = INT_MAX;
      }
    }
  }

  REQ_SQL(db.hdl, sqlite3_reset(db.loadMatchStepPlayerSnapshots));
  REQ_SQL(db.hdl, sqlite3_reset(db.loadMatchStepMatchDataSnapshots));
  REQ_SQL(db.hdl, sqlite3_reset(db.loadMatchCaptureEvents));
  REQ_SQL(db.hdl, sqlite3_reset(db.loadMatchReloadEvents));
  REQ_SQL(db.hdl, sqlite3_reset(db.loadMatchKillEvents));
  REQ_SQL(db.hdl, sqlite3_reset(db.loadMatchPlayerShotEvents));

  return match_data;
}

static void dumpTrajectories(AnalyticsDB &db)
{
  const int traj_len = 100;
  const int slide_len = traj_len / 2;

  assert(db.displayResults.has_value());

  i32 num_dumped = 0;

  for (i32 result_idx = 0;
       result_idx < db.displayResults->size();
       result_idx++) {
    FilterResult result = (*db.displayResults)[result_idx];

    DynArray<i64> match_steps = loadMatchStepIDs(db, result.matchID);

    HeapArray<DumpItem> traj_dump(match_steps.size());

    for (i64 i = 0; i < traj_dump.size(); i++) {
      traj_dump[i] = {
        .stepID = match_steps[i],
        .teamID = result.teamID,
      };
    }

    int window_start = result.windowStart;
    int window_end = result.windowEnd;

    int traj_start_border =
        std::max(window_start - slide_len, 0);
    int traj_end_border =
        std::min(window_end + slide_len, (int)match_steps.size());

    for (int cur_traj_start = traj_start_border;
         cur_traj_start < traj_end_border;
         cur_traj_start += slide_len) {
      int cur_end = cur_traj_start + traj_len;

      int past_end = cur_end - (int)match_steps.size();
      if (past_end > 0) {
        cur_traj_start -= past_end;
        cur_end = match_steps.size();
      }
      
      assert(cur_end - cur_traj_start == traj_len);

      db.trajectoriesOutFile.write(
          (char *)(traj_dump.data() + cur_traj_start),
          traj_len * sizeof(DumpItem));

      num_dumped += 1;

      if (past_end > 0) {
        break;
      }
    }
  }

  printf("Dumped %d\n", (int)num_dumped);
}

static void analyticsDBUI(Engine &ctx, VizState *viz)
{
  AnalyticsDB &db = viz->db;

  if (db.hdl == nullptr) {
    return;
  }

  ImGui::Begin("Analytics");

  float box_width = ImGui::CalcTextSize(" ").x * 7_i32;

  u32 filter_results_status = db.resultsStatus.load_relaxed();
  if (filter_results_status != 0) {
    ImGui::BeginDisabled();
  }

  auto getFilterTypeString =
    [](AnalyticsFilterType type) -> const char *
  {
    switch (type) {
    case AnalyticsFilterType::CaptureEvent: {
      return "Capture Event";
    } break;
    case AnalyticsFilterType::ReloadEvent: {
      return "Reload Event";
    } break;
    case AnalyticsFilterType::KillEvent: {
      return "Kill Event";
    } break;
    case AnalyticsFilterType::PlayerShotEvent: {
      return "Player Shot Event";
    } break;
    case AnalyticsFilterType::PlayerInRegion: {
      return "Player In Region";
    } break;
    default: MADRONA_UNREACHABLE();
    }
  };

  ImGui::PushItemWidth(box_width * 3.f);
  if (ImGui::BeginCombo("Filter Type",
                        getFilterTypeString(db.addFilterType))) {

    for (i32 i = 0; i < (i32)AnalyticsFilterType::NUM_TYPES; i++) {
      AnalyticsFilterType ith_type = (AnalyticsFilterType)i;

      const bool is_selected = db.addFilterType == ith_type;
      if (ImGui::Selectable(getFilterTypeString(ith_type),
                            is_selected)) {
        db.addFilterType = ith_type;
      }

      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }

    ImGui::EndCombo();
  }
  ImGui::PopItemWidth();

  if (ImGui::Button("Add Filter")) {
    AnalyticsFilter added_filter;
    added_filter.type = db.addFilterType;
    switch (db.addFilterType) {
    case AnalyticsFilterType::CaptureEvent: {
      added_filter.captureEvent = {};
    } break;
    case AnalyticsFilterType::ReloadEvent: {
      added_filter.reloadEvent = {};
    } break;
    case AnalyticsFilterType::KillEvent: {
      added_filter.killEvent = {};
    } break;
    case AnalyticsFilterType::PlayerShotEvent: {
      added_filter.playerShotEvent = {};
    } break;
    case AnalyticsFilterType::PlayerInRegion: {
      added_filter.playerInRegion = {};
    } break;
    default: MADRONA_UNREACHABLE(); break;
    }

    db.currentFilters.push_back(added_filter);
  }

  ImGui::Spacing();
  ImGui::Separator();

  auto aabbRegionSelector =
    [&]
  (AABB2D16 *region, const char *label)
  {
    ImGui::PushItemWidth(box_width);

    int region_min_x = region->min.x;
    int region_min_y = region->min.y;
    int region_max_x = region->max.x;
    int region_max_y = region->max.y;

    ImGui::DragInt("##Pos Min X",
                   &region_min_x, 1,
                   -32768, 32767, "%d, ", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SameLine();
    ImGui::DragInt("##Pos Min Y",
                   &region_min_y, 1,
                   -32768, 32767, "%d", ImGuiSliderFlags_AlwaysClamp);

    ImGui::SameLine();
    ImGui::Text("to");
    ImGui::SameLine();

    ImGui::DragInt("##Pos Max X",
                   &region_max_x, 1,
                   -32768, 32767, "%d, ", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SameLine();
    ImGui::DragInt("##Pos Max Y",
                   &region_max_y, 1,
                   -32768, 32767, "%d", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SameLine();
    if (ImGui::SmallButton("^")) {
      db.visualSelectRegion = region;
    }
    ImGui::SameLine();
    ImGui::Text("%s", label);

    region->min.x = (i16)region_min_x;
    region->min.y = (i16)region_min_y;
    region->max.x = (i16)region_max_x;
    region->max.y = (i16)region_max_y;

    ImGui::PopItemWidth();

  };

  for (int filter_idx = 0;
       filter_idx < (int)db.currentFilters.size();
       filter_idx++) {
    AnalyticsFilter &filter = db.currentFilters[filter_idx];

    ImGui::PushID(filter_idx);
    switch (filter.type) {
    case AnalyticsFilterType::CaptureEvent: {
      CaptureEventFilter &capture = filter.captureEvent;

      ImGui::Text("Capture Event");
      ImGui::Separator();

      ImGui::PushItemWidth(box_width);
      ImGui::DragInt("Minimum Players in Zone",
                     &capture.minNumInZone, 1, 1,
                     consts::maxTeamSize, "%d", ImGuiSliderFlags_AlwaysClamp);

      ImGui::DragInt("Zone ID", &capture.zoneIDX,
                     1, -1, ctx.data().zones.numZones - 1,
                     capture.zoneIDX == -1 ? "Any" : "%d",
                     ImGuiSliderFlags_AlwaysClamp);
      ImGui::PopItemWidth();
    } break;
    case AnalyticsFilterType::ReloadEvent: {
      ReloadEventFilter &reload = filter.reloadEvent;

      ImGui::Text("Reload Event");
      ImGui::Separator();

      ImGui::PushItemWidth(box_width);

      ImGui::DragInt("Min Mag Count",
                     &reload.minNumBulletsAtReloadTime, 1,
                     0, 100, "%d", ImGuiSliderFlags_AlwaysClamp);

      ImGui::DragInt("Max Mag Count",
                     &reload.maxNumBulletsAtReloadTime, 1,
                     0, 100, "%d", ImGuiSliderFlags_AlwaysClamp);

      ImGui::PopItemWidth();
    } break;
    case AnalyticsFilterType::KillEvent: {
      KillEventFilter &kill = filter.killEvent;

      ImGui::Text("Kill Event");
      ImGui::Separator();

      ImGui::PushItemWidth(box_width);

      aabbRegionSelector(&kill.killerRegion,
                         "Attacker");
      aabbRegionSelector(&kill.killedRegion,
                         "Target");

      ImGui::PopItemWidth();
    } break;
    case AnalyticsFilterType::PlayerShotEvent: {
      PlayerShotEventFilter &player_shot = filter.playerShotEvent;

      ImGui::Text("Player Shot Event");
      ImGui::Separator();

      ImGui::PushItemWidth(box_width);

      aabbRegionSelector(&player_shot.attackerRegion,
                         "Attacker");
      aabbRegionSelector(&player_shot.targetRegion,
                         "Target");

      ImGui::PopItemWidth();
    } break;
    case AnalyticsFilterType::PlayerInRegion: {
      PlayerInRegionFilter &player_in_region = filter.playerInRegion;

      ImGui::Text("Player In Region");
      ImGui::Separator();

      aabbRegionSelector(&player_in_region.region, "Region");

      ImGui::PushItemWidth(box_width);
      ImGui::DragInt("Min Num In Region",
                     &player_in_region.minNumInRegion, 0.5f,
                     1, 6, "%d", ImGuiSliderFlags_AlwaysClamp);
      ImGui::PopItemWidth();
    } break;
    default: MADRONA_UNREACHABLE(); break;
    }

    ImGui::Spacing();

    ImGui::PopID();
  }

  if (db.visualSelectRegion != nullptr) {
    const UserInput &input = viz->ui->inputState();
    const UserInputEvents &input_events = viz->ui->inputEvents();

    if (input_events.downEvent(InputID::MouseLeft)) {
      Camera cam = viz->flyCam;
      Vector2 mouse_pos = input.mousePosition();

      float aspect_ratio = (f32)viz->window->pixelWidth / viz->window->pixelHeight;

      float tan_fov = tanf(math::toRadians(cam.fov * 0.5f));

      Vector2 screen = {
        (2.f * mouse_pos.x) / viz->window->pixelWidth - 1.f,
        (2.f * mouse_pos.y) / viz->window->pixelHeight - 1.f,
      };

      float x_scale = tan_fov * aspect_ratio;
      float y_scale = -tan_fov;

      Vector3 dir = screen.x * x_scale * cam.right +
                    screen.y * y_scale * cam.up +
                    cam.fwd;
      dir = normalize(dir);

      if (dir.z != 0.f) {
        float t = -cam.position.z / dir.z;

        Vector3 z_plane_intersection = cam.position + t * dir;

        if (db.visualSelectRegionSelectMin) {
          db.visualSelectRegion->min.x = z_plane_intersection.x;
          db.visualSelectRegion->min.y = z_plane_intersection.y;
          db.visualSelectRegionSelectMin = false;
        } else {
          db.visualSelectRegion->max.x = z_plane_intersection.x;
          db.visualSelectRegion->max.y = z_plane_intersection.y;
          db.visualSelectRegionSelectMin = true;
          db.visualSelectRegion = nullptr;
        }
      }
    }
  }

  if (db.currentFilters.size() == 0) {
    ImGui::BeginDisabled();
  }

  ImGui::PushItemWidth(box_width);
  ImGui::DragInt("Time Window", &db.filterTimeWindow,
                 0.25f, 0, consts::episodeLen - 1, "%d",
                 ImGuiSliderFlags_AlwaysClamp);
  ImGui::PopItemWidth();

  if (ImGui::Button("Filter")) {
    db.resultsStatus.store_relaxed(1);
    sendAnalyticsThreadCmd(db, AnalyticsThreadCtrl::Filter);
  }

  if (db.currentFilters.size() == 0) {
    ImGui::EndDisabled();
  }

  ImGui::SameLine();

  if (ImGui::Button("Clear Filters")) {
    db.currentFilters.clear();
    db.filterTimeWindow = 0;
    db.addFilterType = AnalyticsFilterType::CaptureEvent;

    db.displayResults.reset();
    db.currentSelectedResult = -1;
    db.currentVizMatch = -1;
    db.currentVizMatchTimestep = -1;
    db.playLoggedStep = false;
  }

  if (filter_results_status != 0) {
    ImGui::EndDisabled();
  }

  if (filter_results_status == 2) {
    std::atomic_thread_fence(sync::acquire);

    db.displayResults = std::move(db.filteredResults);
    db.resultsStatus.store_relaxed(0);
    filter_results_status = 0;
    db.currentSelectedResult = 0;
    db.currentVizMatch = -1;
    db.currentVizMatchTimestep = -1;
    db.playLoggedStep = false;
  }

  ImGui::Spacing();
  ImGui::Spacing();
  ImGui::PushItemWidth(box_width);
  ImGui::PopItemWidth();
  ImGui::Spacing();

  u32 num_filter_results = 0;
  if (filter_results_status == 0 && db.displayResults.has_value()) {
    num_filter_results = db.displayResults->size();

    if (num_filter_results == 0) {
      db.currentSelectedResult = -1;
      db.currentVizMatch = -1;
      db.currentVizMatchTimestep = -1;
      db.playLoggedStep = false;
    }
  }

  ImGui::Text("Results");
  ImGui::Separator();

  if (num_filter_results == 0) {
    ImGui::BeginDisabled();
  } 

  ImGui::PushItemWidth(box_width);
  int result_idx = db.currentSelectedResult;
  ImGui::DragInt("Result Trajectory", &result_idx,
                 0.25f, 0, num_filter_results - 1,
                 num_filter_results == 0 ? "" :
                   (result_idx == -1 ? "" : "%d"),
                 ImGuiSliderFlags_AlwaysClamp);
  if (result_idx != db.currentSelectedResult) {
    db.currentSelectedResult = result_idx;
    db.currentVizMatchTimestep = -1;
  }

  ImGui::PopItemWidth();

  if (num_filter_results == 0) {
    ImGui::EndDisabled();
  }

  FilterResult cur_viz_result {
    .matchID = -1,
    .teamID = -1,
    .windowStart = -1,
    .windowEnd = -1,
  };

  if (num_filter_results > 0 && db.currentSelectedResult != -1) {
    cur_viz_result = (*db.displayResults)[db.currentSelectedResult];
  }

  ImGui::Spacing();

  if (cur_viz_result.matchID == -1) {
    ImGui::Text("Trajectory Match ID:");
    ImGui::Text("Trajectory Team:");
    ImGui::Text("Trajectory Start Step:");
    ImGui::Text("Trajectory End Step:");
  } else {
    ImGui::Text("Trajectory Match ID:   %ld",
                (long)cur_viz_result.matchID);
    ImGui::Text("Trajectory Team:       %s",
                cur_viz_result.teamID == 0 ? "Blue" : "Red");
    ImGui::Text("Trajectory Start Step: %ld",
                (long)cur_viz_result.windowStart);
    ImGui::Text("Trajectory End Step:   %ld",
                (long)cur_viz_result.windowEnd);
  }

  ImGui::Separator();

  ImGui::PushItemWidth(box_width);

  if (db.currentSelectedResult != -1) {
    db.currentVizMatch = cur_viz_result.matchID;
    ImGui::BeginDisabled();
  }

  {
    int viz_match = db.currentVizMatch;
    ImGui::DragInt("Visualized Match", &viz_match,
                   1, 1, db.numMatches,
                   db.currentVizMatch == -1 ? "" : "%d",
                   ImGuiSliderFlags_AlwaysClamp);

    if (viz_match != db.currentVizMatch) {
      db.currentVizMatch = viz_match;
      db.currentVizMatchTimestep = 0;
    }
  }

  if (db.currentSelectedResult != -1) {
    ImGui::EndDisabled();
  }

  if (num_filter_results == 0 || !db.trajectoriesOutFile.is_open()) {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button("Dump Trajectories")) {
    dumpTrajectories(db);
  }

  if (num_filter_results == 0 || !db.trajectoriesOutFile.is_open()) {
    ImGui::EndDisabled();
  }

  DynArray<i64> match_steps = db.currentVizMatch == -1 ?
    DynArray<i64>(0) : loadMatchStepIDs(db, db.currentVizMatch);

  if (db.currentSelectedResult != -1 &&
      db.currentVizMatchTimestep == -1 &&
      match_steps.size() > 0) {
    db.currentVizMatchTimestep = cur_viz_result.windowStart;
  }

  ImGui::BeginDisabled(match_steps.size() == 0);

  ImGui::DragInt("Visualized Timestep", &db.currentVizMatchTimestep,
                 0.5f, 0, match_steps.size() - 1,
                 db.currentVizMatchTimestep == -1 ? "" : "%d",
                 ImGuiSliderFlags_AlwaysClamp);

  ImGui::EndDisabled();

  ImGui::PopItemWidth();

  if (!db.playLoggedStep) {
    if (ImGui::Button("Play")) {
      db.playLoggedStep = true;
    }
  } else {
    if (ImGui::Button("Pause")) {
      db.playLoggedStep = false;
    }
  }

  ImGui::PushItemWidth(box_width);
  ImGui::DragInt("Replay Hz", &db.matchReplayHz,
                 1, 0, 1000, "%d",
                 ImGuiSliderFlags_AlwaysClamp);
  ImGui::PopItemWidth();

  if (db.playLoggedStep && match_steps.size() > 0) {
    auto now = std::chrono::steady_clock::now();

    // Convert to duration since the steady clock epoch
    auto duration = now - db.lastMatchReplayTick;
    u64 microseconds_elapsed = std::chrono::duration_cast<
        std::chrono::microseconds>(duration).count();

    u64 microseconds_per_step =
      (1000000 + db.matchReplayHz / 2) / db.matchReplayHz;

    if (microseconds_elapsed >= microseconds_per_step) {
      db.currentVizMatchTimestep += 1;
      db.currentVizMatchTimestep %= match_steps.size();

      db.lastMatchReplayTick = now;
    }
  }

  if (db.currentVizMatchTimestep != -1 && match_steps.size() > 0) {
    i64 step_id = match_steps[db.currentVizMatchTimestep];

    db.curSnapshot = loadStepSnapshot(
        db.hdl, db.loadMatchZoneState, db.loadStepPlayerStates, step_id);

    for (i32 i = 0; i < consts::maxTeamSize * 2; i++) {
      auto &player_snapshot = db.curSnapshot.players[i];

      Entity agent = ctx.data().agents[i];
      ctx.get<Position>(agent) = player_snapshot.pos;
      ctx.get<Rotation>(agent) = Quat::angleAxis(
          player_snapshot.yaw, math::up);
      ctx.get<Aim>(agent) = computeAim(
          player_snapshot.yaw, player_snapshot.pitch);
    }

    db.eventTeamConvexHulls = loadTeamConvexHulls(db, step_id);

    auto &zone_state = ctx.singleton<ZoneState>();
    zone_state.curZone = db.curSnapshot.curZone;
    if (db.curSnapshot.curZoneController == -1) {
      zone_state.isCaptured = false;
      zone_state.curControllingTeam = -1;
    } else {
      zone_state.isCaptured = true;
      zone_state.curControllingTeam = db.curSnapshot.curZoneController;
    }
  }

  ImGui::End();
}

static void analyticsBGThread(AnalyticsDB &db)
{
  auto filterResults =
    [&db]
  (DynArray<AnalyticsFilter> &filters, int filter_match_window)
  {
    DynArray<FilterResult> results(1000);

    assert(filters.size() < 64);

    for (int match_id = 1; match_id <= db.numMatches; match_id++) {
      std::array<FiltersMatchState, 2> match_states;
      std::array<FilterResult, 2> running_results;

      AnalyticsMatchData match_data = loadMatchData(db, match_id);

      int num_steps = (int)match_data.steps.size();

      for (int team = 0; team < 2; team++) {
        match_states[team].active = 0;

        running_results[team].matchID = match_id;
        running_results[team].teamID = team;
        running_results[team].windowStart = -100;
        running_results[team].windowEnd = -100;
      }

      for (int step_idx = 0; step_idx < num_steps; step_idx++) {
        AnalyticsStepSnapshot step_snapshot = match_data.steps[step_idx];

        for (int filter_idx = 0; filter_idx < (int)filters.size();
             filter_idx += 1) {
          AnalyticsFilter &filter = db.currentFilters[filter_idx];

          for (int team_idx = 0; team_idx < 2; team_idx++) {
            FiltersMatchState &match_state = match_states[team_idx];
            if ((match_state.active & (1 << filter_idx)) != 0) {
              if (step_idx - match_state.lastMatches[filter_idx] >
                  filter_match_window) {
                match_state.active &= ~(1 << filter_idx);
              }
            }
          }

          switch (filter.type) {
          case AnalyticsFilterType::CaptureEvent: {
            auto &capture_filter = filter.captureEvent;

            for (int capture_event_offset = 0;
                 capture_event_offset < (int)step_snapshot.numCaptureEvents;
                 capture_event_offset++) {
              int capture_event_idx = step_snapshot.captureEventsOffset +
                  capture_event_offset;
              GameEvent::Capture capture_event =
                  match_data.captureEvents[capture_event_idx];

              bool event_match = true;

              if (capture_filter.zoneIDX != -1 &&
                  capture_filter.zoneIDX != capture_event.zoneIDX) {
                event_match = false;
              }

              if (std::popcount(capture_event.inZoneMask) <
                  capture_filter.minNumInZone) {
                event_match = false;
              }

              if (event_match) {
                FiltersMatchState &match_state =
                    match_states[capture_event.captureTeam];
                match_state.active |= 1 << filter_idx;
                match_state.lastMatches[filter_idx] = step_idx;
              }
            }
          } break;
          case AnalyticsFilterType::ReloadEvent: {
            auto &reload_filter = filter.reloadEvent;

            for (int reload_event_offset = 0;
                 reload_event_offset < (int)step_snapshot.numReloadEvents;
                 reload_event_offset++) {
              int reload_event_idx = step_snapshot.reloadEventsOffset +
                reload_event_offset;
              GameEvent::Reload reload_event =
                  match_data.reloadEvents[reload_event_idx];

              bool event_match = reload_event.numBulletsAtReloadTime <=
                                   reload_filter.maxNumBulletsAtReloadTime &&
                                 reload_event.numBulletsAtReloadTime >=
                                   reload_filter.minNumBulletsAtReloadTime;

              int team = reload_event.player / consts::maxTeamSize;

              if (event_match) {
                FiltersMatchState &match_state = match_states[team];
                match_state.active |= 1 << filter_idx;
                match_state.lastMatches[filter_idx] = step_idx;
              }
            }
          } break;
          case AnalyticsFilterType::KillEvent: {
            auto &kill_filter = filter.killEvent;

            for (int kill_event_offset = 0;
                 kill_event_offset < (int)step_snapshot.numKillEvents;
                 kill_event_offset++) {
              int kill_event_idx = step_snapshot.killEventsOffset +
                kill_event_offset;

              GameEvent::Kill kill_event =
                  match_data.killEvents[kill_event_idx];

              PackedPlayerSnapshot attacker_state =
                  step_snapshot.playerData[kill_event.killer];

              PackedPlayerSnapshot target_state =
                  step_snapshot.playerData[kill_event.killed];

              bool event_match =
                  attacker_state.pos[0] >= kill_filter.killerRegion.min.x &&
                  attacker_state.pos[0] <= kill_filter.killerRegion.max.x &&
                  attacker_state.pos[1] >= kill_filter.killerRegion.min.y &&
                  attacker_state.pos[1] <= kill_filter.killerRegion.max.y &&

                  target_state.pos[0] >= kill_filter.killedRegion.min.x &&
                  target_state.pos[0] <= kill_filter.killedRegion.max.x &&
                  target_state.pos[1] >= kill_filter.killedRegion.min.y &&
                  target_state.pos[1] <= kill_filter.killedRegion.max.y;

              int team = kill_event.killer / consts::maxTeamSize;

              if (event_match) {
                FiltersMatchState &match_state = match_states[team];
                match_state.active |= 1 << filter_idx;
                match_state.lastMatches[filter_idx] = step_idx;
              }
            }
          } break;
          case AnalyticsFilterType::PlayerShotEvent: {
            auto &shot_filter = filter.playerShotEvent;

            for (int shot_event_offset = 0;
                 shot_event_offset < (int)step_snapshot.numPlayerShotEvents;
                 shot_event_offset++) {
              int shot_event_idx = step_snapshot.playerShotEventsOffset +
                shot_event_offset;

              GameEvent::PlayerShot shot_event =
                  match_data.playerShotEvents[shot_event_idx];

              PackedPlayerSnapshot attacker_state =
                  step_snapshot.playerData[shot_event.attacker];

              PackedPlayerSnapshot target_state =
                  step_snapshot.playerData[shot_event.target];

              assert((attacker_state.flags & (u8)PackedPlayerStateFlags::FiredShot) != 0); 

              bool event_match =
                  attacker_state.pos[0] >= shot_filter.attackerRegion.min.x &&
                  attacker_state.pos[0] <= shot_filter.attackerRegion.max.x &&
                  attacker_state.pos[1] >= shot_filter.attackerRegion.min.y &&
                  attacker_state.pos[1] <= shot_filter.attackerRegion.max.y &&

                  target_state.pos[0] >= shot_filter.targetRegion.min.x &&
                  target_state.pos[0] <= shot_filter.targetRegion.max.x &&
                  target_state.pos[1] >= shot_filter.targetRegion.min.y &&
                  target_state.pos[1] <= shot_filter.targetRegion.max.y;

              int team = shot_event.attacker / consts::maxTeamSize;

              if (event_match) {
                FiltersMatchState &match_state = match_states[team];
                match_state.active |= 1 << filter_idx;
                match_state.lastMatches[filter_idx] = step_idx;
              }
            }
          } break;
          case AnalyticsFilterType::PlayerInRegion: {
            auto &in_region_filter = filter.playerInRegion;

            for (int team = 0; team < 2; team++) {
              i32 num_in_region = 0;

              for (int team_offset = 0; team_offset < consts::maxTeamSize;
                   team_offset++) {
                int player_idx = team * consts::maxTeamSize + team_offset;

                PackedPlayerSnapshot player_state =
                    step_snapshot.playerData[player_idx];

                bool player_match =
                    player_state.pos[0] >= in_region_filter.region.min.x &&
                    player_state.pos[0] <= in_region_filter.region.max.x &&
                    player_state.pos[1] >= in_region_filter.region.min.y &&
                    player_state.pos[1] <= in_region_filter.region.max.y;

                if (player_match) {
                  num_in_region += 1;
                }
              }

              if (num_in_region >= in_region_filter.minNumInRegion) {
                FiltersMatchState &match_state = match_states[team];
                match_state.active |= 1 << filter_idx;
                match_state.lastMatches[filter_idx] = step_idx;
                break;
              }
            }
          } break;
          default: MADRONA_UNREACHABLE(); break;
          }
        }

        for (int team_idx = 0; team_idx < 2; team_idx++) {
          FiltersMatchState &match_state = match_states[team_idx];
          if (std::popcount(match_state.active) != filters.size()) {
            continue;
          }

          int window_start = match_state.lastMatches[0];
          int window_end = window_start;
          for (int filter_idx = 1; filter_idx < filters.size(); filter_idx++) {
            int filter_match_step = match_state.lastMatches[filter_idx];

            if (filter_match_step < window_start) {
              window_start = filter_match_step;
            }

            if (filter_match_step > window_end) {
              window_end = filter_match_step;
            }
          }

          {
            int filter_time_spread = window_end - window_start;
            int extra_window = filter_match_window - filter_time_spread;

            window_start -= extra_window;
            window_end += extra_window;
          }

          FilterResult &team_running_results = running_results[team_idx];

          int running_window_start = team_running_results.windowStart;
          int running_window_end = team_running_results.windowEnd;

          if (window_start <= running_window_end + 1 &&
              running_window_start <= window_end + 1) {
            if (window_start < running_window_start) {
              running_window_start = window_start;
            }

            if (window_end > running_window_end) {
              running_window_end = window_end;
            }

            team_running_results.windowStart = running_window_start;
            team_running_results.windowEnd = running_window_end;
          } else {
            if (team_running_results.windowStart != -100) {
              results.push_back(team_running_results);
            }
            team_running_results.windowStart = window_start;
            team_running_results.windowEnd = window_end;
          }
        }
      }

      for (int team = 0; team < 2; team++) {
        if (running_results[team].windowStart != -100) {
          results.push_back(running_results[team]);
        }
      }
    }

    return results;
  };

  while (true) {
    db.threadCtrl.wait<sync::acquire>(0);
    auto ctrl = (AnalyticsThreadCtrl)db.threadCtrl.exchange<sync::relaxed>(
      (u32)AnalyticsThreadCtrl::Idle);
    db.threadCtrl.notify_one();

    switch (ctrl) {
    case AnalyticsThreadCtrl::Idle: continue;
    case AnalyticsThreadCtrl::Exit: return;
    case AnalyticsThreadCtrl::Filter: {
      printf("Filtering\n");

      db.filteredResults = filterResults(
          db.currentFilters, db.filterTimeWindow);

      printf("Returned results %ld\n", (long)db.filteredResults->size());
      db.resultsStatus.store_release(2);
    } break;
    default: MADRONA_UNREACHABLE();
    }
  }
}
#endif

static void postDeviceCreateInit(VizState *viz, VizConfig cfg, 
                                 void (*cb)(VizState *, void *), void *data_ptr);

void init(const VizConfig &cfg, void (*cb)(VizState *, void *), void *data_ptr)
{
  VizState *viz = new VizState {};

  viz->ui = UISystem::init(UISystem::Config {
    .enableValidation = false,
    .errorsAreFatal = true,
  });

  viz->window = viz->ui->createMainWindow(
    #ifdef EMSCRIPTEN
      "#canvas",
    #else
      "MadronaMPEnv",
    #endif
      cfg.windowWidth, cfg.windowHeight,
      WindowInitFlags::None);
  
  viz->gpuAPI = viz->ui->gpuLib();

  struct PostDeviceCreateData {
    VizState *viz;
    void *data_ptr;
    void (*cb)(VizState *, void *);
    VizConfig cfg;
  };
  auto *cb_data = new PostDeviceCreateData { viz, data_ptr, cb, std::move(cfg) };

  viz->gpuAPI->createDeviceAsync(0, {viz->window->surface},
    [](GPUDevice *gpu, void *data_ptr) {
      PostDeviceCreateData *cb_data = (PostDeviceCreateData *)data_ptr;
      VizState *viz = cb_data->viz;
      void *user_data_ptr = cb_data->data_ptr;
      void (*cb)(VizState *, void *) = cb_data->cb;
      VizConfig cfg = std::move(cb_data->cfg);
      delete cb_data;

      viz->gpu = gpu;

      postDeviceCreateInit(viz, cfg, cb, user_data_ptr);
    }, cb_data);
}

static void postDeviceCreateInit(VizState *viz, VizConfig cfg, 
                                 void (*cb)(VizState *, void *), void *data_ptr)
{
  GPUDevice *gpu = viz->gpu;

  SwapchainProperties swapchain_properties;
  viz->swapchain = gpu->createSwapchain(
      viz->window->surface, { SwapchainFormat::SDR_SRGB, SwapchainFormat::SDR_UNorm },
      &swapchain_properties);
  viz->swapchainFormat = swapchain_properties.format;

  viz->mainQueue = gpu->getMainQueue();

  viz->persistentAlloc = {};

  std::filesystem::path shader_dir = MADRONA_MP_ENV_OUT_DIR "shaders";
  switch (viz->gpuAPI->backendShaderByteCodeType()) {
    case ShaderByteCodeType::SPIRV: shader_dir /= "spirv"; break;
    case ShaderByteCodeType::WGSL: shader_dir /= "wgsl"; break;
    case ShaderByteCodeType::MTLLib: shader_dir /= "mtl"; break;
    case ShaderByteCodeType::DXIL: shader_dir /= "dxil"; break;
    default: MADRONA_UNREACHABLE(); break;
  }

  viz->vizShaders.load(viz->persistentAlloc, (shader_dir / "mpenv_shaders.shader_blob").c_str());

  viz->depthAttachment = gpu->createTexture({
    .format = TextureFormat::Depth32_Float,
    .width = (u16)viz->window->pixelWidth,
    .height = (u16)viz->window->pixelHeight,
    .usage = TextureUsage::DepthAttachment,
  });

  viz->sceneColor = gpu->createTexture({
    .format = swapchain_properties.format,
    .width = (u16)viz->window->pixelWidth,
    .height = (u16)viz->window->pixelHeight,
    .usage = TextureUsage::ColorAttachment | TextureUsage::ShaderSampled,
    });

  viz->sceneDepth = gpu->createTexture({
      .format = TextureFormat::Depth32_Float,
      .width = (u16)viz->window->pixelWidth,
      .height = (u16)viz->window->pixelHeight,
      .usage = TextureUsage::DepthAttachment | TextureUsage::ShaderSampled,
    });

  viz->offscreenPassInterface = gpu->createRasterPassInterface({
      .uuid = "offscreen_raster_pass"_to_uuid,
      .depthAttachment = {
          .format = TextureFormat::Depth32_Float,
          .loadMode = AttachmentLoadMode::Clear,
      },
      .colorAttachments = {
          {
              .format = swapchain_properties.format,
              .loadMode = AttachmentLoadMode::Clear,
          },
      },
    });

  viz->offscreenPass = gpu->createRasterPass({
      .interface = viz->offscreenPassInterface,
      .depthAttachment = viz->sceneDepth,
      .colorAttachments = { viz->sceneColor },
    });

  viz->mainmenuPass = gpu->createRasterPass({
    .interface = viz->offscreenPassInterface,
    .depthAttachment = viz->sceneDepth,
    .colorAttachments = { viz->swapchain.proxyAttachment() },
  });

  ImGuiSystem::init(viz->ui, gpu, viz->mainQueue,
      viz->offscreenPassInterface, shader_dir.c_str(), DATA_DIR "imgui_font.ttf", 12.f);

  viz->heatmapTexture = {};
  gpu->waitUntilWorkFinished(viz->mainQueue);

  viz->enc = gpu->createCommandEncoder(viz->mainQueue);

  viz->curWorld = 0;
  viz->curView = 0;
  viz->numWorlds = (i32)cfg.numWorlds;
  viz->numViews = (i32)cfg.numViews;
  viz->teamSize = (i32)cfg.teamSize;
  viz->doAI[0] = cfg.doAITeam1;
  viz->doAI[1] = cfg.doAITeam2;

  viz->simTickRate = 0;
  viz->recordedDataPath = cfg.recordedDataPath;

  viz->globalParamBlockType = gpu->createParamBlockType({
    .uuid = "global_pb"_to_uuid,
    .buffers = {
      {
        .type = BufferBindingType::Uniform,
        .shaderUsage = ShaderStage::Vertex | ShaderStage::Fragment,
      },
    },
  });

  viz->mapGeoParamBlockType = gpu->createParamBlockType({
    .uuid = "map_geometry_pb"_to_uuid,
    .buffers = {
      {
        .type = BufferBindingType::Storage,
        .shaderUsage = ShaderStage::Vertex | ShaderStage::Fragment,
      },
      {
        .type = BufferBindingType::Storage,
        .shaderUsage = ShaderStage::Vertex | ShaderStage::Fragment,
      },
    }
  });

  viz->globalPassDataBuffer = gpu->createBuffer({
    .numBytes = sizeof(GlobalPassData),
    .usage = BufferUsage::ShaderUniform | BufferUsage::CopyDst,
  });

  viz->globalParamBlock = gpu->createParamBlock({
    .typeID = viz->globalParamBlockType,
    .buffers = {
      { .buffer = viz->globalPassDataBuffer, .numBytes = sizeof(GlobalPassData) },
    },
  });

  viz->postEffectParamBlockType = gpu->createParamBlockType({
    .uuid = "post_pb"_to_uuid,
    .buffers = {
      {
        .type = BufferBindingType::Uniform,
        .shaderUsage = ShaderStage::Vertex | ShaderStage::Fragment,
      },
    },
    });

  viz->postEffectDataBuffer = gpu->createBuffer({
    .numBytes = sizeof(PostEffectData),
    .usage = BufferUsage::ShaderUniform | BufferUsage::CopyDst,
    });

  viz->sceneSampler = gpu->createSampler({
    .addressMode = SamplerAddressMode::Clamp,
    .anisotropy = 1,
    });

  viz->depthSampler = gpu->createSampler({
    .addressMode = SamplerAddressMode::Clamp,
    .mipmapFilterMode = SamplerFilterMode::Nearest,
    .magnificationFilterMode = SamplerFilterMode::Nearest,
    .minificationFilterMode = SamplerFilterMode::Nearest,
    .anisotropy = 1,
  });

  using enum VertexFormat;

  viz->mapShader = gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(ShaderID::Map),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = viz->offscreenPassInterface,
      .paramBlockTypes = { viz->globalParamBlockType, viz->mapGeoParamBlockType },
      .numPerDrawBytes = sizeof(MapPerDraw),
      .rasterConfig = {
        .depthCompare = DepthCompare::GreaterOrEqual,
        .cullMode = CullMode::None,
      },
    });

  viz->renderableObjectsShader = gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(ShaderID::Objects),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = viz->offscreenPassInterface,
      .paramBlockTypes = { viz->globalParamBlockType },
      .numPerDrawBytes = sizeof(OpaqueGeoPerDraw),
      .vertexBuffers = {{ 
        .stride = sizeof(OpaqueGeoVertex), .attributes = {
          { .offset = offsetof(OpaqueGeoVertex, pos), .format = Vec3_F32 },
          { .offset = offsetof(OpaqueGeoVertex, normal),  .format = Vec3_F32 },
          { .offset = offsetof(OpaqueGeoVertex, uv), .format = Vec2_F32 },
        }
      }},
      .rasterConfig = {
        .depthCompare = DepthCompare::GreaterOrEqual,
      },
    });

  viz->agentShader = gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(ShaderID::Agent),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = viz->offscreenPassInterface,
      .paramBlockTypes = { viz->globalParamBlockType },
      .numPerDrawBytes = sizeof(AgentPerDraw),
      .vertexBuffers = {{ 
        .stride = sizeof(OpaqueGeoVertex), .attributes = {
          { .offset = offsetof(OpaqueGeoVertex, pos), .format = Vec3_F32 },
          { .offset = offsetof(OpaqueGeoVertex, normal),  .format = Vec3_F32 },
          { .offset = offsetof(OpaqueGeoVertex, uv), .format = Vec2_F32 },
        }
      }},
      .rasterConfig = {
        //.depthCompare = DepthCompare::Disabled,
        .writeDepth = true,
      },
    });

  viz->goalRegionsShader = gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(ShaderID::GoalRegions),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = viz->offscreenPassInterface,
      .paramBlockTypes = { viz->globalParamBlockType },
      .numPerDrawBytes = sizeof(GoalRegionPerDraw),
      .rasterConfig = {
        .depthCompare = DepthCompare::GreaterOrEqual,
        .writeDepth = true,
        .blending = { BlendingConfig::additiveDefault() },
      },
    });

  viz->goalRegionsShaderWireframe = gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(ShaderID::GoalRegions),
      .vertexEntry = "vertMainWireframe",
      .fragmentEntry = "fragMainWireframe",
      .rasterPass = viz->offscreenPassInterface,
      .paramBlockTypes = { viz->globalParamBlockType },
      .numPerDrawBytes = sizeof(GoalRegionPerDraw),
      .rasterConfig = {
        .depthCompare = DepthCompare::GreaterOrEqual,
        .writeDepth = false,
        .cullMode = CullMode::None,
      },
    });

  viz->goalRegionsShaderWireframeNoDepth = gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(ShaderID::GoalRegions),
      .vertexEntry = "vertMainWireframe",
      .fragmentEntry = "fragMainWireframeNoDepth",
      .rasterPass = viz->offscreenPassInterface,
      .paramBlockTypes = { viz->globalParamBlockType },
      .numPerDrawBytes = sizeof(GoalRegionPerDraw),
      .rasterConfig = {
        .depthCompare = DepthCompare::Disabled,
        .writeDepth = false,
        .cullMode = CullMode::None,
        .blending = { BlendingConfig::additiveDefault() },
      },
    });

  viz->analyticsTeamHullShader = gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(ShaderID::TeamHull),
      .vertexEntry = "vertMain",
      .fragmentEntry = "triFrag",
      .rasterPass = viz->offscreenPassInterface,
      .paramBlockTypes = { viz->globalParamBlockType },
      .numPerDrawBytes = sizeof(AnalyticsTeamHullPerDraw),
      .vertexBuffers = {{
        .stride = sizeof(Vector3), .attributes = {
          { .offset = 0, .format = VertexFormat::Vec3_F32 },
        },
      }},
      .rasterConfig = {
        .depthCompare = DepthCompare::Disabled,
        .writeDepth = false,
        .blending = { BlendingConfig::additiveDefault() },
      },
    });

  viz->agentPathsParamBlockType = gpu->createParamBlockType({
    .uuid = "agent_paths_pb"_to_uuid,
    .buffers = {
      {
        .type = BufferBindingType::Storage,
        .shaderUsage = ShaderStage::Vertex,
      },
    },
  });
  viz->agentPathsShader = gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(ShaderID::Paths),
      .vertexEntry = "pathVert",
      .fragmentEntry = "pathFrag",
      .rasterPass = viz->offscreenPassInterface,
      .paramBlockTypes = {
        viz->globalParamBlockType,
        viz->agentPathsParamBlockType,
      },
      .numPerDrawBytes = sizeof(Vector4),
      .rasterConfig = {
        .depthBias = 200000,
        .depthBiasSlope = 2e-2f,
        .depthBiasClamp = 1e-4f,
        .cullMode = CullMode::None,
        .blending = { BlendingConfig::additiveDefault() },
      },
    });

  viz->shotVizParamBlockType = gpu->createParamBlockType({
    .uuid = "shot_vz_pb"_to_uuid,
    .buffers = {
      { 
        .type = BufferBindingType::Storage,
        .shaderUsage = ShaderStage::Vertex,
      },
    },
  });

  viz->shotVizShader = gpu->createRasterShader({
      .byteCode = viz->vizShaders.getByteCode(ShaderID::ShotViz),
      .vertexEntry = "vertMain",
      .fragmentEntry = "fragMain",
      .rasterPass = viz->offscreenPassInterface,
      .paramBlockTypes = {
        viz->globalParamBlockType,
        viz->shotVizParamBlockType,
      },
      .numPerDrawBytes = 0,
      .rasterConfig = {
        .writeDepth = false,
        .cullMode = CullMode::None,
        .blending = { BlendingConfig::additiveDefault() },
      },
    });


  loadAssets(viz);

  if (cfg.trajectoryDBPath) {
    viz->trajectoryDB = openTrajectoryDB(cfg.trajectoryDBPath);
    assert(viz->trajectoryDB);
  }

#ifdef DB_SUPPORT
  if (cfg.analyticsDBPath != nullptr) {
    loadAnalyticsDB(viz, cfg);
  }
#endif

  if (cfg.skipMainMenu) {
    viz->mainMenu = false;
    viz->gameRunning = true;
    viz->simTickRate = 20;
  }

  cb(viz, data_ptr);
}

void shutdown(VizState *viz)
{
  if (viz->trajectoryDB) {
    closeTrajectoryDB(viz->trajectoryDB);
  }

#ifdef DB_SUPPORT
  if (viz->db.hdl != nullptr) {
    unloadAnalyticsDB(viz->db);
  }
#endif

  GPUDevice *gpu = viz->gpu;

  gpu->waitUntilWorkFinished(viz->mainQueue);
  gpu->waitUntilIdle();

  if (!viz->mapBuffer.null()) {
    gpu->destroyBuffer(viz->mapBuffer);
  }

  for (AssetGroup &group : viz->objectAssetGroups) {
    gpu->destroyBuffer(group.geometryBuffer);
  }

  gpu->destroyCommandEncoder(viz->enc);

  gpu->destroyRasterShader(viz->agentPathsShader);
  gpu->destroyParamBlockType(viz->agentPathsParamBlockType);

  gpu->destroyRasterShader(viz->shotVizShader);
  gpu->destroyParamBlockType(viz->shotVizParamBlockType);

  gpu->destroyRasterShader(viz->analyticsTeamHullShader);

  gpu->destroyRasterShader(viz->goalRegionsShaderWireframeNoDepth);
  gpu->destroyRasterShader(viz->goalRegionsShaderWireframe);
  gpu->destroyRasterShader(viz->goalRegionsShader);

  gpu->destroyRasterShader(viz->agentShader);

  gpu->destroyRasterShader(viz->renderableObjectsShader);
  gpu->destroyRasterShader(viz->mapShader);

  ImGuiSystem::shutdown(gpu);

  gpu->destroyParamBlock(viz->globalParamBlock);
  gpu->destroyBuffer(viz->globalPassDataBuffer);
  gpu->destroyParamBlockType(viz->globalParamBlockType);
  gpu->destroyBuffer(viz->postEffectDataBuffer);
  gpu->destroyParamBlockType(viz->postEffectParamBlockType);

  gpu->destroyRasterPass(viz->mainmenuPass);
  gpu->destroyRasterPass(viz->offscreenPass);

  gpu->destroyRasterPassInterface(viz->offscreenPassInterface);

  viz->ssaoPass.Destroy();
  viz->finalPass.Destroy();

  gpu->destroyTexture(viz->depthAttachment);

  gpu->destroySwapchain(viz->swapchain);
  viz->ui->destroyMainWindow();

  viz->gpuAPI->destroyDevice(gpu);

  viz->ui->shutdown();

  delete viz;
}

void initWorld(Context &ctx, VizState *viz)
{
  auto &viz_world = ctx.singleton<VizWorld>();
  viz_world.viz = viz;

  viz_world.renderableObjectsQuery = ctx.query<Position, Rotation, Scale, ObjectID>();
}

static Vector2 getMouseDelta(const UserInput &input)
{
  auto [x, y] = input.mouseDelta();
  return { x, y };
}

static Vector2 getMouseScroll(const UserInputEvents &input_events)
{
  auto [x, y] = input_events.mouseScroll();
  return { x, y };
}

static void handleCamera(VizState *viz, float delta_t)
{
  Camera &cam = viz->flyCam;

  // Toggle this to switch the free camera between oribit and fly.
  cam.orbit = true;

  Vector3 translate = Vector3::zero();

  const UserInput &input = viz->ui->inputState();
  const UserInputEvents &input_events = viz->ui->inputEvents();

  Vector2 mouse_scroll = getMouseScroll(input_events);

  if (cam.orbit) {
    // Rotate around the focus point.
    cam.fine_aim = false;
    if (input.isDown(InputID::MouseRight) ||
      input.isDown(InputID::Shift)) {
      viz->ui->enableRawMouseInput(viz->window);

      Vector2 mouse_delta = getMouseDelta(input);

      cam.azimuth -= mouse_delta.y * MOUSE_SPEED * delta_t;
      cam.heading += mouse_delta.x * MOUSE_SPEED * delta_t;
      cam.azimuth = fmin(fmax(cam.azimuth, -PI * 0.49f), PI * 0.49f);
      while (cam.heading > PI)
        cam.heading -= PI * 2.0f;
      while (cam.heading < -PI) 
        cam.heading += PI * 2.0f;
    } else {
      viz->ui->disableRawMouseInput(viz->window);
    }

    if (mouse_scroll.y != 0.f) {
      float zoomChange = -mouse_scroll.y * MOUSE_SCROLL_SPEED * delta_t;
      if (zoomChange < 0.0f)
        cam.zoom /= 1.0f - zoomChange;
      if (zoomChange > 0.0f)
        cam.zoom *= 1.0f + zoomChange;
    }

    // Move the focus point.
    if (input.isDown(InputID::W)) {
      translate += normalize(cross(Vector3(0.0f, 0.0f, 1.0f), cam.right));
    }
    if (input.isDown(InputID::A)) {
      translate -= cam.right;
    }
    if (input.isDown(InputID::S)) {
      translate -= normalize(cross(Vector3(0.0f, 0.0f, 1.0f), cam.right));
    }
    if (input.isDown(InputID::D)) {
      translate += cam.right;
    }

    // Convert from heading / azimuth to transform.
    cam.fwd = Vector3(cos(cam.heading) * cos(cam.azimuth), sin(cam.heading) * cos(cam.azimuth), -sin(cam.azimuth));
    cam.up = Vector3(0.0f, 0.0f, 1.0f);
    cam.right = normalize(cross(cam.fwd, cam.up));
    cam.up = normalize(cross(cam.right, cam.fwd));
    cam.fine_aim = false;

    cam.target += translate * viz->cameraMoveSpeed * delta_t;
    cam.position = cam.target - cam.fwd * (cam.mapMax - cam.mapMin).length() * cam.zoom;
  }
  else
  {

    if (input.isDown(InputID::MouseRight) ||
      input.isDown(InputID::Shift)) {
      viz->ui->enableRawMouseInput(viz->window);
      cam.fine_aim = false;

      Vector2 mouse_delta = getMouseDelta(input);

      auto around_right = Quat::angleAxis(
        -mouse_delta.y * MOUSE_SPEED * delta_t, cam.right);

      auto around_up = Quat::angleAxis(
        -mouse_delta.x * MOUSE_SPEED * delta_t, math::up);

      auto rotation = (around_up * around_right).normalize();

      cam.up = rotation.rotateVec(cam.up);
      cam.fwd = rotation.rotateVec(cam.fwd);
      cam.right = rotation.rotateVec(cam.right);

      if (input.isDown(InputID::W)) {
        translate += cam.fwd;
      }

      if (input.isDown(InputID::A)) {
        translate -= cam.right;
      }

      if (input.isDown(InputID::S)) {
        translate -= cam.fwd;
      }

      if (input.isDown(InputID::D)) {
        translate += cam.right;
      }
    }
    else {
      viz->ui->disableRawMouseInput(viz->window);

      if (input.isDown(InputID::W)) {
        translate += cam.up;
      }

      if (input.isDown(InputID::A)) {
        translate -= cam.right;
      }

      if (input.isDown(InputID::S)) {
        translate -= cam.up;
      }

      if (input.isDown(InputID::D)) {
        translate += cam.right;
      }
    }

    cam.position += translate * viz->cameraMoveSpeed * delta_t;

  }

  //printf("\n"
  //       "(%f %f %f)\n"
  //       "(%f %f %f)\n"
  //       "(%f %f %f)\n"
  //       "(%f %f %f)\n\n",
  //  cam.position.x, cam.position.y, cam.position.z,
  //  cam.right.x, cam.right.y, cam.right.z,
  //  cam.up.x, cam.up.y, cam.up.z,
  //  cam.fwd.x, cam.fwd.y, cam.fwd.z);
}

float smoothStep(int step, int range)
{
    float t = float(step) / float(range);
    return (3.0f - 2.0f * t) * t * t;
}

static std::vector<PvPDiscreteAction> actions;
static std::vector<PvPAimAction> aimActions;
void planAI(Engine& ctx, VizState* viz, int world, int player)
{
  if (actions.empty())
    return;

  // Get the Agent data.
  Entity agent = ctx.data().agents[player];
  const Magazine &magazine = ctx.get<Magazine>(agent);
  const Navmesh &navmesh = ctx.singleton<LevelData>().navmesh;
  Position agent_pos = ctx.get<Position>(agent);
  Aim agent_aim = ctx.get<Aim>(agent);
  const OpponentsVisibility& enemies = ctx.get<OpponentsVisibility>(agent);
  const FwdLidar& fwd_lidar = ctx.get<FwdLidar>(agent);
  const Zones& zones = ctx.data().zones;
  const ZoneState &zone_mode_state = ctx.singleton<ZoneState>();

  int move_amount = std::rand() % 2;
  int move_angle = std::rand() % 2;
  int r_yaw = std::rand() % 5;
  int r_pitch = 2;// std::rand() % 5;
  int r = magazine.numBullets == 0 ? 1 : 0;
  int stand = std::rand() % 2;

  // If we can see an enemy, fire.
  int f = 0;
  int numAgents = ctx.data().numAgents;
  for (int i = 0; i < numAgents / 2; i++)
  {
    if (enemies.canSee[i])
      f = 1;
  }


  // If there's an active zone, move to it.
  int zoneIdx = zone_mode_state.curZone;
  assert(zoneIdx >= 0 && zoneIdx < (int)zones.numZones);

  // Get a target point in the zone.
  Vector3 center = zones.bboxes[zoneIdx].centroid();

  //Entity agent1 = ctx.data().agents[0];
  //  const GlobalPosObservation& c = ctx.get<GlobalPosObservation>(agent1);
  //  center = Vector3(c.globalX, c.globalY, 0.0f);

  // Pathfind to the target point.
  Vector3 pos = Vector3(agent_pos.x, agent_pos.y, 0.0f);
  center = NavUtils::PathfindToPoint(navmesh, pos, center);
  /*int navTri = NavUtils::NearestNavTri(navmesh, center);
    if (navTri >= 0)
    center = NavUtils::CenterOfTri(navmesh, navTri);*/

  // Turn to face the target, and if we're facing the right way, move forward.
  center.z = 0.0f;
  Vector3 fwd = Vector3(-sin(agent_aim.yaw), cos(agent_aim.yaw));
  Vector3 tgtDir = (center - pos).normalize();
  move_amount = dot(fwd, tgtDir) > 0.6f ? 1 : 0;
  r_yaw = cross(fwd, tgtDir).z < 0.0f ? 0 + move_amount : 4 - move_amount;
  move_amount *= 2;
  move_angle = 0;
  stand = 0;

  // If we're facing a wall, and it's right in our face, push off of it.
  float collisionAng = 0.0f;
  float collisionNorm = 0.0f;
  for (int y = 0; y < consts::fwdLidarHeight; y++)
  {
      for (int x = 0; x < consts::fwdLidarWidth; x++)
      {
          if (fwd_lidar.data[y][x].depth < 16.0f)
          {
              collisionNorm++;
              collisionAng += x;
          }
      }
  }

  // If anything invades our personal space, backpedal from it.
  if (collisionNorm > 0.0f)
  {
      collisionAng /= collisionNorm;
      // The range should be devided into 8 equal segments, but we don't care about the back ones, and we only see half of the extreme side ones.
      // So we look at 16 half-segments...
      move_amount = 1;
      switch ((int)(collisionAng / consts::fwdLidarWidth * 8.0f))
      {
      case 0:
          move_angle = 2;
          break;
	  case 1:
      case 2:
		  move_angle = 3;
          break;
	  case 3:
      case 4:
		  move_angle = 4;
          move_amount = 2;
          break;
	  case 5:
      case 6:
		  move_angle = 5;
          break;
	  case 7:
		  move_angle = 6;
          break;
      }
  }
  
  // Don't try to fire while reloading, and don't try to turn while firing.
  if (r)
      f = 0;
  if (f)
      r_yaw = 2;

  actions[world * viz->numViews + player] = PvPDiscreteAction{
    .moveAmount = move_amount,
    .moveAngle = move_angle,
    .fire = r > 0 ? 2 : f,
    .stand = stand,
  };

  aimActions[world * viz->numViews + player] = PvPAimAction {
    .yaw = (float)r_yaw,
    .pitch = (float)r_pitch,
  };
}

void doAI(VizState* viz, Manager& mgr, int world, int player)
{
	if (actions.empty())
		return;
    mgr.setPvPAction(
        world, player,
        actions[world * viz->numViews + player],
        aimActions[world * viz->numViews + player],
        {});
}

static void populateAgentTrajectories(VizState *viz, Manager &mgr)
{
  Engine &ctx = mgr.getWorldContext(viz->curWorld);

  const auto &query = ctx.query<Position, TeamInfo, Done, CombatState>();

  MatchInfo &match_info = ctx.singleton<MatchInfo>();

  ctx.iterateQuery(query,
    [&]
  (Vector3 pos, TeamInfo team_info, Done &done, CombatState &combat_state)
  {
    i32 agent_idx = team_info.team * viz->teamSize + team_info.offset;
    AgentRecentTrajectory &traj = viz->agentTrajectories[agent_idx];

    if (done.v || combat_state.wasKilled || match_info.curStep == 1) {
      traj.curOffset = 0;
    }

    traj.points[traj.curOffset++ % traj.points.size()] = pos;
  });

}

static void trackHumanTrace(VizState *viz, Manager &mgr,
                            float mouse_yaw_delta, float mouse_pitch_delta)
{
  if (!viz->trajectoryDB) {
    return;
  }

  if (viz->curControl == 0) {
    viz->humanTrace.clear();
    return;
  }

  if (viz->simEventsState.downEvent(InputID::T)) {
    u32 num_timesteps = (u32)viz->humanTrace.size();
    printf("Saving trajectory %d\n", num_timesteps);

    for (u32 cur_offset = 0; cur_offset < num_timesteps; cur_offset++) {
      u32 start_offset = cur_offset;
      for (; cur_offset < num_timesteps; cur_offset++) {
        if (viz->humanTrace[cur_offset].combatState.wasKilled) {
          break;
        }
      }

      if (cur_offset - start_offset > 0) {
        Span<AgentTrajectoryStep> trace_subset(
            viz->humanTrace.data() + start_offset, cur_offset - start_offset);

        saveTrajectory(viz->trajectoryDB, TrajectoryType::Human, -1, "", trace_subset);
      }
    }
    viz->humanTrace.clear();
  }

  Engine &ctx = mgr.getWorldContext(viz->curWorld);

  i32 agent_idx = viz->curControl - 1;
  Entity agent = ctx.data().agents[agent_idx];

  Aim aim = ctx.get<Aim>(agent);

  AgentTrajectoryStep snapshot;
  snapshot.pos = ctx.get<Position>(agent);
  snapshot.yaw = aim.yaw;
  snapshot.pitch = aim.pitch;

  snapshot.combatState = ctx.get<CombatState>(agent);

  snapshot.discreteAction = ctx.get<PvPDiscreteAction>(agent);
  snapshot.continuousAimAction = {
    .yaw = mouse_yaw_delta,
    .pitch = mouse_pitch_delta,
  };

  snapshot.selfObs = ctx.get<SelfObservation>(agent);
  snapshot.teammateObs = ctx.get<TeammateObservation>(agent);
  snapshot.opponentObs = ctx.get<OpponentObservation>(agent);
  snapshot.opponentLastKnownObs = ctx.get<OpponentLastKnownObservations>(agent);

  viz->humanTrace.push_back(snapshot);
}

static bool tick(VizState *viz, Manager &mgr)
{
  bool running = true;

  auto action_tensor = mgr.pvpDiscreteActionTensor();
  GPUDevice *gpu = viz->gpu;
  gpu->waitUntilReady(viz->mainQueue);

  auto [swapchain_tex, swapchain_status] =
    gpu->acquireSwapchainImage(viz->swapchain);
  assert(swapchain_status == SwapchainStatus::Valid);

  {
    bool should_exit = viz->ui->processEvents();
    if (should_exit || (viz->window->state & WindowState::ShouldClose) != 
        WindowState::None) {
      running = false;
    }
  }

  viz->simEventsState.merge(viz->ui->inputEvents());

  auto cur_frame_start_time = std::chrono::steady_clock::now();

  float frontend_delta_t;
  {
    std::chrono::duration<float> duration =
        cur_frame_start_time - viz->last_frontend_tick_time;
    frontend_delta_t = duration.count();
  }
  viz->last_frontend_tick_time = cur_frame_start_time;

  if (viz->curControl == 0) {
    handleCamera(viz, frontend_delta_t);
  } else {
    viz->ui->enableRawMouseInput(viz->window);
    const UserInput &input = viz->ui->inputState();
    Vector2 mouse_move = getMouseDelta(input);
    mouse_move.x /= (0.5f * viz->window->pixelWidth);
    mouse_move.y /= (0.5f * viz->window->pixelHeight);

    Engine &ctx = mgr.getWorldContext(viz->curWorld);

    Entity agent = ctx.data().agents[viz->curControl - 1];

    const float mouse_max_move = 1000.f;
    const float mouse_accelleration = 0.8f;
    const float fine_aim_multiplier = 0.3f;

    Vector2 mouse_delta = mouse_move * viz->mouseSensitivity * frontend_delta_t;
    // If we're holding right-mouse, do fine aim.
    viz->flyCam.fine_aim = false;
    if (input.isDown(InputID::MouseRight)) {
      mouse_delta *= fine_aim_multiplier;
      viz->flyCam.fine_aim = true;
    }

    // Mouse accelleration.
    static Vector2 prev_mouse_delta = { 0.f, 0.f };
    float mouse_delta_len = fmaxf(mouse_delta.length(), 0.01f);
    mouse_delta = mouse_delta / mouse_delta_len;
    mouse_delta_len = fminf(mouse_delta_len + fmaxf(0.0f, prev_mouse_delta.dot(mouse_delta)) * mouse_accelleration, mouse_max_move);
    mouse_delta *= mouse_delta_len;
    prev_mouse_delta = mouse_delta;

    viz->mouse_yaw_delta -= mouse_delta.x;
    viz->mouse_pitch_delta -= mouse_delta.y;

    Aim aim = ctx.get<Aim>(agent);
    aim.yaw -= mouse_delta.x;
    aim.pitch -= mouse_delta.y;

    ctx.get<Aim>(agent) = computeAim(aim.yaw, aim.pitch);
    ctx.get<Rotation>(agent) = Quat::angleAxis(aim.yaw, math::up);
  }

  auto sim_delta_t = std::chrono::duration<float>(1.f / (float)viz->simTickRate);

  if (cur_frame_start_time - viz->last_sim_tick_time >= sim_delta_t) {
    UserInput &input = viz->ui->inputState();
    UserInputEvents &input_events = viz->simEventsState;
    //world_input_fn(world_input_data, vizCtrl.worldIdx, user_input);

    if (viz->curControl != 0) {
      int32_t x = 0;
      int32_t y = 0;
      int32_t r_yaw = consts::discreteAimNumYawBuckets / 2;
      int32_t r_pitch = consts::discreteAimNumPitchBuckets / 2;
      int32_t f = 0;
      int32_t r = 0;

      int32_t stand;
      {
        PvPDiscreteAction action_readback;
        PvPDiscreteAction *src_action =
            (PvPDiscreteAction *)action_tensor.devicePtr();
        src_action += viz->curWorld * viz->numViews + viz->curControl - 1;
        memcpy(&action_readback, src_action, sizeof(PvPDiscreteAction));
        stand = action_readback.stand;
      }

      bool shift_pressed = input.isDown(InputID::Shift);

      if (input.isDown(InputID::R)) {
        r = 1;
      }

      if (input.isDown(InputID::W)) {
        y += 1;
      }
      if (input.isDown(InputID::S)) {
        y -= 1;
      }

      if (input.isDown(InputID::D)) {
        x += 1;
      }
      if (input.isDown(InputID::A)) {
        x -= 1;
      }

      if (input.isDown(InputID::F) ||
          input.isDown(InputID::MouseLeft) ||
          input_events.downEvent(InputID::MouseLeft)) {
        f = 1;
      }

      if (input.isDown(InputID::C)) {
        stand = (stand + (shift_pressed ? 2 : 1)) % 3;
      }

      if (input.isDown(InputID::Z)) {
        r_pitch = shift_pressed ? 0 : 2;
      }
      if (input.isDown(InputID::X)) {
        r_pitch = shift_pressed ? 6 : 4;
      }

      if (input.isDown(InputID::Q)) {
        r_yaw = shift_pressed ? 12 : 7;
      }
      if (input.isDown(InputID::E)) {
        r_yaw = shift_pressed ? 0 : 5;
      }

      int32_t move_amount;
      if (x == 0 && y == 0) {
        move_amount = 0;
      } else if (shift_pressed) {
        move_amount = consts::numMoveAmountBuckets - 1;
      } else {
        move_amount = 1;
      }

      int32_t move_angle;
      if (x == 0 && y == 1) {
        move_angle = 0;
      } else if (x == 1 && y == 1) {
        move_angle = 1;
      } else if (x == 1 && y == 0) {
        move_angle = 2;
      } else if (x == 1 && y == -1) {
        move_angle = 3;
      } else if (x == 0 && y == -1) {
        move_angle = 4;
      } else if (x == -1 && y == -1) {
        move_angle = 5;
      } else if (x == -1 && y == 0) {
        move_angle = 6;
      } else if (x == -1 && y == 1) {
        move_angle = 7;
      } else {
        move_angle = 0;
      }

      mgr.setPvPAction(viz->curWorld, viz->curControl - 1, PvPDiscreteAction {
        .moveAmount = move_amount,
        .moveAngle = move_angle,
        .fire = r > 0 ? 2 : f,
        .stand = stand,
      }, PvPAimAction {
        .yaw = 0,
        .pitch = 0,
      }, PvPDiscreteAimAction {
        .yaw = r_yaw,
        .pitch = r_pitch,
      });

      (void)r_yaw;
      (void)r_pitch;

      for (int world = 0; world < viz->numWorlds; world++)
      {
          for (int agent = 0; agent < 6; agent++)
          {
              if ((viz->doAI[0] && agent < 3) || (viz->doAI[1] && agent > 2))
                  doAI(viz, mgr, world, agent);
          }
      }

      if (input_events.downEvent(InputID::K)) {
        mgr.setHP(viz->curWorld, viz->curControl - 1, 0);
      }
    }

    if (viz->simEventsState.downEvent(InputID::K1)) {
      mgr.triggerReset(viz->curWorld);
    }

    if (viz->simEventsState.downEvent(InputID::K0)) {
      mgr.setUniformAgentPolicy(AgentPolicy { -1 });
    }

    if (viz->simEventsState.downEvent(InputID::K9)) {
      mgr.setUniformAgentPolicy(AgentPolicy { 0 });
    }

    trackHumanTrace(viz, mgr, viz->mouse_yaw_delta, viz->mouse_pitch_delta);
    viz->mouse_yaw_delta = 0;
    viz->mouse_pitch_delta = 0;

    mgr.step();
    viz->simEventsState.clear();

    populateAgentTrajectories(viz, mgr);

    //step_fn(step_data);

    viz->last_sim_tick_time = cur_frame_start_time;
  }

  vizStep(viz, mgr, frontend_delta_t);

  gpu->presentSwapchainImage(viz->swapchain);
  
  return running;
}

void loop(VizState *viz, Manager &mgr)
{
  viz->last_sim_tick_time = std::chrono::steady_clock::now();
  viz->last_frontend_tick_time = std::chrono::steady_clock::now();

  initMapCamera(viz, mgr);

  loadHeatmapData(viz);

  viz->mouse_yaw_delta = 0;
  viz->mouse_pitch_delta = 0;


#ifdef EMSCRIPTEN
  static VizState *global_viz = viz;
  static Manager *global_mgr = &mgr;

  emscripten_set_main_loop([]() {
    bool running = tick(global_viz, *global_mgr);
    if (!running) {
      emscripten_cancel_main_loop();
      VizSystem::shutdown(global_viz);
      delete global_mgr;
      exit(0);
    }
  }, 0, 0);
#else
  while (tick(viz, mgr)) {}
  VizSystem::shutdown(viz);
  delete &mgr;
#endif
}

void registerTypes(ECSRegistry &registry)
{
  registry.registerComponent<VizCamera>();
  registry.registerSingleton<VizWorld>();
}

// https://lemire.me/blog/2021/06/03/computing-the-number-of-digits-of-an-integer-even-faster/
static int32_t numDigits(uint32_t x)
{
  static uint64_t table[] = {
      4294967296,  8589934582,  8589934582,  8589934582,  12884901788,
      12884901788, 12884901788, 17179868184, 17179868184, 17179868184,
      21474826480, 21474826480, 21474826480, 21474826480, 25769703776,
      25769703776, 25769703776, 30063771072, 30063771072, 30063771072,
      34349738368, 34349738368, 34349738368, 34349738368, 38554705664,
      38554705664, 38554705664, 41949672960, 41949672960, 41949672960,
      42949672960, 42949672960};

  uint32_t idx = 31 - __builtin_clz(x | 1);
  return (x + table[idx]) >> 32;
}

static void flyCamUI(Camera &cam)
{
  auto side_size = ImGui::CalcTextSize(" Bottom " );
  side_size.y *= 1.4f;
  ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign,
                      ImVec2(0.5f, 0.f));

  if (ImGui::Button("Top", side_size)) {
    cam.position = 10.f * math::up;
    cam.fwd = -math::up;
    cam.up = -math::fwd;
    cam.right = cross(cam.fwd, cam.up);
  }

  ImGui::SameLine();

  if (ImGui::Button("Left", side_size)) {
    cam.position = -10.f * math::right;
    cam.fwd = math::right;
    cam.up = math::up;
    cam.right = cross(cam.fwd, cam.up);
  }

  ImGui::SameLine();

  if (ImGui::Button("Right", side_size)) {
    cam.position = 10.f * math::right;
    cam.fwd = -math::right;
    cam.up = math::up;
    cam.right = cross(cam.fwd, cam.up);
  }

  ImGui::SameLine();

  if (ImGui::Button("Bottom", side_size)) {
    cam.position = -10.f * math::up;
    cam.fwd = math::up;
    cam.up = math::fwd;
    cam.right = cross(cam.fwd, cam.up);
  }

  ImGui::PopStyleVar();

  auto ortho_size = ImGui::CalcTextSize(" Orthographic ");
  ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign,
                      ImVec2(0.5f, 0.f));
  if (ImGui::Selectable("Perspective", cam.perspective, 0,
                        ortho_size)) {
    cam.perspective = true;
  }
  ImGui::SameLine();

  if (ImGui::Selectable("Orthographic", !cam.perspective, 0,
                        ortho_size)) {
    cam.perspective = false;
  }

  ImGui::SameLine();

  ImGui::PopStyleVar();

  ImGui::TextUnformatted("Projection");

  float digit_width = ImGui::CalcTextSize("0").x;
  ImGui::SetNextItemWidth(digit_width * 6);
  if (cam.perspective) {
    ImGui::DragFloat("FOV", &cam.fov, 1.f, 1.f, 179.f, "%.0f");
  } else {
    ImGui::DragFloat("View Size", &cam.orthoHeight,
                     0.5f, 0.f, 100.f, "%0.1f");
  }
}

static void cfgUI(VizState *viz, Manager &mgr)
{ 
  ImGui::Begin("Controls");

  {
    float worldbox_width = ImGui::CalcTextSize(" ").x * (
      std::max(numDigits(viz->numWorlds) + 2, i32(7)));

    if (viz->numWorlds == 1) {
      ImGui::BeginDisabled();
    }

    ImGui::PushItemWidth(worldbox_width);
    ImGui::DragInt("Current World ID", &viz->curWorld, 1.f, 0,
        viz->numWorlds - 1, "%d", ImGuiSliderFlags_AlwaysClamp);
    ImGui::PopItemWidth();

    if (viz->numWorlds == 1) {
      ImGui::EndDisabled();
    }
  }

  ImGui::Checkbox("Control Current View", &viz->linkViewControl);

  {
    StackAlloc str_alloc;
    const char **cam_opts = str_alloc.allocN<const char *>(viz->numViews + 1);
    cam_opts[0] = "Free Camera";

    ImVec2 combo_size = ImGui::CalcTextSize(" Free Camera ");

    for (i32 i = 0; i < viz->numViews; i++) {
      const char *agent_prefix = "Agent ";

      i32 num_bytes = strlen(agent_prefix) + numDigits(i) + 1;
      cam_opts[i + 1] = str_alloc.allocN<char>(num_bytes);
      snprintf((char *)cam_opts[i + 1], num_bytes, "%s%u",
               agent_prefix, (uint32_t)i);
    }

    ImGui::PushItemWidth(combo_size.x * 1.25f);
    if (ImGui::BeginCombo("Current View", cam_opts[viz->curView])) {
      for (i32 i = 0; i < viz->numViews + 1; i++) {
        const bool is_selected = viz->curView == i;
        if (ImGui::Selectable(cam_opts[i], is_selected)) {
          viz->curView = (uint32_t)i;
        }

        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }

      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    i32 old_control = viz->curControl;
    if (viz->linkViewControl) {
      viz->curControl = viz->curView;
      ImGui::BeginDisabled();
    }

    ImGui::PushItemWidth(combo_size.x * 1.25f);
    if (ImGui::BeginCombo("Input Control", cam_opts[viz->curControl])) {
      for (CountT i = 0; i < viz->numViews + 1; i++) {
        const bool is_selected = viz->curControl == (i32)i;
        if (ImGui::Selectable(cam_opts[i], is_selected)) {
          viz->curControl = (i32)i;
        }

        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }

      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    if (viz->linkViewControl) {
      ImGui::EndDisabled();
    }

    if (old_control != viz->curControl) {
      if (old_control != 0) {
        mgr.setAgentPolicy(viz->curWorld, old_control - 1, {0});
        mgr.setPvPAction(viz->curWorld, old_control - 1, PvPDiscreteAction {
          .moveAmount = 0,
          .moveAngle = 0,
          .fire = 0,
          .stand = 0,
        }, PvPAimAction {
          .yaw = 0,
          .pitch = 0,
        }, PvPDiscreteAimAction {
          .yaw = consts::discreteAimNumYawBuckets / 2,
          .pitch = consts::discreteAimNumPitchBuckets / 2,
        });
      }
    }
  }

  ImGui::Text("Current Free Camera");
  ImGui::Text("    Position: (%d %d %d)",
      (int)viz->flyCam.position.x,
      (int)viz->flyCam.position.y,
      (int)viz->flyCam.position.z);
  ImGui::Text("    Forward: (%.2f %.2f %.2f)",
      viz->flyCam.fwd.x,
      viz->flyCam.fwd.y,
      viz->flyCam.fwd.z);
  ImGui::Text("    Up:      (%.2f %.2f %.2f)",
      viz->flyCam.up.x,
      viz->flyCam.up.y,
      viz->flyCam.up.z);

  ImGui::Spacing();
  ImGui::TextUnformatted("Simulation Settings");
  ImGui::Separator();

  ImGui::PushItemWidth(ImGui::CalcTextSize(" ").x * 7);
  ImGui::DragInt("Tick Rate (Hz)", (int *)&viz->simTickRate, 5.f, 0, 1000);
  if (viz->simTickRate < 0) {
    viz->simTickRate = 0;
  }
  ImGui::PopItemWidth();

  ImGui::Spacing();

  if (viz->curView != 0) {
    ImGui::BeginDisabled();
  }

  ImGui::TextUnformatted("Free Camera Config");
  ImGui::Separator();

  flyCamUI(viz->flyCam);

  if (viz->curView != 0) {
    ImGui::EndDisabled();
  }

  ImGui::Spacing();
  ImGui::TextUnformatted("Player Info Settings");
  ImGui::Separator();

  //ImGui::Checkbox("Show Unmasked Minimaps", &viz->showUnmaskedMinimaps);

  ImGui::Checkbox("Show Debug menus", &viz->debugMenus);

  ImGui::End();
}

static inline void agentInfoUI(Engine &ctx, VizState *viz)
{
  MatchResult &match_result = ctx.singleton<MatchResult>();

  auto drawDepthArray = [](const LidarData *arr, CountT num_elems, float rescale) {
    float box_width = 10.f;
    float box_height = 10.f;

    for (CountT i = 0; i < num_elems; i++) {
      float d = arr[i].depth;

      ImU32 color;
      if (d == -1.f) {
        color = ImGui::GetColorU32(ImVec4(0.3f, 0.0f, 0.0f, 1.f));
      } else {
        float v = fmaxf((d - consts::agentRadius) * rescale, 0.f);
        v = fmaxf(fminf(logf(v + 1.f), 1.f), 0.f);
        color = ImGui::GetColorU32(ImVec4(v, v, v, 1.f));
      }

      ImVec2 p0 = ImGui::GetCursorScreenPos();
      ImVec2 p1 = ImVec2(p0.x + box_width, p0.y + box_height);

      ImGui::GetWindowDrawList()->AddRectFilled(p0, p1, color);
      ImGui::SetCursorScreenPos(ImVec2(p0.x + box_width, p0.y));
    }
  };

  for (int64_t i = 0; i < viz->numViews; i++) {
    auto player_str = std::string("Player ") + std::to_string(i);
    ImGui::Begin(player_str.c_str());
    //float old_size = ImGui::GetFont()->Scale;
    //ImGui::GetFont()->Scale *= 1.5f;
    //ImGui::PushFont(ImGui::GetFont());
    //

    Entity agent = ctx.data().agents[i];

    const Position &cur_pos = ctx.get<Position>(agent);
    const Aim &cur_aim = ctx.get<Aim>(agent);
    const TeamInfo &team = ctx.get<TeamInfo>(agent);

    const HP &hp = ctx.get<HP>(agent);
    const Magazine &mag = ctx.get<Magazine>(agent);
    const Reward &reward = ctx.get<Reward>(agent);

    const FwdLidar &fwd_lidar = ctx.get<FwdLidar>(agent);
    const RearLidar &rear_lidar = ctx.get<RearLidar>(agent);
    const AgentMap &agent_map = ctx.get<AgentMap>(agent);
    const UnmaskedAgentMap &unmasked_agent_map =
        ctx.get<UnmaskedAgentMap>(agent);

    ImGui::Text("Position        (%.1f %.1f %.1f)",
                cur_pos.x, cur_pos.y, cur_pos.z);
    ImGui::Text("Aim             (%.2f, %.2f)",
                cur_aim.yaw, cur_aim.pitch);
    ImGui::Text("HP              %.1f", hp.hp);
    ImGui::Text("Magazine Count  %d", mag.numBullets);
    ImGui::Text("Is Reloading?   %d", mag.isReloading > 0);
    ImGui::Text("Reward          %f", reward.v);
    ImGui::Text("Team            %s",
                team.team == 0 ? "A" : (
                  team.team == 1 ? "B" : "?"));
    ImGui::Spacing();

    ImGui::Text("Fwd Lidar");
    ImVec2 lidar_start = ImGui::GetCursorScreenPos();
    lidar_start.x += 10.f;
    for (CountT y = consts::fwdLidarHeight - 1; y >= 0; y--) {
      ImGui::SetCursorScreenPos(lidar_start);
      drawDepthArray(fwd_lidar.data[y], consts::fwdLidarWidth,
                     1.f / 150.f);
      lidar_start.y += 10.f;
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Text("Rear Lidar");
    lidar_start = ImGui::GetCursorScreenPos();
    lidar_start.x += 10.f;
    for (CountT y = consts::rearLidarHeight - 1; y >= 0; y--) {
      ImGui::SetCursorScreenPos(lidar_start);
      drawDepthArray(rear_lidar.data[y], consts::rearLidarWidth,
                     1.f / 150.f);
      lidar_start.y += 10.f;
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Text("Minimap");
    {
      const MapItem (*minimap)[AgentMap::res];
      if (viz->showUnmaskedMinimaps) {
        minimap = unmasked_agent_map.data;
      } else {
        minimap = agent_map.data;
      }

      ImVec2 row_start = ImGui::GetCursorScreenPos();
      row_start.x += 10.f;
      for (i32 y = 0; y < AgentMap::res; y++) {
        const float box_width = 10.f;
        const float box_height = 10.f;

        ImGui::SetCursorScreenPos(row_start);
        for (i32 x = 0; x < AgentMap::res; x++) {
          MapItem map_cell = minimap[y][x];

          ImVec4 color_vec = { 0, 0, 0, 1 };
          if (map_cell.numOpponentsPresent > 0) {
            color_vec.x = 1.f;
          }

          if (map_cell.numPastOpponentsPresent > 0) {
            color_vec.y = 1.f;
          }

          if (map_cell.numTeammatesPresent > 0) {
            color_vec.z = 1.f;
          }

          if (map_cell.iAmPresent) {
            color_vec.z = 1.f;
          }

          ImU32 color = ImGui::GetColorU32(color_vec);

          ImVec2 p0 = ImGui::GetCursorScreenPos();
          ImVec2 p1 = ImVec2(p0.x + box_width, p0.y + box_height);

          ImGui::GetWindowDrawList()->AddRectFilled(p0, p1, color);
          ImGui::SetCursorScreenPos(ImVec2(p0.x + box_width, p0.y));
        }

        row_start.y += box_height;
      }
    }

    //ImGui::GetFont()->Scale = old_size;
    //ImGui::PopFont();
    ImGui::End();
  }

  ImGui::Begin("Match Results");
  ImGui::Text("Team A Kills %u", (uint32_t)match_result.teamTotalKills[0]);
  ImGui::Text("Team B Kills %u", (uint32_t)match_result.teamTotalKills[1]);
  ImGui::Text("Team A Obj   %u", (uint32_t)match_result.teamObjectivePoints[0]);
  ImGui::Text("Team B Obj   %u", (uint32_t)match_result.teamObjectivePoints[1]);
  ImGui::End();

  if (viz->doAI[0] || viz->doAI[1]) {
    if (actions.empty())
        actions.resize(viz->numWorlds * viz->numViews);
    for (int agent = 0; agent < viz->numViews; agent++)
    {
        planAI(ctx, viz, ctx.worldID().idx, agent);
    }
  }
}

static void playerInfoUI(Engine &ctx, i32 agent_idx)
{
  Entity agent = ctx.data().agents[agent_idx];

  auto viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(ImVec2(viewport->WorkSize.x, 0.f),
                          0, ImVec2(1.f, 0.f));
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
  ImGui::Begin("Player Info", nullptr,
               ImGuiWindowFlags_NoMove |
               ImGuiWindowFlags_NoInputs |
               ImGuiWindowFlags_NoTitleBar |
               ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PopStyleVar();
  ImGui::SetWindowFontScale(3.0f);

  HP hp = ctx.get<HP>(agent);
  Magazine magazine = ctx.get<Magazine>(agent);
  CombatState combat_state = ctx.get<CombatState>(agent);
  const WeaponStats &weapon_stats =
      ctx.data().weaponTypeStats[combat_state.weaponType];
  
  MatchResult match_result = ctx.singleton<MatchResult>();

  TeamInfo team = ctx.get<TeamInfo>(agent);

  i32 my_team_points = match_result.teamObjectivePoints[team.team];
  i32 enemy_team_points = match_result.teamObjectivePoints[team.team ^ 1];

  ImGui::Text("HP %d", (int)hp.hp);
  if (magazine.isReloading) {
    ImGui::Text("Reloading........");
  } else {
    ImGui::Text("Magazine: %d / %d", (int)magazine.numBullets,
                weapon_stats.magSize);
  }

  ImGui::Text("  Our Points: %d", my_team_points);
  ImGui::Text("Enemy Points: %d", enemy_team_points);

  ImGui::End();

  ImVec2 display_size = ImGui::GetIO().DisplaySize;
  ImVec2 center = ImVec2(display_size.x * 0.5f, display_size.y * 0.5f);
  
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  
  float radius = 5.0f;
  ImU32 color = IM_COL32(255, 255, 255, 255);
  int segments = 32;
  float thickness = 2.0f;
  
  draw_list->AddCircle(center, radius, color, segments, thickness);

}

static void trajectoryDBUI(Engine &ctx, VizState *viz)
{
  ImGui::Begin("Trajectory DB");

  ImGui::Text("Num Trajectories: %ld", numTrajectories(viz->trajectoryDB));

  const float button_width = 100.f;

  int num_trajectories = numTrajectories(viz->trajectoryDB);

  ImGui::PushItemWidth(button_width);

  ImGui::Text("Trajectory Working Set Size: %ld", viz->curWorkingTrajectories.size());

  if (ImGui::Button("Clear Trajectory Working Set")) {
    viz->curWorkingTrajectories.clear();
  }

  if (ImGui::Button("Select All Trajectories")) {
    for (i64 i = 0; i < num_trajectories; i++) {
      viz->curWorkingTrajectories.push_back(i);
    }
  }

  bool working_trajectories_empty = viz->curWorkingTrajectories.size() == 0;

  if (working_trajectories_empty) {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button("Create Training Set")) {
    buildTrajectoryTrainingSet(viz->trajectoryDB, viz->curWorkingTrajectories, "training_set.bin");
    viz->curWorkingTrajectories.clear();
  }

  if (working_trajectories_empty) {
    ImGui::EndDisabled();
  }

  ImGui::NewLine();

  if (num_trajectories == 0) {
    ImGui::BeginDisabled();
  }

  int new_index = viz->curVizTrajectoryIndex;
  ImGui::DragInt("Select Trajectory", &new_index, 0.25f, -1, num_trajectories - 1,
                 viz->curVizTrajectoryIndex == -1 ? "None" : "%d", ImGuiSliderFlags_AlwaysClamp);

  if (new_index == -1) {
    viz->curVizTrajectoryID = -1;
  } else if (new_index != viz->curVizTrajectoryIndex) {
    viz->curVizTrajectoryID = advanceNTrajectories(viz->trajectoryDB,
      viz->curVizTrajectoryID, new_index - viz->curVizTrajectoryIndex);
  }
  viz->curVizTrajectoryIndex = new_index;

  if (num_trajectories == 0) {
    ImGui::EndDisabled();
  }

  const char *trajectory_type_str = nullptr;
  const char *trajectory_tag_str = nullptr;
  if (viz->curVizTrajectoryID == -1) {
    ImGui::Text("Trajectory ID: [None]");
    trajectory_type_str = "None";
    trajectory_tag_str = "";
  } else {
    ImGui::Text("Trajectory ID: %ld", viz->curVizTrajectoryID);
    TrajectoryType trajectory_type = getTrajectoryType(viz->trajectoryDB, viz->curVizTrajectoryID);
    switch (trajectory_type) {
      case TrajectoryType::Human: {
        trajectory_type_str = "Human";
      } break;
      case TrajectoryType::RL: {
        trajectory_type_str = "RL";
      } break;
      case TrajectoryType::Hardcoded: {
        trajectory_type_str = "Hardcoded";
      } break;
      default: {
        trajectory_type_str = "Unknown";
      } break;
    }

    trajectory_tag_str = getTrajectoryTag(viz->trajectoryDB, viz->curVizTrajectoryID);
    if (trajectory_tag_str == nullptr) {
      trajectory_tag_str = "";
    }
  }

  ImGui::Text("Trajectory Type: %s", trajectory_type_str);
  ImGui::Text("Trajectory Tag: %s", trajectory_tag_str);

  bool is_valid_trajectory = viz->curVizTrajectoryID != -1;
  if (!is_valid_trajectory) {
    ImGui::BeginDisabled();
  }

  if (ImGui::Button("Delete Trajectory")) {
    removeTrajectory(viz->trajectoryDB, viz->curVizTrajectoryID);
    viz->curVizTrajectoryID = -1;
    viz->curVizTrajectoryIndex = -1;
  }

  if (ImGui::Button("Add Trajectory to Working Set")) {
    viz->curWorkingTrajectories.push_back(viz->curVizTrajectoryID);
  }

  if (!is_valid_trajectory) {
    ImGui::EndDisabled();
  }

  ImGui::PopItemWidth();

  ImGui::End();
}

static Engine & uiLogic(VizState *viz, Manager &mgr)
{
  const UserInputEvents &input_events = viz->ui->inputEvents();
  if (input_events.downEvent(InputID::Esc)) {
    if (viz->curControl != 0) {
        mgr.setAgentPolicy(viz->curWorld, viz->curControl - 1, {0});
        mgr.setPvPAction(viz->curWorld, viz->curControl - 1, PvPDiscreteAction {
          .moveAmount = 0,
          .moveAngle = 0,
          .fire = 0,
          .stand = 0,
        }, PvPAimAction {
          .yaw = 0,
          .pitch = 0,
        }, PvPDiscreteAimAction {
          .yaw = consts::discreteAimNumYawBuckets / 2,
          .pitch = consts::discreteAimNumPitchBuckets / 2,
        });
    }

    viz->mainMenu = true;
    viz->simTickRate = 0;
    viz->curControl = 0;
    viz->curView = 0;
  }

  ImGuiSystem::newFrame(viz->ui, viz->window->systemUIScale, 1.f / 60.f);

  if (viz->curView == 0) {
    cfgUI(viz, mgr);
  }

  if (viz->curControl != 0) {
    mgr.setAgentPolicy(viz->curWorld, viz->curControl - 1, {consts::humanPolicyID});
    mgr.setPvPAction(viz->curWorld, viz->curControl - 1, PvPDiscreteAction {
      .moveAmount = 0,
      .moveAngle = 0,
      .fire = 0,
      .stand = 0,
    }, PvPAimAction {
      .yaw = 0,
      .pitch = 0,
    }, PvPDiscreteAimAction {
      .yaw = consts::discreteAimNumYawBuckets / 2,
      .pitch = consts::discreteAimNumPitchBuckets / 2,
    });
  }

  Engine &ctx = mgr.getWorldContext(viz->curWorld);

  if (viz->curView == 0) {
    if (viz->debugMenus) {
      agentInfoUI(ctx, viz);
    }

    if (viz->trajectoryDB) {
      trajectoryDBUI(ctx, viz);
    }

#ifdef DB_SUPPORT
    analyticsDBUI(ctx, viz);
#endif
  } else {
    playerInfoUI(ctx, viz->curView - 1);
  }

  return ctx;
}

static void setupViewData(Engine &ctx,
                          const Camera &cam,
                          VizState *viz,
                          GlobalPassData *out)
{
  (void)ctx;

  float aspect_ratio = (f32)viz->window->pixelWidth / viz->window->pixelHeight;

  float fov = cam.fine_aim ? cam.fov * 0.5f : cam.fov;
  float fov_scale = 1.f / tanf(math::toRadians(fov * 0.5f));

  float screen_x_scale = fov_scale / aspect_ratio;
  float screen_y_scale = fov_scale;

  out->view.camTxfm.rows[0] = Vector4::fromVec3W(cam.right, cam.position.x);
  out->view.camTxfm.rows[1] = Vector4::fromVec3W(cam.up, cam.position.y);
  out->view.camTxfm.rows[2] = Vector4::fromVec3W(cam.fwd, cam.position.z);

  out->view.fbDims =
    { (u32)viz->window->pixelWidth, (u32)viz->window->pixelHeight };
  out->view.screenScale = Vector2(screen_x_scale, screen_y_scale);
  out->view.zNear = 1.f;
}

static void setupLightData(VizState *viz, GlobalPassData *out)
{
  shader::LightData lights;
  lights.sunPosition = {
    0.65f * (viz->flyCam.mapMin.x + viz->flyCam.mapMax.x),
    0.45f * (viz->flyCam.mapMin.y + viz->flyCam.mapMax.y),
    viz->flyCam.mapMax.z,
  };

  lights.sunColor = { 0.3f, 0.7f, 0.7f };
  lights.sunColor *= 1.2f;

  out->lights = lights;
}

static NonUniformScaleObjectTransform computeNonUniformScaleTxfm(
    Vector3 t, Quat r, Diag3x3 s)
{
  float x2 = r.x * r.x;
  float y2 = r.y * r.y;
  float z2 = r.z * r.z;
  float xz = r.x * r.z;
  float xy = r.x * r.y;
  float yz = r.y * r.z;
  float wx = r.w * r.x;
  float wy = r.w * r.y;
  float wz = r.w * r.z;

  float y2z2 = y2 + z2;
  float x2z2 = x2 + z2;
  float x2y2 = x2 + y2;

  Diag3x3 ds = 2.f * s;
  Diag3x3 i_s = 1.f / s;
  Diag3x3 i_ds = 2.f * i_s;

  NonUniformScaleObjectTransform out;
  out.o2w = {{
    { s.d0 - ds.d0 * y2z2, ds.d1 * (xy - wz), ds.d2 * (xz + wy), t.x },
    { ds.d0 * (xy + wz), s.d1 - ds.d1 * x2z2, ds.d2 * (yz - wx), t.y },
    { ds.d0 * (xz - wy), ds.d1 * (yz + wx), s.d2 - ds.d2 * x2y2, t.z },
  }};

  Vector3 w2o_r0 = 
      { i_s.d0 - i_ds.d0 * y2z2, i_ds.d1 * (xy + wz), ds.d2 * (xz - wy) };
  Vector3 w2o_r1 =
      { i_ds.d0 * (xy - wz), i_s.d1 - i_ds.d1 * x2z2, i_ds.d2 * (yz + wx) };
  Vector3 w2o_r2 =
      { i_ds.d0 * (xz + wy), i_ds.d1 * (yz - wx), i_s.d2 - i_ds.d2 * x2y2 };

  out.w2o = {{
    Vector4::fromVec3W(w2o_r0, -dot(w2o_r0, t)),
    Vector4::fromVec3W(w2o_r1, -dot(w2o_r1, t)),
    Vector4::fromVec3W(w2o_r2, -dot(w2o_r2, t)),
  }};

  return out;
}

static void renderMap(VizState *viz,
                      RasterPassEncoder &raster_enc)
{
  raster_enc.setShader(viz->mapShader);
  raster_enc.setParamBlock(0, viz->globalParamBlock);
  raster_enc.setParamBlock(1, viz->mapGeoParamBlock);

  raster_enc.setIndexBufferU32(viz->mapBuffer);

  float wireframe_width = viz->curView == 0 ? 0.5f : 1.0f;

  for (uint32_t mesh_idx = 0; mesh_idx < (uint32_t)viz->mapMeshes.size();
       mesh_idx++) {
    const MapGeoMesh &mesh = viz->mapMeshes[mesh_idx];

    raster_enc.drawData(MapPerDraw {
      .wireframeConfig = { 1.0f, 0.8f, 0.2f, wireframe_width },
      .meshVertexOffset = mesh.vertOffset,
      .meshIndexOffset = mesh.indexOffset,
      .metallic = 0.1f,
      .roughness = 0.7f,
    });

    raster_enc.drawIndexed(0, mesh.indexOffset, mesh.numTris);
  }
}

static void renderObjects(Engine &ctx, VizState *viz,
                          RasterPassEncoder &raster_enc)
{
  raster_enc.setShader(viz->renderableObjectsShader);
  raster_enc.setParamBlock(0, viz->globalParamBlock);

  ctx.iterateQuery(ctx.singleton<VizWorld>().renderableObjectsQuery,
    [&]
  (Position pos, Rotation rot, Scale scale, ObjectID obj_id)
  {
    if ((size_t)obj_id.idx >= viz->objects.size()) {
      return;
    }

    Object obj = viz->objects[obj_id.idx];

    for (i32 i = 0; i < obj.numMeshes; i++) {
      Mesh mesh = viz->meshes[i + obj.meshOffset];

      raster_enc.drawData(OpaqueGeoPerDraw {
        .txfm = computeNonUniformScaleTxfm(pos, rot, scale),
        .baseColor = Vector4::fromVec3W(
            viz->meshMaterials[mesh.materialIndex % viz->meshMaterials.size()].color, 1.f),
      });

      raster_enc.setVertexBuffer(0, mesh.buffer);
      raster_enc.setIndexBufferU32(mesh.buffer);

      raster_enc.drawIndexed(
          mesh.vertexOffset, mesh.indexOffset, mesh.numTriangles);
    }
  });
}

static void renderGoalRegions(Engine &ctx, VizState *viz,
                              RasterPassEncoder &raster_enc)
{
  const GoalRegionsState &regions_state = ctx.singleton<GoalRegionsState>();

  Vector3 attacker_region_color = { 0, 0, 1 };
  Vector3 defender_region_color = { 1, 0, 0 };

  auto renderGoalRegions = 
    [&]
  (u32 tris_per_region)
  {
    for (int i = 0; i < (int)ctx.data().numGoalRegions; i++) {
      bool region_active = regions_state.regionsActive[i];
      if (!region_active) {
        continue;
      }

      const GoalRegion &region = ctx.data().goalRegions[i];

      for (int j = 0; j < region.numSubRegions; j++) {
        ZOBB zobb = region.subRegions[j];

        Vector3 diag = zobb.pMax - zobb.pMin;
        Vector3 center = 0.5f * (zobb.pMax + zobb.pMin);

        raster_enc.drawData(GoalRegionPerDraw {
          .txfm = computeNonUniformScaleTxfm(
              center, Quat::angleAxis(zobb.rotation, math::up),
              Diag3x3 { diag.x, diag.y, diag.z }),
          .color = Vector4::fromVec3W(
              region.attackerTeam ? attacker_region_color : defender_region_color,
              0.3f),
          .lineWidth = 1.f,
        });

        raster_enc.draw(0, tris_per_region);
      }
    }
  };

  raster_enc.setParamBlock(0, viz->globalParamBlock);

  raster_enc.setShader(viz->goalRegionsShaderWireframe);
  renderGoalRegions(24);

  raster_enc.setShader(viz->goalRegionsShader);
  renderGoalRegions(12);

  raster_enc.setShader(viz->goalRegionsShaderWireframeNoDepth);
  renderGoalRegions(24);
}

static void renderShotViz(Engine &ctx, VizState *viz,
                          RasterPassEncoder &raster_enc, float delta_t)
{
  MappedTmpBuffer line_data_buf;
  int num_lines = 0;

  /*{
      const Navmesh& navmesh = ctx.singleton<LevelData>().navmesh;
      num_lines = navmesh.numTris * 3;

      line_data_buf =
          raster_enc.tmpBuffer(sizeof(ShotVizLineData) * num_lines, 256);

      ShotVizLineData* out_lines = (ShotVizLineData*)line_data_buf.ptr;

      for (int tri = 0; tri < navmesh.numTris; tri++)
      {
          for (int edge = 0; edge < 3; edge++)
          {
              Vector3 a = navmesh.vertices[navmesh.triIndices[tri * 3 + edge]];
              Vector3 b = navmesh.vertices[navmesh.triIndices[tri * 3 + ((edge + 1) % 3)]];

              float alpha = 0.5;

              Vector3 color = { 0, 1, 0 };

              *out_lines++ = {
                .start = a,
                .end = b,
                .color = Vector4::fromVec3W(color, alpha),
              };
          }
      }
  }*/

  {
    const auto& query = ctx.query<ShotVizState, ShotVizRemaining>();

    const int particleLifetime = 5;
    ctx.iterateQuery(query, [&](ShotVizState&, ShotVizRemaining &remaining)
    {
      num_lines += 1;
      if (remaining.numStepsTotal - remaining.numStepsRemaining < particleLifetime)
      {
        num_lines += ShotVizRemaining::numParticles;
        if (remaining.hitEffect)
          num_lines += ShotVizRemaining::numParticles;
      }
    });

    if (num_lines == 0) {
        return;
    }

    line_data_buf =
        raster_enc.tmpBuffer(sizeof(ShotVizLineData) * num_lines, 256);

    ShotVizLineData* out_lines = (ShotVizLineData*)line_data_buf.ptr;
    int lineCheck = 0;

    ctx.iterateQuery(query,
    [&](ShotVizState& state, ShotVizRemaining& remaining)
    {
      Vector3 a = state.from;
      Vector3 b = state.from + state.dir * state.hitT;

      float alpha = remaining.numStepsRemaining / 30.f;
      alpha *= alpha;

      Vector3 color;
      if (state.team == 0) {
        color = { 0, 0, 1 };
      }
      else {
        color = { 1, 0, 0 };
      }

      if (!state.hit) {
        color *= 0.5f;
        alpha *= 0.25f;
      }

      *out_lines++ = {
        .start = a,
        .pad = {},
        .end = b,
        .pad2 = {},
        .color = Vector4::fromVec3W(color, alpha),
      };
      lineCheck++;

      // Simulate the particle effects.
      const float particleSpeed = 200.0f;
      const float particleSize = 0.03f;
      if (!remaining.initialized)
      {
        for (int i = 0; i < ShotVizRemaining::numParticles; i++) {
          remaining.hitParticles[i].vel.x = ((std::rand() % 1024) / 512.0f - 1.0f) * particleSpeed;
          remaining.hitParticles[i].vel.y = ((std::rand() % 1024) / 512.0f - 1.0f) * particleSpeed;
          remaining.hitParticles[i].vel.z = ((std::rand() % 1024) / 512.0f - 1.0f) * particleSpeed;
          remaining.muzzleParticles[i].vel.x += ((std::rand() % 1024) / 2048.0f - 0.25f);
          remaining.muzzleParticles[i].vel.y += ((std::rand() % 1024) / 2048.0f - 0.25f);
          remaining.muzzleParticles[i].vel.z += ((std::rand() % 1024) / 2048.0f - 0.25f);
          remaining.muzzleParticles[i].vel *= particleSpeed * 2.0f;
          remaining.initialized = true;
        }
      }
      if (remaining.numStepsTotal - remaining.numStepsRemaining < particleLifetime) {
        for (int i = 0; i < ShotVizRemaining::numParticles; i++) {
          Vector3 muzzleColor = { 1.0f, 0.5f, 0.0f };
          Vector3 hitColor = { 0.75f, 0.75f, 1.0f };
          *out_lines++ = {
            .start = remaining.muzzleParticles[i].pos,
            .pad = {},
            .end = remaining.muzzleParticles[i].pos + remaining.muzzleParticles[i].vel * particleSize,
            .pad2 = {},
            .color = Vector4::fromVec3W(muzzleColor, 0.5f),
          };
          lineCheck++;
          if (remaining.hitEffect) {
            *out_lines++ = {
              .start = remaining.hitParticles[i].pos,
              .pad = {},
              .end = remaining.hitParticles[i].pos + remaining.hitParticles[i].vel * particleSize,
              .pad2 = {},
              .color = Vector4::fromVec3W(hitColor, 1.0f),
            };
            lineCheck++;
          }
          remaining.muzzleParticles[i].pos += remaining.muzzleParticles[i].vel * delta_t;
          remaining.hitParticles[i].pos += remaining.hitParticles[i].vel * delta_t;
        }
      }
    });
    assert(lineCheck ==  num_lines);
  }

  ParamBlock tmp_geo_block = raster_enc.createTemporaryParamBlock({
    .typeID = viz->shotVizParamBlockType,
    .buffers = {{
      .buffer = line_data_buf.buffer,
      .offset = line_data_buf.offset,
    }},
  });

  raster_enc.setShader(viz->shotVizShader);
  raster_enc.setParamBlock(0, viz->globalParamBlock);
  raster_enc.setParamBlock(1, tmp_geo_block);

  raster_enc.draw(0, num_lines * 2);
}

static void renderZones(Engine &ctx, VizState *viz,
                        RasterPassEncoder &raster_enc)
{
  auto renderZones =
    [&]
  (u32 num_tris)
  {
    ZoneState &zone_state = ctx.singleton<ZoneState>();

    for (CountT i = 0; i < ctx.data().zones.numZones; i++) {
      AABB aabb = ctx.data().zones.bboxes[i];
      float rotation = ctx.data().zones.rotations[i];

      Vector3 diag = aabb.pMax - aabb.pMin;
      Vector3 center = 0.5f * (aabb.pMax + aabb.pMin);

      Vector4 color;

      if (i != zone_state.curZone) {
        color = rgb8ToFloat(10, 175, 10, 1.f);
        continue;
      } else if (zone_state.curControllingTeam == -1 ||
                 !zone_state.isCaptured) {
        color = rgb8ToFloat(230, 230, 100, 1.f);
      } else if (zone_state.curControllingTeam == 0) {
        color = rgb8ToFloat(100, 100, 230, 1.f);
      } else if (zone_state.curControllingTeam == 1) {
        color = rgb8ToFloat(230, 100, 100, 1.f);
      } else {
        color = rgb8ToFloat(255, 255, 255, 1);
      }

      raster_enc.drawData(GoalRegionPerDraw {
        .txfm = computeNonUniformScaleTxfm(
            center, Quat::angleAxis(rotation, math::up),
            Diag3x3 { diag.x, diag.y, diag.z }),
        .color = color,
        .lineWidth = 10.f,
      });

      raster_enc.draw(0, num_tris);
    }
  };

  raster_enc.setShader(viz->goalRegionsShaderWireframe);
  renderZones(24);

  if (viz->curView == 0) {
    raster_enc.setShader(viz->goalRegionsShaderWireframeNoDepth);
    renderZones(24);
  }
}

static void renderSubZones(Engine &ctx, VizState *viz,
                           RasterPassEncoder &raster_enc)
{
  if (viz->curView != 0) {
    return;
  }

  raster_enc.setShader(viz->goalRegionsShaderWireframeNoDepth);

  for (Entity sub_zone_entity : ctx.data().subZones) {
    SubZone &sub_zone = ctx.get<SubZone>(sub_zone_entity);

    ZOBB zobb = sub_zone.zobb;
  
    Vector3 diag = zobb.pMax - zobb.pMin;
    Vector3 center = 0.5f * (zobb.pMax + zobb.pMin);
  
    Vector4 color = rgb8ToFloat(230, 100, 230, 1.f);
  
    raster_enc.drawData(GoalRegionPerDraw {
      .txfm = computeNonUniformScaleTxfm(
          center, Quat::angleAxis(zobb.rotation, math::up),
          Diag3x3 { diag.x, diag.y, diag.z }),
      .color = color,
      .lineWidth = 10.f,
    });
  
    raster_enc.draw(0, 24);
  }
}

static void renderAgents(Engine &ctx, VizState *viz,
                         RasterPassEncoder &raster_enc)
{
  const auto &query = ctx.query<Position, Rotation, Scale, CombatState,
      Magazine, HP, TeamInfo>();

  raster_enc.setShader(viz->agentShader);
  raster_enc.setParamBlock(0, viz->globalParamBlock);

  ctx.iterateQuery(query,
    [&]
  (Vector3 pos, Quat rot, Diag3x3 scale,
   CombatState &combat_state, Magazine &mag, HP &hp, TeamInfo &team_info)
  {
    (void)combat_state;

    Object obj = viz->objects[0];

    Vector3 agent_color;

    if (team_info.team == 0) {
      agent_color = { 0, 0, 1 };
    } else {
      agent_color = { 1, 0, 0 };
    }

    if (mag.isReloading) {
      agent_color.y = 1.f;
    }

    float agent_alpha = (float)hp.hp / 100;
    agent_alpha = fmaxf(agent_alpha, 0.1f);

    for (i32 i = 0; i < obj.numMeshes; i++) {
      Mesh mesh = viz->meshes[i + obj.meshOffset];

      raster_enc.drawData(AgentPerDraw {
        .txfm = computeNonUniformScaleTxfm(pos, rot, scale),
        .color = Vector4::fromVec3W(i == 0 ? agent_color : Vector3::zero(),
                                    agent_alpha),
      });

      raster_enc.setVertexBuffer(0, mesh.buffer);
      raster_enc.setIndexBufferU32(mesh.buffer);

      raster_enc.drawIndexed(
          mesh.vertexOffset, mesh.indexOffset, mesh.numTriangles);
    }
  });
}

static void renderAgentPaths(VizState *viz, RasterPassEncoder &raster_enc)
{
  raster_enc.setShader(viz->agentPathsShader);
  raster_enc.setParamBlock(0, viz->globalParamBlock);


  for (int team_idx = 0; team_idx < 2; team_idx++) {
    uint32_t total_num_line_verts = 0;
    for (int i = 0; i < viz->teamSize; i++) {
      AgentRecentTrajectory &traj = viz->agentTrajectories[team_idx * viz->teamSize + i];

      if (traj.curOffset == 0) {
        continue;
      }

      total_num_line_verts += 2 * (std::min(traj.curOffset, (i32)traj.points.size()) - 1);
    }

    if (total_num_line_verts == 0) {
      continue;
    }

    uint32_t num_buffer_bytes = total_num_line_verts * sizeof(Vector4);

    MappedTmpBuffer paths_buffer = raster_enc.tmpBuffer(num_buffer_bytes);

    Vector4 *path_vert_staging = (Vector4 *)paths_buffer.ptr;
    for (int i = 0; i < viz->teamSize; i++) {
      AgentRecentTrajectory &traj = viz->agentTrajectories[team_idx * viz->teamSize + i];

      i32 start_offset = std::max(i32(traj.curOffset - traj.points.size()), 0);
      i32 traj_len = traj.curOffset - start_offset;

      i32 cur_offset = start_offset;
      while (cur_offset != traj.curOffset - 1) {
        Vector3 a = traj.points[cur_offset % traj.points.size()];
        Vector3 b = traj.points[(cur_offset + 1) % traj.points.size()];

        float a_alpha = float(cur_offset - start_offset) / float(traj_len);
        float b_alpha = float(cur_offset + 1 - start_offset) / float(traj_len);

        *path_vert_staging++ = Vector4::fromVec3W(a, a_alpha);
        *path_vert_staging++ = Vector4::fromVec3W(b, b_alpha);

        cur_offset++;
      }
    }

    assert((char *)path_vert_staging == (char *)paths_buffer.ptr + num_buffer_bytes);

    ParamBlock paths_tmp_pb = raster_enc.createTemporaryParamBlock({
      .typeID = viz->agentPathsParamBlockType,
      .buffers = {
        { .buffer = paths_buffer.buffer, .offset = paths_buffer.offset, .numBytes = num_buffer_bytes },
      },
    });

    raster_enc.setParamBlock(1, paths_tmp_pb);
    raster_enc.drawData(team_idx == 0 ? Vector4(0, 0, 1, 1) : Vector4(1, 0, 0, 1));
    raster_enc.draw(0, total_num_line_verts);
  }
}

static void trajectoryDBRender(VizState *viz, RasterPassEncoder &raster_enc)
{
  if (viz->curVizTrajectoryID != -1) {
    Span<const AgentTrajectoryStep> steps = getTrajectorySteps(viz->trajectoryDB, viz->curVizTrajectoryID);

    raster_enc.drawData(Vector4(0, 1, 0, 1));
    i64 num_steps = steps.size();

    i64 total_num_verts = (num_steps - 1) * 2;
    i64 num_buffer_bytes = total_num_verts * sizeof(Vector4);

    MappedTmpBuffer line_data_buf = raster_enc.tmpBuffer(num_buffer_bytes);

    Vector4 *line_data_staging = (Vector4 *)line_data_buf.ptr;

    for (i64 i = 0; i < num_steps - 1; i++) {
      Vector3 a = steps[i].pos;
      Vector3 b = steps[i + 1].pos;

      *line_data_staging++ = Vector4::fromVec3W(a, 1.f);
      *line_data_staging++ = Vector4::fromVec3W(b, 1.f);
    }

    ParamBlock tmp_geo_block = raster_enc.createTemporaryParamBlock({
      .typeID = viz->agentPathsParamBlockType,
      .buffers = {
        { .buffer = line_data_buf.buffer, .offset = line_data_buf.offset, .numBytes = (u32)num_buffer_bytes },
      },
    });

    raster_enc.setParamBlock(1, tmp_geo_block);
    raster_enc.draw(0, total_num_verts);
  }
}

#ifdef DB_SUPPORT
static void renderAnalyticsViz(Engine &ctx, VizState *viz,
                               RasterPassEncoder &raster_enc)
{
  (void)ctx;
  AnalyticsDB &db = viz->db;

  if (db.currentVizMatchTimestep == -1) {
    return;
  }

  int num_shot_viz_lines = 0;
  {
    for (i32 i = 0; i < consts::maxTeamSize * 2; i++) {
      PlayerSnapshot player = db.curSnapshot.players[i];
      if (player.firedShot) {
        num_shot_viz_lines += 1;
      }
    }
  }

  if (num_shot_viz_lines > 0) {
    MappedTmpBuffer line_data_buf = raster_enc.tmpBuffer(
          sizeof(ShotVizLineData) * num_shot_viz_lines, 256);

    ShotVizLineData* out_lines = (ShotVizLineData*)line_data_buf.ptr;

    for (i32 i = 0; i < consts::maxTeamSize * 2; i++) {
      PlayerSnapshot player = db.curSnapshot.players[i];

      if (!player.firedShot) {
        continue;
      }

      Aim aim = computeAim(player.yaw, player.pitch);

      Vector3 a = player.pos;

      a.z += consts::standHeight;

      Vector3 dir = aim.rot.rotateVec(math::fwd);

      float hit_t = FLT_MAX;
      Entity hit_entity = Entity::none();
      traceRayAgainstWorld(ctx, a, dir, &hit_t, &hit_entity);

      i32 team_id = i / consts::maxTeamSize;

      float alpha = 0.75f;

      Vector3 color;
      if (team_id == 0) {
        color = { 0, 0, 1 };
      }
      else {
        color = { 1, 0, 0 };
      }

      if (hit_t == FLT_MAX) {
        hit_t = 10000;
      }

      if (hit_entity == Entity::none()) {
        color *= 0.5f;
        alpha *= 0.25f;
      }

      Vector3 b = a + dir * hit_t;

      *out_lines++ = {
        .start = a,
        .pad = {},
        .end = b,
        .pad2 = {},
        .color = Vector4::fromVec3W(color, alpha),
      };
    }

    ParamBlock tmp_geo_block = raster_enc.createTemporaryParamBlock({
      .typeID = viz->shotVizParamBlockType,
      .buffers = {{
        .buffer = line_data_buf.buffer,
        .offset = line_data_buf.offset,
      }},
    });

    raster_enc.setShader(viz->shotVizShader);
    raster_enc.setParamBlock(0, viz->globalParamBlock);
    raster_enc.setParamBlock(1, tmp_geo_block);

    raster_enc.draw(0, num_shot_viz_lines * 2);
  }

  {
    const auto &team_hulls = db.eventTeamConvexHulls;

    i32 total_num_verts =
      2 * (team_hulls[0].numVerts + team_hulls[1].numVerts);

    u32 num_tmp_vert_bytes = total_num_verts * sizeof(Vector3);
    MappedTmpBuffer tmp_verts = raster_enc.tmpBuffer(
        num_tmp_vert_bytes, 256 * 3);

    Vector3 *out_verts_ptr = (Vector3 *)tmp_verts.ptr;

    i32 total_num_tris = total_num_verts - 2;
    u32 num_tmp_tri_bytes = (u32)total_num_tris * 3 * sizeof(u32);

    MappedTmpBuffer tmp_tri_idxs = raster_enc.tmpBuffer(
        num_tmp_tri_bytes, sizeof(u32));

    u32 *out_tri_idxs_ptr = (u32 *)tmp_tri_idxs.ptr;

    i32 cur_vert_idx = 0;
    for (i32 team_idx = 0; team_idx < 2; team_idx++) {
      i32 hull_num_verts = team_hulls[team_idx].numVerts;
      i32 hull_base_vert_idx = cur_vert_idx;
      for (i32 i = 0; i < hull_num_verts; i++) {
        out_verts_ptr[cur_vert_idx] = Vector3 {
          .x = (f32)team_hulls[team_idx].verts[i].x,
          .y = (f32)team_hulls[team_idx].verts[i].y,
          .z = 0,
        };

        cur_vert_idx += 1;
      }

      for (i32 tri_fan_offset = 1; tri_fan_offset < hull_num_verts - 1;
           tri_fan_offset++) {
        *out_tri_idxs_ptr++ = hull_base_vert_idx;
        *out_tri_idxs_ptr++ = hull_base_vert_idx + tri_fan_offset;
        *out_tri_idxs_ptr++ = hull_base_vert_idx + tri_fan_offset + 1;
      }
    }

    raster_enc.setShader(viz->analyticsTeamHullShader);
    raster_enc.setParamBlock(0, viz->globalParamBlock);

    if (team_hulls[0].numVerts > 0) {
      raster_enc.drawData(AnalyticsTeamHullPerDraw {
        .color = { 0, 0, 1, 0.3 },
      });

      raster_enc.setVertexBuffer(0, tmp_verts.buffer);
      raster_enc.setIndexBufferU32(tmp_tri_idxs.buffer);
      raster_enc.drawIndexed(tmp_verts.offset / sizeof(Vector3),
                             tmp_tri_idxs.offset / sizeof(u32),
                             team_hulls[0].numVerts - 2);
    }

    if (team_hulls[1].numVerts > 0) {
      raster_enc.drawData(AnalyticsTeamHullPerDraw {
        .color = { 1, 0, 0, 0.3 },
      });

      raster_enc.setVertexBuffer(0, tmp_verts.buffer);
      raster_enc.setIndexBufferU32(tmp_tri_idxs.buffer);
      raster_enc.drawIndexed(tmp_verts.offset / sizeof(Vector3),
                             tmp_tri_idxs.offset / sizeof(u32) +
                               3 * (team_hulls[0].numVerts - 2),
                             team_hulls[1].numVerts - 2);
    }
  }
}
#endif

inline void mainMenuSystem(VizState *viz, std::string *out_path)
{
  ImGuiSystem::newFrame(viz->ui, viz->window->systemUIScale, 1.f / 60.f);

  auto viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(
    ImVec2(viewport->WorkSize.x / 2, viewport->WorkSize.y / 2),
    ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(100, 150));
  ImGui::Begin("Main Menu", nullptr,
               ImGuiWindowFlags_NoMove |
               ImGuiWindowFlags_NoTitleBar |
               ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::SetWindowFontScale(3.0f);

  float window_width = ImGui::GetWindowSize().x;
  float button_width = 400.0f; // Adjust width as needed
  float x_pos = (window_width - button_width) * 0.5f;

  if (out_path) {
    static std::array<char, 16384> path_str { "data/simple_map" };
    ImGui::SetCursorPosX(x_pos);
    ImGui::PushItemWidth(button_width);
    ImGui::InputText("Map Path", path_str.data(), path_str.size());
    ImGui::PopItemWidth();
    *out_path = path_str.data();

    ImGui::NewLine();
    ImGui::NewLine();
  }
  
  ImGui::SetCursorPosX(x_pos);

  ImGui::PushItemWidth(button_width);
  ImGui::DragFloat("Mouse Sensitivity", &viz->mouseSensitivity, 10.f, 1000.f,
                   ImGuiSliderFlags_AlwaysClamp);
  ImGui::PopItemWidth();
  ImGui::NewLine();
  ImGui::SetCursorPosX(x_pos);

  const char *play_button_name = viz->gameRunning ? "Resume" : "Play";

  if (ImGui::Button(play_button_name, ImVec2(button_width, 50))) {
    viz->mainMenu = false;
    viz->gameRunning = true;
    viz->simTickRate = 20;
    viz->curWorld = 0;
    viz->curView = 1;
    viz->curControl = 1;
  }

  ImGui::SetCursorPosX(x_pos);
  if (ImGui::Button("Top Down View", ImVec2(button_width, 50))) {
    viz->mainMenu = false;
    viz->gameRunning = true;
    viz->simTickRate = 20;
    viz->curWorld = 0;
    viz->curView = 0;
    viz->curControl = 0;
  }

  ImGui::PopStyleVar();
  ImGui::PopStyleVar();
  ImGui::End();

  GPUDevice *gpu = viz->gpu;
  viz->enc.beginEncoding();

  RasterPassEncoder pass = viz->enc.beginRasterPass(viz->mainmenuPass);

  ImGuiSystem::render(pass);

  viz->enc.endRasterPass(pass);

  viz->enc.endEncoding();
  gpu->submit(viz->mainQueue, viz->enc);
}

inline void renderSystem(Engine &ctx, VizState *viz, float delta_t)
{
  GPUDevice *gpu = viz->gpu;

  viz->enc.beginEncoding();

  {
    CopyPassEncoder copy_enc = viz->enc.beginCopyPass();
    MappedTmpBuffer global_param_staging =
        copy_enc.tmpBuffer(sizeof(GlobalPassData));

    GlobalPassData *global_param_staging_ptr =
      (GlobalPassData *)global_param_staging.ptr;

    Camera cam;
    {
      if (viz->curView == 0) {
        cam = viz->flyCam;
      } else {
        Entity agent = ctx.data().agents[viz->curView - 1];

        Vector3 pos = ctx.get<Position>(agent);
        Aim aim = ctx.get<Aim>(agent);

        StandState stand_state = ctx.get<StandState>(agent);

        if (stand_state.curPose == Pose::Stand) {
          pos.z += consts::standHeight;
        } else if (stand_state.curPose == Pose::Crouch) {
          pos.z += consts::crouchHeight;
        } else if (stand_state.curPose == Pose::Prone) {
          pos.z += consts::proneHeight;
        }

        pos.z -= 0.6f * consts::agentRadius;

        cam = {
          .position = pos,
          .fwd = aim.rot.rotateVec(math::fwd),
          .up = aim.rot.rotateVec(math::up),
          .right = aim.rot.rotateVec(math::right),
          .fine_aim = viz->flyCam.fine_aim,
        };
      }
    }

    setupViewData(ctx, cam, viz, global_param_staging_ptr);
    setupLightData(viz, global_param_staging_ptr);

    copy_enc.copyBufferToBuffer(
        global_param_staging.buffer, viz->globalPassDataBuffer,
        global_param_staging.offset, 0, sizeof(GlobalPassData));

    viz->enc.endCopyPass(copy_enc);
  }

  // ---  SCENE RENDERING  ---

  RasterPassEncoder offscreen_raster_enc = viz->enc.beginRasterPass(viz->offscreenPass);

  renderMap(viz, offscreen_raster_enc);
  renderObjects(ctx, viz, offscreen_raster_enc);
  renderAgents(ctx, viz, offscreen_raster_enc);
  //renderGoalRegions(ctx, viz, offscreen_raster_enc);
  (void)renderGoalRegions;
  renderShotViz(ctx, viz, offscreen_raster_enc, delta_t);
  renderZones(ctx, viz, offscreen_raster_enc);
  if ((ctx.data().simFlags & SimFlags::SubZones) == SimFlags::SubZones) {
    renderSubZones(ctx, viz, offscreen_raster_enc);
  }
  if (viz->curView == 0) {
    renderAgentPaths(viz, offscreen_raster_enc);
    if (viz->trajectoryDB) {
      trajectoryDBRender(viz, offscreen_raster_enc);
    }
  }

#ifdef DB_SUPPORT
  renderAnalyticsViz(ctx, viz, offscreen_raster_enc);
#endif

  viz->enc.endRasterPass(offscreen_raster_enc);

  // ---  POST EFFECTS  ---

  // ---  SSAO  ---

  viz->ssaoPass.Prepare(viz, ShaderID::SSAO, 1.0f, 1.0f, 1, false);
  viz->ssaoPass.AddDepthInput(viz->sceneDepth);
  viz->ssaoPass.SetParams(Vector4(0.0f, 0.0f, 0.0f, 0.0f));
  viz->ssaoPass.Execute( false );

  // Do a multi-pass bloom.
  for (int pass = 0; pass < DownsamplePasses; pass++)
  {
    // ---  DOWN SAMPLE  ---

    float downsample_factor = 0.25f / (pass + 1);
    viz->downsamplePasses[pass].Prepare(viz, ShaderID::Downsample, downsample_factor, downsample_factor, 1, false);
    viz->downsamplePasses[pass].AddTextureInput( pass == 0 ? viz->sceneColor : viz->downsamplePasses[pass-1].Output(0));
    viz->downsamplePasses[pass].SetParams(Vector4(4.0f, 4.0f, 0.0f, 0.0f)); // X and Y are downsample factors.
    viz->downsamplePasses[pass].Execute(false);

    // ---  BLOOM HORIZONTAL  ---

    viz->bloomHorizontalPasses[pass].Prepare(viz, ShaderID::Bloom, downsample_factor, downsample_factor, 1, false);
    viz->bloomHorizontalPasses[pass].AddTextureInput(viz->downsamplePasses[pass].Output(0));
    viz->bloomHorizontalPasses[pass].SetParams(Vector4(0.0f, 0.0f, 0.0f, 0.0f)); // 0 in X is horizontal pass.
    viz->bloomHorizontalPasses[pass].Execute(false);

    // ---  BLOOM VERTICAL  ---

    viz->bloomVerticalPasses[pass].Prepare(viz, ShaderID::Bloom, downsample_factor, downsample_factor, 1, false);
    viz->bloomVerticalPasses[pass].AddTextureInput(viz->bloomHorizontalPasses[pass].Output(0));
    viz->bloomVerticalPasses[pass].SetParams(Vector4(1.0f, 0.0f, 0.0f, 0.0f)); // 1 in X is vertical pass.
    viz->bloomVerticalPasses[pass].Execute(false);
  }

  // ---  COMPOSITE  ---

  viz->finalPass.Prepare(viz, ShaderID::PostEffect, 1.0f, 1.0f, 1, true);
  viz->finalPass.AddTextureInput(viz->ssaoPass.Output(0));
  for (int i = 0; i < DownsamplePasses; i++)
  {
    viz->finalPass.AddTextureInput(viz->bloomVerticalPasses[i].Output(0));
  }
  viz->finalPass.AddTextureInput(viz->heatmapTexture, true );
  viz->finalPass.AddTextureInput(viz->sceneColor);
  viz->finalPass.AddDepthInput(viz->sceneDepth);
  // The fog parameters are the input. Half-distance density, half-height, and height offset.
  viz->finalPass.SetParams(Vector4(200.0f, 1000.0f, viz->flyCam.mapMin.z + 50.0f, 0.0f));
  RasterPassEncoder final = viz->finalPass.Execute( true );

  // ---  UI  ---

  ImGuiSystem::render(final);
  viz->enc.endRasterPass(final);

  viz->enc.endEncoding();
  gpu->submit(viz->mainQueue, viz->enc);
}


void setupGameTasks(VizState *, TaskGraphBuilder &)
{
}

void vizStep(VizState *viz, Manager &mgr, float delta_t)
{
  if (viz->mainMenu) {
    mainMenuSystem(viz, nullptr);
  } else {
    Engine &ctx = uiLogic(viz, mgr);
    renderSystem(ctx, viz, delta_t);
  }
}

static bool bootmenu_tick(VizState *viz, std::string *map_path) {
  GPUDevice *gpu = viz->gpu;
  gpu->waitUntilReady(viz->mainQueue);

  auto [swapchain_tex, swapchain_status] =
    gpu->acquireSwapchainImage(viz->swapchain);
  assert(swapchain_status == SwapchainStatus::Valid);

  bool should_exit = viz->ui->processEvents();

  mainMenuSystem(viz, map_path);
  gpu->presentSwapchainImage(viz->swapchain);

  if (should_exit || (viz->window->state & WindowState::ShouldClose) != 
      WindowState::None) {
    *map_path = "";
    return false;
  } else if (!viz->mainMenu) {
      return false;
  }

  return true;
}

void bootMenu(VizState *viz, void (*cb)(VizState *, std::string scene_dir, void *), void *data_ptr)
{

  std::string map_path = "";

#ifdef EMSCRIPTEN
  static VizState *global_viz = viz;
  static void (*global_cb)(VizState *, std::string scene_dir, void *) = cb;
  static void *global_data_ptr = data_ptr;
  static std::string global_map_path = "";

  emscripten_set_main_loop([]() {
    bool running = bootmenu_tick(global_viz, &global_map_path);
    if (!running) {
      emscripten_cancel_main_loop();
      global_cb(global_viz, global_map_path, global_data_ptr);
    }
  }, 0, 0);
#else
  while (bootmenu_tick(viz, &map_path)) {}
  cb(viz, map_path, data_ptr);
#endif
}

void loadMapAssets(VizState *viz, const char *map_assets_path)
{
  AABB world_bounds;
  auto collision_data = importCollisionData(
      map_assets_path, Vector3::zero(), 0.f, &world_bounds);
  
  MapRenderableCollisionData map_render_data =
      convertCollisionDataToRenderMeshes(collision_data);

  GPUDevice *gpu = viz->gpu;
  CommandEncoder &enc = viz->enc;

  u32 total_num_bytes;
  {
    u32 cur_num_bytes = 0;
    for (const MapGeoMesh &in_mesh : map_render_data.meshes) {
      cur_num_bytes = utils::roundUp(cur_num_bytes, (u32)sizeof(MapGeoVertex));
      cur_num_bytes += sizeof(MapGeoVertex) * in_mesh.numVertices;
      cur_num_bytes += sizeof(u32) * in_mesh.numTris * 3;
    }
    
    total_num_bytes = cur_num_bytes;
  }

  Buffer staging = gpu->createStagingBuffer(total_num_bytes);
  Buffer map_buffer = gpu->createBuffer({
    .numBytes = total_num_bytes,
    .usage = BufferUsage::DrawVertex | BufferUsage::DrawIndex |
        BufferUsage::CopyDst | BufferUsage::ShaderStorage,
  });

  u8 *staging_ptr;
  gpu->prepareStagingBuffers(1, &staging, (void **)&staging_ptr);

  u32 cur_buf_offset = 0;
  for (const MapGeoMesh &in_mesh : map_render_data.meshes) {
    cur_buf_offset = utils::roundUp(cur_buf_offset, (u32)sizeof(MapGeoVertex));
    u32 vertex_offset = cur_buf_offset / sizeof(MapGeoVertex);

    MapGeoVertex *vertex_staging =
        (MapGeoVertex *)(staging_ptr + cur_buf_offset);

    for (i32 i = 0; i < (i32)in_mesh.numVertices; i++) {
      vertex_staging[i] = MapGeoVertex {
        .pos = map_render_data.positions[in_mesh.vertOffset + i],
      };
    }

    cur_buf_offset += sizeof(MapGeoVertex) * in_mesh.numVertices;

    u32 index_offset = cur_buf_offset / sizeof(u32);
    u32 *indices_staging = (u32 *)(staging_ptr + cur_buf_offset);

    u32 num_index_bytes = sizeof(u32) * in_mesh.numTris * 3;
    memcpy(indices_staging, map_render_data.indices.data() + in_mesh.indexOffset,
           num_index_bytes);
    cur_buf_offset += num_index_bytes;

    viz->mapMeshes.push_back({
      .vertOffset = vertex_offset,
      .indexOffset = index_offset, 
      .numVertices = in_mesh.numVertices,
      .numTris = in_mesh.numTris,
    });
  }
  
  assert(cur_buf_offset == total_num_bytes);

  gpu->flushStagingBuffers(1, &staging);

  gpu->waitUntilReady(viz->mainQueue);

  {
    enc.beginEncoding();
    CopyPassEncoder copy_enc = enc.beginCopyPass();

    copy_enc.copyBufferToBuffer(staging, map_buffer, 0, 0, total_num_bytes);

    enc.endCopyPass(copy_enc);
    enc.endEncoding();
  }

  gpu->submit(viz->mainQueue, enc);
  gpu->waitUntilWorkFinished(viz->mainQueue);

  gpu->destroyStagingBuffer(staging);

  assert(viz->mapBuffer.null());
  viz->mapBuffer = map_buffer;

  viz->mapGeoParamBlock = gpu->createParamBlock({
    .typeID = viz->mapGeoParamBlockType,
    .buffers = {
      { .buffer = viz->mapBuffer },
      { .buffer = viz->mapBuffer },
    },
  });
}

}

}
