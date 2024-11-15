#pragma once

#include <madrona/importer.hpp>
#include <madrona/heap_array.hpp>

#include "mesh_bvh.hpp"
#include "types.hpp"

#include <filesystem>

namespace madronaMPEnv {

enum class CollisionMaterialFlags : uint32_t {
    None,
    BulletsOnly = 1 << 0,
};

struct MapCollisionAssets {
    struct MeshInfo {
        uint32_t vertexOffset;
        uint32_t numVertices;
        uint32_t triOffset;
        uint32_t numTris;
    };

    madrona::HeapArray<CollisionMaterialFlags> materials;
    madrona::HeapArray<char> materialNameBuffer;
    madrona::HeapArray<const char *> materialNames;

    madrona::HeapArray<madrona::math::Vector3> vertices;
    madrona::HeapArray<uint32_t> indices;
    madrona::HeapArray<uint32_t> triCollisionMaterials;
    madrona::HeapArray<MeshInfo> meshes;
};

struct MapRenderableCollisionData {
    madrona::HeapArray<madrona::math::Vector3> positions;
    madrona::HeapArray<madrona::math::Vector3> normals;
    madrona::HeapArray<madrona::math::Vector2> uvs;
    madrona::HeapArray<uint32_t> indices;

    madrona::HeapArray<madrona::imp::SourceMesh> meshes;
    madrona::HeapArray<madrona::imp::SourceObject> objects;
};

struct MapNavmesh {
    madrona::HeapArray<madrona::math::Vector3> verts;
    madrona::HeapArray<uint32_t> indices;
    madrona::HeapArray<uint32_t> faceStarts;
    madrona::HeapArray<uint32_t> faceCounts;
};

struct MapSpawnData {
    DynArray<Spawn> aSpawns;
    DynArray<Spawn> bSpawns;
    DynArray<Spawn> commonRespawns;
    DynArray<RespawnRegion> respawnRegions;
};

struct ZoneData {
    DynArray<AABB> aabbs;
    DynArray<float> rotations;
};

MapCollisionAssets importCollisionData(
    const char *path, madrona::math::Vector3 translation, float rot_around_z,
    madrona::math::AABB *world_bounds);

MapRenderableCollisionData convertCollisionDataToRenderMeshes(
    const MapCollisionAssets &collision_data);

void * buildMeshBVH(
    MapCollisionAssets &collision_data,
    MeshBVH *out_bvh);

MapNavmesh importNavmesh(const char *path,
    madrona::math::AABB world_bounds);

MapSpawnData loadMapSpawnData(std::filesystem::path spawn_file_path);

ZoneData loadMapZones(std::filesystem::path zone_file_path);

TDMEpisode * loadEpisodeData(
    std::filesystem::path episode_data_path, uint32_t *num_out);

}
