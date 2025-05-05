#include "map_importer.hpp"
#include <madrona/stack_alloc.hpp>
#include <madrona/physics_assets.hpp>
#include "mesh_bvh_builder.hpp"

#include <fstream>
#include <meshoptimizer.h>

using namespace madrona;

namespace madronaMPEnv {

using madrona::math::Vector2;
using madrona::math::Vector3;
using madrona::math::Quat;
using madrona::math::AABB;

namespace {

struct ImportedMaterialData {
    HeapArray<char> materialNameBuffer;
    HeapArray<const char *> materialNames;
    HeapArray<CollisionMaterialFlags> materialFlags;
};

struct ImportedMeshData {
    HeapArray<Vector3> vertices;
    HeapArray<uint32_t> indices;
    HeapArray<uint32_t> triCollisionMaterials;
    HeapArray<MapCollisionAssets::MeshInfo> meshes;
};

}

static ImportedMaterialData readMaterials(std::ifstream &file)
{
    uint64_t num_materials;
    file.read((char *)&num_materials, sizeof(uint64_t));

    uint64_t num_material_name_bytes;
    file.read((char *)&num_material_name_bytes, sizeof(uint64_t));

    HeapArray<char> material_name_buffer((CountT)num_material_name_bytes);
    file.read(material_name_buffer.data(), num_material_name_bytes);

    HeapArray<const char *> material_names((CountT)num_materials);
    const char *cur_material_name_ptr = material_name_buffer.data();
    for (CountT i = 0; i < (CountT)num_materials; i++) {
        material_names[i] = cur_material_name_ptr;
        size_t num_chars = strlen(material_names[i]);
        cur_material_name_ptr += num_chars + 1;
    }

    HeapArray<CollisionMaterialFlags> materials(num_materials);
    file.read((char *)materials.data(),
              sizeof(CollisionMaterialFlags) * num_materials);

    return ImportedMaterialData {
        .materialNameBuffer = std::move(material_name_buffer),
        .materialNames = std::move(material_names),
        .materialFlags = std::move(materials),
    };
}

static ImportedMeshData readMeshes(std::ifstream &file)
{
    uint64_t num_meshes;
    file.read((char *)&num_meshes, sizeof(uint64_t));

    uint64_t total_num_verts;
    file.read((char *)&total_num_verts, sizeof(uint64_t));

    uint64_t total_num_tris;
    file.read((char *)&total_num_tris, sizeof(uint64_t));

    HeapArray<MapCollisionAssets::MeshInfo> meshes(num_meshes);
    HeapArray<Vector3> vertices(total_num_verts);
    HeapArray<uint32_t> indices(total_num_tris * 3);
    HeapArray<uint32_t> tri_collision_mats(total_num_tris);

    file.read((char *)vertices.data(), vertices.size() * sizeof(Vector3));
    file.read((char *)indices.data(), indices.size() * sizeof(uint32_t));
    file.read((char *)tri_collision_mats.data(),
              tri_collision_mats.size() * sizeof(uint32_t));

    file.read((char *)meshes.data(),
              meshes.size() * sizeof(MapCollisionAssets::MeshInfo));

#if 0
    HeapArray<Vector3> vertices({
        Vector3 { -1000, -1000, 800 },
        Vector3 { 1000, -1000, 800 },
        Vector3 { 1000, 1000, 800 },
        Vector3 { -1000, 1000, 800 },

        Vector3 { -100, 20, 800 },
        Vector3 { 100, 20, 800 },
        Vector3 { 100, 20 , 870 },
        Vector3 { -100, 20, 870 },
    });

    HeapArray<uint32_t> indices({
        0, 1, 3,
        1, 2, 3,
        4, 5, 7,
        5, 6, 7,
    });

    HeapArray<uint32_t> tri_collision_mats {
        0, 0,
    };

    HeapArray<CollisionAssets::MeshInfo> meshes({
        { 0, (uint32_t)vertices.size(), 0, (uint32_t)indices.size() / 3, }
    });
#endif

    return ImportedMeshData {
        .vertices = std::move(vertices),
        .indices = std::move(indices),
        .triCollisionMaterials = std::move(tri_collision_mats),
        .meshes = std::move(meshes),
    };
}

static ImportedMeshData filterMeshes(
    const ImportedMeshData &orig,
    const ImportedMaterialData &materials,
    AABB world_bounds)
{
    DynArray<Vector3> new_vertices(orig.vertices.size());
    DynArray<uint32_t> new_indices(orig.indices.size());
    DynArray<uint32_t> new_collision_mats(orig.triCollisionMaterials.size());
    DynArray<MapCollisionAssets::MeshInfo> new_meshes(orig.meshes.size());

    for (const MapCollisionAssets::MeshInfo &orig_mesh : orig.meshes) {
        uint32_t new_vert_offset = new_vertices.size();
        uint32_t new_tri_offset = new_indices.size() / 3;

        for (uint32_t i = 0; i < orig_mesh.numTris; i++) {
            uint32_t tri_mat_idx =
                orig.triCollisionMaterials[i + orig_mesh.triOffset];

            CollisionMaterialFlags material_flags =
                materials.materialFlags[tri_mat_idx];

            if (material_flags == CollisionMaterialFlags::BulletsOnly) {
                continue;
            }

            uint32_t a_i = 3 * (orig_mesh.triOffset + i);
            uint32_t b_i = a_i + 1;
            uint32_t c_i = a_i + 2;

            uint32_t orig_tri_a_idx = orig_mesh.vertexOffset + orig.indices[a_i];
            uint32_t orig_tri_b_idx = orig_mesh.vertexOffset + orig.indices[b_i];
            uint32_t orig_tri_c_idx = orig_mesh.vertexOffset + orig.indices[c_i];

            Vector3 a = orig.vertices[orig_tri_a_idx];
            Vector3 b = orig.vertices[orig_tri_b_idx];
            Vector3 c = orig.vertices[orig_tri_c_idx];

            if (!world_bounds.contains(a) &&
                    !world_bounds.contains(b) &&
                    !world_bounds.contains(c)) {
                //continue;
            }

            uint32_t new_tri_a_idx = new_vertices.size() - new_vert_offset;
            new_vertices.push_back(a);
            uint32_t new_tri_b_idx = new_vertices.size() - new_vert_offset;
            new_vertices.push_back(b);
            uint32_t new_tri_c_idx = new_vertices.size() - new_vert_offset;
            new_vertices.push_back(c);

            new_indices.push_back(new_tri_a_idx);
            new_indices.push_back(new_tri_b_idx);
            new_indices.push_back(new_tri_c_idx);

            new_collision_mats.push_back(tri_mat_idx);
        }

        uint32_t num_verts = new_vertices.size() - new_vert_offset;
        uint32_t num_tris = new_indices.size() / 3 - new_tri_offset;

        if (num_tris == 0) {
            assert(num_verts == 0);
            continue;
        }

        new_meshes.push_back({
            .vertexOffset = new_vert_offset,
            .numVertices = num_verts,
            .triOffset = new_tri_offset,
            .numTris = num_tris,
        });
    }

    HeapArray<Vector3> out_vertices(new_vertices.size());
    memcpy(out_vertices.data(), new_vertices.data(),
           new_vertices.size() * sizeof(Vector3));

    HeapArray<uint32_t> out_indices(new_indices.size());
    memcpy(out_indices.data(), new_indices.data(),
           new_indices.size() * sizeof(uint32_t));

    HeapArray<uint32_t> out_tri_mats(new_collision_mats.size());
    memcpy(out_tri_mats.data(), new_collision_mats.data(),
           new_collision_mats.size() * sizeof(uint32_t));

    HeapArray<MapCollisionAssets::MeshInfo> out_meshes(new_meshes.size());
    memcpy(out_meshes.data(), new_meshes.data(),
           new_meshes.size() * sizeof(MapCollisionAssets::MeshInfo));

    return {
        .vertices = std::move(out_vertices),
        .indices = std::move(out_indices),
        .triCollisionMaterials = std::move(out_tri_mats),
        .meshes = std::move(out_meshes),
    };
}

MapCollisionAssets importCollisionData(const char *path,
                                           Vector3 global_translation,
                                           float rot_around_z,
                                           AABB *world_bounds)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
      FATAL("Failed to open %s", path);
    }

    file.read((char *)world_bounds, sizeof(AABB));

    auto mat_data = readMaterials(file);
    auto mesh_data = readMeshes(file);

    Quat global_rotation = Quat::angleAxis(
        math::toRadians(rot_around_z), math::up).inv();
    
    for (Vector3 &v : mesh_data.vertices) {
        v = global_rotation.rotateVec(v - global_translation);
    }

    auto filtered_meshes = filterMeshes(mesh_data, mat_data, *world_bounds);
    
    return MapCollisionAssets {
        .materials = std::move(mat_data.materialFlags),
        .materialNameBuffer = std::move(mat_data.materialNameBuffer),
        .materialNames = std::move(mat_data.materialNames),
        .vertices = std::move(filtered_meshes.vertices),
        .indices = std::move(filtered_meshes.indices),
        .triCollisionMaterials = std::move(filtered_meshes.triCollisionMaterials),
        .meshes = std::move(filtered_meshes.meshes),
    };
}

MapRenderableCollisionData convertCollisionDataToRenderMeshes(
    const MapCollisionAssets &collision_data)
{
  using namespace madrona::imp;

  using VertexType = Vector3;

  DynArray<VertexType> all_verts(0);
  DynArray<uint32_t> all_indices(0);
  HeapArray<MapGeoMesh> meshes_out(collision_data.meshes.size());

  for (uint32_t mesh_idx = 0; mesh_idx < (uint32_t)collision_data.meshes.size();
       mesh_idx++) {
    const MapCollisionAssets::MeshInfo &mesh = collision_data.meshes[mesh_idx];
    const Vector3 *mesh_verts = collision_data.vertices.data() +
        mesh.vertexOffset;

    const uint32_t *mesh_indices = collision_data.indices.data() +
        mesh.triOffset * 3;

    HeapArray<VertexType> unindexed_verts(mesh.numTris * 3);
    for (uint32_t i = 0; i < mesh.numTris; i++) {
      uint32_t a_i = 3 * i;
      uint32_t b_i = 3 * i + 1;
      uint32_t c_i = 3 * i + 2;

      uint32_t a_read_idx = mesh_indices[a_i];
      uint32_t b_read_idx = mesh_indices[b_i];
      uint32_t c_read_idx = mesh_indices[c_i];

      Vector3 a = mesh_verts[a_read_idx];
      Vector3 b = mesh_verts[b_read_idx];
      Vector3 c = mesh_verts[c_read_idx];

      unindexed_verts[a_i] = a;
      unindexed_verts[b_i] = b;
      unindexed_verts[c_i] = c;
    }

    HeapArray<uint32_t> vert_remap(mesh.numTris * 3);
    uint32_t num_unique_verts = meshopt_generateVertexRemap(
      vert_remap.data(), nullptr, unindexed_verts.size(),
      unindexed_verts.data(), unindexed_verts.size(), sizeof(VertexType));

    HeapArray<VertexType> unique_verts(num_unique_verts);
    meshopt_remapVertexBuffer(unique_verts.data(), unindexed_verts.data(),
      unindexed_verts.size(), sizeof(VertexType), vert_remap.data());

    HeapArray<uint32_t> unique_vert_idxs(mesh.numTris * 3);
    meshopt_remapIndexBuffer(
      unique_vert_idxs.data(), nullptr, unindexed_verts.size(), vert_remap.data());
    
    HeapArray<uint32_t> cache_optimized_index_buffer(unique_vert_idxs.size());
    meshopt_optimizeVertexCache(cache_optimized_index_buffer.data(),
      unique_vert_idxs.data(), unique_vert_idxs.size(), unique_verts.size());

    HeapArray<uint32_t> provoking_index_buffer(cache_optimized_index_buffer.size());

    HeapArray<uint32_t> provoking_reorder(num_unique_verts + mesh.numTris);

    size_t num_reorder = meshopt_generateProvokingIndexBuffer(
      provoking_index_buffer.data(), provoking_reorder.data(),
      cache_optimized_index_buffer.data(), cache_optimized_index_buffer.size(),
      num_unique_verts);

    HeapArray<VertexType> provoking_reordered_verts(num_reorder);
    for (uint32_t i = 0; i < (uint32_t)num_reorder; i++) {
      uint32_t reorder_idx = provoking_reorder[i];
      provoking_reordered_verts[i] = unique_verts[reorder_idx];
    }

    MapGeoMesh mesh_out {
      .vertOffset = (uint32_t)all_verts.size(),
      .indexOffset = (uint32_t)all_indices.size(),
      .numVertices = (uint32_t)provoking_reordered_verts.size(),
      .numTris = mesh.numTris,
    };

    for (uint32_t i = 0; i < (uint32_t)provoking_reordered_verts.size(); i++) {
      all_verts.push_back(provoking_reordered_verts[i]);
    }

    for (uint32_t i = 0; i < mesh.numTris * 3; i++) {
      all_indices.push_back(provoking_index_buffer[i]);
    }

    meshes_out[mesh_idx] = mesh_out;
  }

  HeapArray<VertexType> verts_out(all_verts.size());
  for (uint32_t i = 0; i < all_verts.size(); i++) {
    verts_out[i] = all_verts[i];
  }

  HeapArray<uint32_t> idxs_out(all_indices.size());
  for (uint32_t i = 0; i < all_indices.size(); i++) {
    idxs_out[i] = all_indices[i];
  }

  return MapRenderableCollisionData {
      .positions = std::move(verts_out),
      .indices = std::move(idxs_out),
      .meshes = std::move(meshes_out),
  };
}

void * buildMeshBVH(
    MapCollisionAssets &collision_data,
    MeshBVH *out_bvh)
{
    HeapArray<imp::SourceMesh> src_meshes(collision_data.meshes.size());

    for (CountT i = 0; i < collision_data.meshes.size(); i++) {
        MapCollisionAssets::MeshInfo &mesh_info =
            collision_data.meshes[i];
        src_meshes[i] = imp::SourceMesh {
            .positions = collision_data.vertices.data() +
                mesh_info.vertexOffset,
            .normals = nullptr,
            .tangentAndSigns = nullptr,
            .uvs = nullptr,
            .indices = collision_data.indices.data() +
                mesh_info.triOffset * 3,
            .faceCounts = nullptr,
            .faceMaterials = nullptr,
            .numVertices = mesh_info.numVertices,
            .numFaces = mesh_info.numTris,
            .materialIDX = 0,
        };
    }

    MeshBVH bvh = MeshBVHBuilder::build(src_meshes);

    int64_t num_node_bytes = (int64_t)sizeof(MeshBVH::Node) * bvh.numNodes;
    int64_t num_mat_bytes = (int64_t)sizeof(MeshBVH::LeafMaterial) * bvh.numLeaves;
    int64_t num_vert_bytes = (int64_t)sizeof(MeshBVH::BVHVertex) * bvh.numVerts;

    int64_t buffer_offsets[2];
    size_t total_bvh_buffer_bytes = utils::computeBufferOffsets(
        { num_node_bytes, num_mat_bytes, num_vert_bytes },
        buffer_offsets, 64);

    char *mesh_bvh_buffer = (char *)malloc(total_bvh_buffer_bytes);

    memcpy(mesh_bvh_buffer, bvh.nodes, num_node_bytes);
    memcpy(mesh_bvh_buffer + buffer_offsets[0], bvh.leafMats, num_mat_bytes);
    memcpy(mesh_bvh_buffer + buffer_offsets[1], bvh.vertices, num_vert_bytes);

    out_bvh->nodes = (MeshBVH::Node *)mesh_bvh_buffer;
    out_bvh->leafMats =
        (MeshBVH::LeafMaterial *)(mesh_bvh_buffer + buffer_offsets[0]);
    out_bvh->vertices =
        (MeshBVH::BVHVertex *)(mesh_bvh_buffer + buffer_offsets[1]);

    out_bvh->rootAABB = bvh.rootAABB;
    out_bvh->numNodes = bvh.numNodes;
    out_bvh->numLeaves = bvh.numLeaves;
    out_bvh->numVerts = bvh.numVerts;
    out_bvh->magic = bvh.magic;

    return mesh_bvh_buffer;
}

MapNavmesh importNavmesh(const char *path, AABB world_bounds)
{
    (void)world_bounds;

    std::ifstream navmesh_file(path, std::ios::binary);
    assert(navmesh_file.is_open());

    uint32_t num_verts;
    navmesh_file.read((char *)&num_verts, sizeof(uint32_t));

    HeapArray<Vector3> verts(num_verts);
    navmesh_file.read((char *)verts.data(), sizeof(Vector3) * num_verts);

    uint32_t num_faces;
    navmesh_file.read((char *)&num_faces, sizeof(uint32_t));

    HeapArray<uint32_t> face_counts(num_faces);
    navmesh_file.read((char *)face_counts.data(), sizeof(uint32_t) * num_faces);

    uint32_t num_indices;
    navmesh_file.read((char *)&num_indices, sizeof(uint32_t));

    HeapArray<uint32_t> indices(num_indices);
    navmesh_file.read((char *)indices.data(), sizeof(uint32_t) * num_indices);

    HeapArray<uint32_t> face_starts(num_faces);

    uint32_t cur_face_offset = 0;
    for (uint32_t i = 0; i < num_faces; i++) {
        face_starts[i] = cur_face_offset;

        cur_face_offset += face_counts[i];
    }

    // HACK: remove out of range navmeshes tiles
    DynArray<uint32_t> new_face_counts(num_faces);
    DynArray<uint32_t> new_face_starts(num_faces);
    DynArray<uint32_t> new_indices(num_indices);

    for (uint32_t i = 0; i < num_faces; i++) {
        uint32_t start_idx = face_starts[i];
        uint32_t num_face_verts = face_counts[i];

        bool valid = true;
#if 0
        for (uint32_t j = 0; j < num_face_verts; j++) {
            uint32_t idx = indices[start_idx + j];
            Vector3 v = verts[idx];

            if (world_bounds.contains(v)) {
                valid = false;
                break;
            }
        }
#endif

        if (valid) {
            new_face_starts.push_back((uint32_t)new_indices.size());
            new_face_counts.push_back(num_face_verts);
            for (uint32_t j = 0; j < num_face_verts; j++) {
                uint32_t idx = indices[start_idx + j];
                new_indices.push_back(idx);
            }
        }
    }

    face_counts = HeapArray<uint32_t>(new_face_counts.size());
    face_starts = HeapArray<uint32_t>(new_face_starts.size());
    indices = HeapArray<uint32_t>(new_indices.size());

    memcpy(face_counts.data(), new_face_counts.data(),
           new_face_counts.size() * sizeof(uint32_t));

    memcpy(face_starts.data(), new_face_starts.data(),
           new_face_starts.size() * sizeof(uint32_t));

    memcpy(indices.data(), new_indices.data(),
           new_indices.size() * sizeof(uint32_t));

    return MapNavmesh {
        .verts = std::move(verts),
        .indices = std::move(indices),
        .faceStarts = std::move(face_starts),
        .faceCounts = std::move(face_counts),
    };
}

MapSpawnData loadMapSpawnData(std::filesystem::path spawn_file_path)
{

    std::ifstream spawn_file(spawn_file_path, std::ios::binary);

    uint32_t num_a_spawns;
    spawn_file.read((char *)&num_a_spawns, sizeof(uint32_t));

    DynArray<Spawn> a_spawns(num_a_spawns);
    a_spawns.resize(num_a_spawns, [](auto) {});
    spawn_file.read((char *)a_spawns.data(), sizeof(Spawn) * num_a_spawns);

    uint32_t num_b_spawns;
    spawn_file.read((char *)&num_b_spawns, sizeof(uint32_t));

    DynArray<Spawn> b_spawns(num_b_spawns);
    b_spawns.resize(num_b_spawns, [](auto) {});
    spawn_file.read((char *)b_spawns.data(), sizeof(Spawn) * num_b_spawns);

    uint32_t num_common_respawns;
    spawn_file.read((char *)&num_common_respawns, sizeof(uint32_t));

    DynArray<Spawn> common_respawns(num_common_respawns);
    common_respawns.resize(num_common_respawns, [](auto) {});
    spawn_file.read((char *)common_respawns.data(),
                    sizeof(Spawn) * num_common_respawns);

    DynArray<RespawnRegion> respawn_regions(0);

    return MapSpawnData {
        .aSpawns = std::move(a_spawns),
        .bSpawns = std::move(b_spawns),
        .commonRespawns = std::move(common_respawns),
        .respawnRegions = std::move(respawn_regions),
    };
}

ZoneData loadMapZones(std::filesystem::path zone_file_path)
{
    std::ifstream zones_file(zone_file_path, std::ios::binary);

    uint32_t num_zones;
    zones_file.read((char *)&num_zones, sizeof(uint32_t));

    DynArray<AABB> zone_aabbs(num_zones);
    zone_aabbs.resize(num_zones, [](auto){});
    DynArray<float> zone_rotations(num_zones);
    zone_rotations.resize(num_zones, [](auto){});

    zones_file.read((char *)zone_aabbs.data(),
                         sizeof(AABB) * num_zones);

    zones_file.read((char *)zone_rotations.data(),
                         sizeof(float) * num_zones);

    return ZoneData {
        .aabbs = std::move(zone_aabbs),
        .rotations = std::move(zone_rotations),
    };
}

TDMEpisode *loadEpisodeData(
    std::filesystem::path episode_data_path, uint32_t *num_out)
{
    uint32_t num_episodes;
    std::ifstream episodes_file(episode_data_path, std::ios::binary);
    episodes_file.read((char *)&num_episodes, sizeof(uint32_t));
    TDMEpisode *episodes = (TDMEpisode *)malloc(sizeof(TDMEpisode) * num_episodes);
    episodes_file.read((char *)episodes, sizeof(TDMEpisode) * num_episodes);

    *num_out = num_episodes;
    return episodes;
}

}
