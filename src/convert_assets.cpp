#include "map_importer.hpp"
#include "nav/nav_build.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <madrona/importer.hpp>

using namespace madrona;
using namespace madronaMPEnv;

int main(int argc, char *argv[])
{
  if (argc < 2) {
    fprintf(stderr, "%s DIR\n", argv[0]);
    exit(1);
  }

  imp::AssetImporter importer;

  std::string dir_str(argv[1]);

  auto convertPosition =
    [](Vector3 v)
  {
    return v * 39.3701f;
  };

  StackAlloc tmp_alloc;

  {
    std::string collisions_in_str = dir_str + "/collisions.obj";
    auto collisions_in_c_str = collisions_in_str.c_str();

    std::array<char, 1024> import_err;
    auto imported_assets = importer.importFromDisk(
      Span(&collisions_in_c_str, 1),
      Span(import_err.data(), import_err.size()), true);

    if (!imported_assets.has_value()) {
      FATAL("Failed to load: %s", import_err.data());
    }
    
    imp::SourceObject &meter_src_obj = imported_assets->objects[0];

    printf("%lu\n", meter_src_obj.meshes.size());

    u64 num_meshes = meter_src_obj.meshes.size();

    imp::SourceObject src_obj {
      .meshes =
          Span(tmp_alloc.allocN<imp::SourceMesh>(num_meshes), num_meshes),
    };

    HeapArray<MapCollisionAssets::MeshInfo> collision_meshinfos(num_meshes);
    AABB world_bounds = AABB::invalid();
    u64 total_num_verts = 0;
    u64 total_num_tris = 0;

    for (u64 i = 0; i < num_meshes; i++) {
      const auto &in_mesh = meter_src_obj.meshes[i];
      auto &out_mesh = src_obj.meshes[i];

      out_mesh = in_mesh;
      out_mesh.positions = tmp_alloc.allocN<Vector3>(out_mesh.numVertices);

      for (u32 j = 0; j < out_mesh.numVertices; j++) {
        out_mesh.positions[j] = convertPosition(in_mesh.positions[j]);
        world_bounds.expand(out_mesh.positions[j]);
      }

      collision_meshinfos[i] = {
        .vertexOffset = (u32)total_num_verts,
        .numVertices = out_mesh.numVertices,
        .triOffset = (u32)total_num_tris,
        .numTris = out_mesh.numFaces,
      };

      assert(out_mesh.faceCounts == nullptr);
      total_num_verts += out_mesh.numVertices;
      total_num_tris += out_mesh.numFaces;
    }

    std::ofstream collisions_out(dir_str + "/collisions.bin", std::ios::binary);
    assert(collisions_out.is_open() && collisions_out.good());

    collisions_out.write((char *)&world_bounds, sizeof(AABB));

    u64 num_mats = 1;
    collisions_out.write((char *)&num_mats, sizeof(u64));

    const char *mat_name = "a";
    u64 num_mat_name_bytes = strlen(mat_name) + 1;
    collisions_out.write((char *)&num_mat_name_bytes, sizeof(u64));
    collisions_out.write(mat_name, num_mat_name_bytes);

    CollisionMaterialFlags flags = CollisionMaterialFlags::None;
    collisions_out.write((char *)&flags, sizeof(CollisionMaterialFlags));

    collisions_out.write((char *)&num_meshes, sizeof(u64));
    collisions_out.write((char *)&total_num_verts, sizeof(u64));
    collisions_out.write((char *)&total_num_tris, sizeof(u64));

    for (const auto &src_mesh : src_obj.meshes) {
      collisions_out.write((char *)src_mesh.positions,
                           sizeof(Vector3) * src_mesh.numVertices);
    }

    for (const auto &src_mesh : src_obj.meshes) {
      collisions_out.write((char *)src_mesh.indices,
                           sizeof(u32) * src_mesh.numFaces * 3);
    }

    for (const auto &src_mesh : src_obj.meshes) {
      HeapArray<u32> face_collision_mats(src_mesh.numFaces);
      for (CountT i = 0; i < face_collision_mats.size(); i++) {
        face_collision_mats[i] = 0;
      }

      collisions_out.write((char *)face_collision_mats.data(),
                           sizeof(u32) * src_mesh.numFaces);
    }

    collisions_out.write((char *)collision_meshinfos.data(),
        sizeof(MapCollisionAssets::MeshInfo) * num_meshes);

    std::ofstream navmesh_out(dir_str + "/navmesh.bin", std::ios::binary);
    assert(navmesh_out.is_open() && navmesh_out.good());

    Navmesh navmesh = createMadronaNavmesh(buildNavmeshFromSourceObjects(
        Span<const imp::SourceObject>(&src_obj, 1)));

    u32 num_verts = navmesh.numVerts;

    navmesh_out.write((char *)&num_verts, sizeof(u32));

    navmesh_out.write((char *)navmesh.vertices, sizeof(Vector3) * num_verts);

    u32 num_tris = navmesh.numTris;

    navmesh_out.write((char *)&num_tris, sizeof(u32));
    HeapArray<u32> face_counts(num_tris);
    for (u32 i = 0; i < num_tris; i++) {
      face_counts[i] = 3;
    }

    navmesh_out.write((char *)face_counts.data(), sizeof(u32) * num_tris);
    u32 num_indices = num_tris * 3;
    navmesh_out.write((char *)&num_indices, sizeof(u32));
    navmesh_out.write((char *)navmesh.triIndices, num_indices * sizeof(u32));

    tmp_alloc.release();
  }

  {
    std::string spawns_in_str = dir_str + "/spawns.obj";
    auto spawns_in_c_str = spawns_in_str.c_str();

    std::array<char, 1024> import_err;
    auto imported_assets = importer.importFromDisk(
      Span(&spawns_in_c_str, 1),
      Span(import_err.data(), import_err.size()), true);

    if (!imported_assets.has_value()) {
      FATAL("Failed to load: %s", import_err.data());
    }
    
    imp::SourceObject &src_obj = imported_assets->objects[0];

    u32 num_spawns = src_obj.meshes.size();

    printf("%u\n", num_spawns);

    assert(num_spawns % 2 == 0);

    Spawn *all_spawns = tmp_alloc.allocN<Spawn>(num_spawns);

    for (u32 i = 0; i < num_spawns; i++) {
      const imp::SourceMesh &src_mesh = src_obj.meshes[i];
      Vector3 spawn_pos = Vector3::zero();
      for (u32 j = 0; j < src_mesh.numVertices; j++) {
        spawn_pos += (1.f / src_mesh.numVertices) * src_mesh.positions[j];
      }
      spawn_pos = convertPosition(spawn_pos);

      float yaw_min, yaw_max;
      if (spawn_pos.x < 0.f) {
        yaw_min = 0.25f * math::pi;
        yaw_max = 0.75f * math::pi;
      } else {
        yaw_min = -0.75f * math::pi;
        yaw_max = -0.25f * math::pi;
      }

      all_spawns[i] = {
        .region = {
          .pMin = spawn_pos,
          .pMax = spawn_pos,
        },
        .yawMin = yaw_min,
        .yawMax = yaw_max,
      };
    }

    u32 num_a_spawns = num_spawns / 2;
    u32 num_b_spawns = num_a_spawns;

    std::ofstream spawns_out(dir_str + "/spawns.bin", std::ios::binary);
    assert(spawns_out.is_open() && spawns_out.good());

    spawns_out.write((char *)&num_a_spawns, sizeof(u32));
    spawns_out.write((char *)all_spawns, sizeof(Spawn) * num_a_spawns);

    spawns_out.write((char *)&num_b_spawns, sizeof(u32));
    spawns_out.write((char *)(all_spawns + num_a_spawns),
                     sizeof(Spawn) * num_b_spawns);

    spawns_out.write((char *)&num_spawns, sizeof(u32));
    spawns_out.write((char *)all_spawns, sizeof(Spawn) * num_spawns);
  }

  {
    std::string zones_in_str = dir_str + "/zones.obj";
    auto zones_in_c_str = zones_in_str.c_str();

    std::array<char, 1024> import_err;
    auto imported_assets = importer.importFromDisk(
      Span(&zones_in_c_str, 1),
      Span(import_err.data(), import_err.size()), true);

    if (!imported_assets.has_value()) {
      FATAL("Failed to load: %s", import_err.data());
    }
    
    imp::SourceObject &src_obj = imported_assets->objects[0];

    u32 num_zones = src_obj.meshes.size();

    printf("%u\n", num_zones);

    AABB *zone_aabbs = tmp_alloc.allocN<AABB>(num_zones);
    float *zone_rotations = tmp_alloc.allocN<float>(num_zones);

    for (u32 i = 0; i < num_zones; i++) {
      imp::SourceMesh &src_mesh = src_obj.meshes[i];

      AABB zone_aabb = AABB::invalid();
      for (u32 j = 0; j < src_mesh.numVertices; j++) {
        zone_aabb.expand(convertPosition(src_mesh.positions[j]));
      }

      zone_aabbs[i] = zone_aabb;
      zone_rotations[i] = 0.f;
    }

    std::ofstream zones_out(dir_str + "/zones.bin", std::ios::binary);
    assert(zones_out.is_open() && zones_out.good());

    zones_out.write((char *)&num_zones, sizeof(u32));
    zones_out.write((char *)zone_aabbs, sizeof(AABB) * num_zones);
    zones_out.write((char *)zone_rotations, sizeof(float) * num_zones);
  }
}
