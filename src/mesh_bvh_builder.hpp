#include "mesh_bvh.hpp"

#include <madrona/importer.hpp>

namespace madronaMPEnv {

struct MeshBVHBuilder {
    static MeshBVH build(
        Span<const imp::SourceMesh> src_meshes);

};

}
