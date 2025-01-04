#pragma once

#include <cstdint>

namespace madronaMPEnv {

enum class SimFlags : uint32_t {
    Default                = 0,
    SpawnInMiddle          = 1 << 0,
    RandomizeHPMagazine    = 1 << 1,
    NavmeshSpawn           = 1 << 2,
    NoRespawn              = 1 << 3,
    StaggerStarts          = 1 << 4,
    EnableCurriculum       = 1 << 5,
    HardcodedSpawns        = 1 << 6,
    RandomFlipTeams        = 1 << 7,
    StaticFlipTeams        = 1 << 8,
    FullTeamPolicy         = 1 << 9,
    SimEvalMode            = 1 << 10,
};

inline SimFlags & operator|=(SimFlags &a, SimFlags b);
inline SimFlags operator|(SimFlags a, SimFlags b);
inline SimFlags & operator&=(SimFlags &a, SimFlags b);
inline SimFlags operator&(SimFlags a, SimFlags b);

}

#include "sim_flags.inl"
