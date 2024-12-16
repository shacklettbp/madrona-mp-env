#pragma once

#include "consts.hpp"

namespace madronaMPEnv {

struct NavPoly {
  static constexpr inline i32 noAdjacent = 0xFFFF;

  u16 numVerts;
  u16 subTriOffset;
  u16 vertIndices[navMaxVertsPerPoly];
  u16 edgeAdjacency[navMaxVertsPerPoly];
};

struct NavAliasTableRow {
  float tau;
  u32 alias;
};

struct NavTriangleData {
  u16 poly;
  u16 triFanOffset;
};

struct Navmesh {
  i32 numVerts;
  i32 numPolys;
  i32 numTris;

  Vector3 verts[navMaxPolys * navMaxVertsPerPoly / 2];
  NavPoly polys[navMaxPolys];

  static constexpr inline i32 maxNumTris =
      navMaxPolys * (navMaxVertsPerPoly - 2);
  NavTriangleData subTris[maxNumTris];

  NavAliasTableRow polySampleAliasTable[navMaxPolys];
};

struct NavPolyPathEdge {
  u16 leftIDX;
  u16 rightIDX;
};

struct NavPolyPath {
  static constexpr inline u16 maxPathLen = 256;

  u16 pathLen;
  u16 startPoly;
  u16 endPoly;

  NavPolyPathEdge edges[maxPathLen];
};


}
