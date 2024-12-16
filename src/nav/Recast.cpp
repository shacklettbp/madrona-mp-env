// Recast source heavily modified by Brennan Shacklett 2024

//
// Copyright (c) 2009-2010 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#include "Recast.h"

#ifndef MADRONA_GPU_MODE
#include <stdarg.h>
#endif

void rcContext::log(const rcLogCategory category, const char *format, ...)
{
#ifndef MADRONA_GPU_MODE
  if (!m_logEnabled) {
    return;
  }
  static const int MSG_SIZE = 512;
  char msg[MSG_SIZE];
  va_list argList;
  va_start(argList, format);
  int len = vsnprintf(msg, MSG_SIZE, format, argList);
  if (len >= MSG_SIZE) {
    len = MSG_SIZE - 1;
    msg[MSG_SIZE - 1] = '\0';

    const char *errorMessage = "Log message was truncated";
    doLog(RC_LOG_ERROR, errorMessage, (int)strlen(errorMessage));
  }
  va_end(argList);
  doLog(category, msg, len);
#endif
}

void rcContext::doResetLog()
{
  // Defined out of line to fix the weak v-tables warning
}

void rcCalcGridSize(const float *minBounds,
                    const float *maxBounds,
                    const float cellSize,
                    int *sizeX,
                    int *sizeZ)
{
  *sizeX = (int)((maxBounds[0] - minBounds[0]) / cellSize + 0.5f);
  *sizeZ = (int)((maxBounds[2] - minBounds[2]) / cellSize + 0.5f);
}

bool rcInitHeightfield(rcContext &context,
                       rcHeightfield &heightfield,
                       int sizeX,
                       int sizeY,
                       const rcVec3 &minBounds,
                       const rcVec3 &maxBounds,
                       float cellSize,
                       float cellHeight)
{
  rcIgnoreUnused(context);

  heightfield.numCellsX = sizeX;
  heightfield.numCellsY = sizeY;
  heightfield.bmin = minBounds;
  heightfield.bmax = maxBounds;
  heightfield.cs = cellSize;
  heightfield.ch = cellHeight;
  const size_t num_span_array_bytes =
      sizeof(rcSpan *) * heightfield.numCellsX * heightfield.numCellsY;

  heightfield.spans =
      (rcSpan **)context.tmpAlloc(num_span_array_bytes);
  if (!heightfield.spans) {
    return false;
  }
  memset(heightfield.spans, 0, num_span_array_bytes);

  heightfield.pools = nullptr;
  heightfield.freelist = nullptr;

  return true;
}

static rcVec3 calcTriNormal(rcVec3 v0, rcVec3 v1, rcVec3 v2)
{
  rcVec3 e0 = v1 - v0;
  rcVec3 e1 = v2 - v0;

  rcVec3 faceNormal = cross(e0, e1);
  return normalize(faceNormal);
}

void rcMarkWalkableTriangles(rcContext &context,
                             const float walkableSlopeAngle,
                             const rcVec3 *verts,
                             const int numVerts,
                             const unsigned int *tris,
                             const int numTris,
                             unsigned char *triAreaIDs)
{
  rcIgnoreUnused(context);
  rcIgnoreUnused(numVerts);

  const float walkableThr = cosf(walkableSlopeAngle);

  for (int i = 0; i < numTris; ++i) {
    const unsigned int *tri = &tris[i * 3];
    rcVec3 norm = calcTriNormal(verts[tri[0]], verts[tri[1]], verts[tri[2]]);
    // Check if the face is walkable.
    if (norm.z > walkableThr) {
      triAreaIDs[i] = RC_WALKABLE_AREA;
    }
  }
}

int rcGetHeightFieldSpanCount(const rcHeightfield &heightfield)
{
  const int numCols = heightfield.numCellsX * heightfield.numCellsY;
  int spanCount = 0;
  for (int columnIndex = 0; columnIndex < numCols; ++columnIndex) {
    for (rcSpan *span = heightfield.spans[columnIndex]; span != NULL;
         span = span->next) {
      if (span->area != RC_NULL_AREA) {
        spanCount++;
      }
    }
  }
  return spanCount;
}

bool rcBuildCompactHeightfield(rcContext &context,
                               const int walkableHeight,
                               const int walkableClimb,
                               const rcHeightfield &heightfield,
                               rcCompactHeightfield &compactHeightfield)
{
  rcScopedTimer timer(context, RC_TIMER_BUILD_COMPACTHEIGHTFIELD);

  const int xSize = heightfield.numCellsX;
  const int ySize = heightfield.numCellsY;
  const int spanCount = rcGetHeightFieldSpanCount(heightfield);

  // Fill in header.
  compactHeightfield.numCellsX = xSize;
  compactHeightfield.numCellsY = ySize;
  compactHeightfield.spanCount = spanCount;
  compactHeightfield.walkableHeight = walkableHeight;
  compactHeightfield.walkableClimb = walkableClimb;
  compactHeightfield.maxRegions = 0;
  compactHeightfield.bmin = heightfield.bmin;
  compactHeightfield.bmax = heightfield.bmax;
  compactHeightfield.bmax.z += walkableHeight * heightfield.ch;
  compactHeightfield.cs = heightfield.cs;
  compactHeightfield.ch = heightfield.ch;
  compactHeightfield.cells = (rcCompactCell *)context.tmpAlloc(
      sizeof(rcCompactCell) * xSize * ySize);
  if (!compactHeightfield.cells) {
    context.log(RC_LOG_ERROR,
                "rcBuildCompactHeightfield: Out of memory 'chf.cells' (%d)",
                xSize * ySize);
    return false;
  }
  memset(compactHeightfield.cells, 0, sizeof(rcCompactCell) * xSize * ySize);
  compactHeightfield.spans = (rcCompactSpan *)context.tmpAlloc(
      sizeof(rcCompactSpan) * spanCount);
  if (!compactHeightfield.spans) {
    context.log(RC_LOG_ERROR,
                "rcBuildCompactHeightfield: Out of memory 'chf.spans' (%d)",
                spanCount);
    return false;
  }
  memset(compactHeightfield.spans, 0, sizeof(rcCompactSpan) * spanCount);
  compactHeightfield.areas = (unsigned char *)context.tmpAlloc(
      sizeof(unsigned char) * spanCount);
  if (!compactHeightfield.areas) {
    context.log(RC_LOG_ERROR,
                "rcBuildCompactHeightfield: Out of memory 'chf.areas' (%d)",
                spanCount);
    return false;
  }
  memset(compactHeightfield.areas, RC_NULL_AREA,
         sizeof(unsigned char) * spanCount);

  const int MAX_HEIGHT = 0xffff;

  // Fill in cells and spans.
  int currentCellIndex = 0;
  const int numColumns = xSize * ySize;
  for (int columnIndex = 0; columnIndex < numColumns; ++columnIndex) {
    const rcSpan *span = heightfield.spans[columnIndex];

    // If there are no spans at this cell, just leave the data to index=0,
    // count=0.
    if (span == NULL) {
      continue;
    }

    rcCompactCell &cell = compactHeightfield.cells[columnIndex];
    cell.index = currentCellIndex;
    cell.count = 0;

    for (; span != NULL; span = span->next) {
      if (span->area != RC_NULL_AREA) {
        const int bot = (int)span->smax;
        const int top = span->next ? (int)span->next->smin : MAX_HEIGHT;
        compactHeightfield.spans[currentCellIndex].z =
            (unsigned short)rcClamp(bot, 0, 0xffff);
        compactHeightfield.spans[currentCellIndex].h =
            (unsigned char)rcClamp(top - bot, 0, 0xff);
        compactHeightfield.areas[currentCellIndex] = span->area;
        currentCellIndex++;
        cell.count++;
      }
    }
  }

  return rcConnectCompactHeightfieldNeighbors(context, compactHeightfield);
}

bool rcConnectCompactHeightfieldNeighbors(
    rcContext &context,
    rcCompactHeightfield &compactHeightfield)
{
  int xSize = compactHeightfield.numCellsX;
  int ySize = compactHeightfield.numCellsY;
  int walkableHeight = compactHeightfield.walkableHeight;
  int walkableClimb = compactHeightfield.walkableClimb;

  // Find neighbour connections.
  const int MAX_LAYERS = RC_NOT_CONNECTED - 1;
  int maxLayerIndex = 0;
  const int yStride = xSize;  // for readability
  for (int y = 0; y < ySize; ++y) {
    for (int x = 0; x < xSize; ++x) {
      const rcCompactCell &cell = compactHeightfield.cells[x + y * yStride];
      for (int i = (int)cell.index, ni = (int)(cell.index + cell.count);
           i < ni; ++i) {
        rcCompactSpan &span = compactHeightfield.spans[i];

        for (int dir = 0; dir < 4; ++dir) {
          rcSetCon(span, dir, RC_NOT_CONNECTED);
          const int neighborX = x + rcGetDirOffsetX(dir);
          const int neighborY = y + rcGetDirOffsetY(dir);
          // First check that the neighbour cell is in bounds.
          if (neighborX < 0 || neighborY < 0 || neighborX >= xSize ||
              neighborY >= ySize) {
            continue;
          }

          // Iterate over all neighbour spans and check if any of the is
          // accessible from current cell.
          const rcCompactCell &neighborCell =
              compactHeightfield.cells[neighborX + neighborY * yStride];
          for (int k = (int)neighborCell.index,
                   nk = (int)(neighborCell.index + neighborCell.count);
               k < nk; ++k) {
            const rcCompactSpan &neighborSpan = compactHeightfield.spans[k];
            const int bot = rcMax(span.z, neighborSpan.z);
            const int top =
                rcMin(span.z + span.h, neighborSpan.z + neighborSpan.h);

            // Check that the gap between the spans is walkable,
            // and that the climb height between the gaps is not too high.
            if ((top - bot) >= walkableHeight &&
                rcAbs((int)neighborSpan.z - (int)span.z) <= walkableClimb) {
              // Mark direction as walkable.
              const int layerIndex = k - (int)neighborCell.index;
              if (layerIndex < 0 || layerIndex > MAX_LAYERS) {
                maxLayerIndex = rcMax(maxLayerIndex, layerIndex);
                continue;
              }
              rcSetCon(span, dir, layerIndex);
              break;
            }
          }
        }
      }
    }
  }

  if (maxLayerIndex > MAX_LAYERS) {
    context.log(RC_LOG_ERROR,
                "rcBuildCompactHeightfield: Heightfield has too many layers "
                "%d (max: %d)",
                maxLayerIndex, MAX_LAYERS);

    return false;
  }

  return true;
}
