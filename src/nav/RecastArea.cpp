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
#include "RecastAlloc.h"

#include <cstring>  // for memcpy and memset

/// Sorts the given data in-place using insertion sort.
///
/// @param	data		The data to sort
/// @param	dataLength	The number of elements in @p data
static void insertSort(unsigned char *data, const int dataLength)
{
  for (int valueIndex = 1; valueIndex < dataLength; valueIndex++) {
    const unsigned char value = data[valueIndex];
    int insertionIndex;
    for (insertionIndex = valueIndex - 1;
         insertionIndex >= 0 && data[insertionIndex] > value;
         insertionIndex--) {
      // Shift over values
      data[insertionIndex + 1] = data[insertionIndex];
    }

    // Insert the value in sorted order.
    data[insertionIndex + 1] = value;
  }
}

bool rcErodeWalkableArea(rcContext &context,
                         const int erosionRadius,
                         rcCompactHeightfield &compactHeightfield)
{
  const int xSize = compactHeightfield.numCellsX;
  const int ySize = compactHeightfield.numCellsY;
  const int yStride = xSize;  // For readability

  rcScopedTimer timer(context, RC_TIMER_ERODE_AREA);

  unsigned char *distanceToBoundary = (unsigned char *)rcAlloc(
      sizeof(unsigned char) * compactHeightfield.spanCount, RC_ALLOC_TEMP);
  if (!distanceToBoundary) {
    context.log(RC_LOG_ERROR, "erodeWalkableArea: Out of memory 'dist' (%d).",
                compactHeightfield.spanCount);
    return false;
  }
  memset(distanceToBoundary, 0xff,
         sizeof(unsigned char) * compactHeightfield.spanCount);

  // Mark boundary cells.
  for (int y = 0; y < ySize; ++y) {
    for (int x = 0; x < xSize; ++x) {
      const rcCompactCell &cell = compactHeightfield.cells[x + y * yStride];
      for (int spanIndex = (int)cell.index,
               maxSpanIndex = (int)(cell.index + cell.count);
           spanIndex < maxSpanIndex; ++spanIndex) {
        if (compactHeightfield.areas[spanIndex] == RC_NULL_AREA) {
          distanceToBoundary[spanIndex] = 0;
          continue;
        }
        const rcCompactSpan &span = compactHeightfield.spans[spanIndex];

        // Check that there is a non-null adjacent span in each of the 4
        // cardinal directions.
        int neighborCount = 0;
        for (int direction = 0; direction < 4; ++direction) {
          const int neighborConnection = rcGetCon(span, direction);
          if (neighborConnection == RC_NOT_CONNECTED) {
            break;
          }

          const int neighborX = x + rcGetDirOffsetX(direction);
          const int neighborY = y + rcGetDirOffsetY(direction);
          const int neighborSpanIndex =
              (int)compactHeightfield.cells[neighborX + neighborY * yStride]
                  .index +
              neighborConnection;

          if (compactHeightfield.areas[neighborSpanIndex] == RC_NULL_AREA) {
            break;
          }
          neighborCount++;
        }

        // At least one missing neighbour, so this is a boundary cell.
        if (neighborCount != 4) {
          distanceToBoundary[spanIndex] = 0;
        }
      }
    }
  }

  unsigned char newDistance;

  // Pass 1
  for (int y = 0; y < ySize; ++y) {
    for (int x = 0; x < xSize; ++x) {
      const rcCompactCell &cell = compactHeightfield.cells[x + y * yStride];
      const int maxSpanIndex = (int)(cell.index + cell.count);
      for (int spanIndex = (int)cell.index; spanIndex < maxSpanIndex;
           ++spanIndex) {
        const rcCompactSpan &span = compactHeightfield.spans[spanIndex];

        if (rcGetCon(span, 0) != RC_NOT_CONNECTED) {
          // (-1,0)
          const int aX = x + rcGetDirOffsetX(0);
          const int aY = y + rcGetDirOffsetY(0);
          const int aIndex =
              (int)compactHeightfield.cells[aX + aY * xSize].index +
              rcGetCon(span, 0);
          const rcCompactSpan &aSpan = compactHeightfield.spans[aIndex];
          newDistance =
              (unsigned char)rcMin((int)distanceToBoundary[aIndex] + 2, 255);
          if (newDistance < distanceToBoundary[spanIndex]) {
            distanceToBoundary[spanIndex] = newDistance;
          }

          // (-1,-1)
          if (rcGetCon(aSpan, 3) != RC_NOT_CONNECTED) {
            const int bX = aX + rcGetDirOffsetX(3);
            const int bY = aY + rcGetDirOffsetY(3);
            const int bIndex =
                (int)compactHeightfield.cells[bX + bY * xSize].index +
                rcGetCon(aSpan, 3);
            newDistance =
                (unsigned char)rcMin((int)distanceToBoundary[bIndex] + 3, 255);
            if (newDistance < distanceToBoundary[spanIndex]) {
              distanceToBoundary[spanIndex] = newDistance;
            }
          }
        }
        if (rcGetCon(span, 3) != RC_NOT_CONNECTED) {
          // (0,-1)
          const int aX = x + rcGetDirOffsetX(3);
          const int aY = y + rcGetDirOffsetY(3);
          const int aIndex =
              (int)compactHeightfield.cells[aX + aY * xSize].index +
              rcGetCon(span, 3);
          const rcCompactSpan &aSpan = compactHeightfield.spans[aIndex];
          newDistance =
              (unsigned char)rcMin((int)distanceToBoundary[aIndex] + 2, 255);
          if (newDistance < distanceToBoundary[spanIndex]) {
            distanceToBoundary[spanIndex] = newDistance;
          }

          // (1,-1)
          if (rcGetCon(aSpan, 2) != RC_NOT_CONNECTED) {
            const int bX = aX + rcGetDirOffsetX(2);
            const int bY = aY + rcGetDirOffsetY(2);
            const int bIndex =
                (int)compactHeightfield.cells[bX + bY * xSize].index +
                rcGetCon(aSpan, 2);
            newDistance =
                (unsigned char)rcMin((int)distanceToBoundary[bIndex] + 3, 255);
            if (newDistance < distanceToBoundary[spanIndex]) {
              distanceToBoundary[spanIndex] = newDistance;
            }
          }
        }
      }
    }
  }

  // Pass 2
  for (int y = ySize - 1; y >= 0; --y) {
    for (int x = xSize - 1; x >= 0; --x) {
      const rcCompactCell &cell = compactHeightfield.cells[x + y * yStride];
      const int maxSpanIndex = (int)(cell.index + cell.count);
      for (int spanIndex = (int)cell.index; spanIndex < maxSpanIndex;
           ++spanIndex) {
        const rcCompactSpan &span = compactHeightfield.spans[spanIndex];

        if (rcGetCon(span, 2) != RC_NOT_CONNECTED) {
          // (1,0)
          const int aX = x + rcGetDirOffsetX(2);
          const int aY = y + rcGetDirOffsetY(2);
          const int aIndex =
              (int)compactHeightfield.cells[aX + aY * xSize].index +
              rcGetCon(span, 2);
          const rcCompactSpan &aSpan = compactHeightfield.spans[aIndex];
          newDistance =
              (unsigned char)rcMin((int)distanceToBoundary[aIndex] + 2, 255);
          if (newDistance < distanceToBoundary[spanIndex]) {
            distanceToBoundary[spanIndex] = newDistance;
          }

          // (1,1)
          if (rcGetCon(aSpan, 1) != RC_NOT_CONNECTED) {
            const int bX = aX + rcGetDirOffsetX(1);
            const int bY = aY + rcGetDirOffsetY(1);
            const int bIndex =
                (int)compactHeightfield.cells[bX + bY * xSize].index +
                rcGetCon(aSpan, 1);
            newDistance =
                (unsigned char)rcMin((int)distanceToBoundary[bIndex] + 3, 255);
            if (newDistance < distanceToBoundary[spanIndex]) {
              distanceToBoundary[spanIndex] = newDistance;
            }
          }
        }
        if (rcGetCon(span, 1) != RC_NOT_CONNECTED) {
          // (0,1)
          const int aX = x + rcGetDirOffsetX(1);
          const int aY = y + rcGetDirOffsetY(1);
          const int aIndex =
              (int)compactHeightfield.cells[aX + aY * xSize].index +
              rcGetCon(span, 1);
          const rcCompactSpan &aSpan = compactHeightfield.spans[aIndex];
          newDistance =
              (unsigned char)rcMin((int)distanceToBoundary[aIndex] + 2, 255);
          if (newDistance < distanceToBoundary[spanIndex]) {
            distanceToBoundary[spanIndex] = newDistance;
          }

          // (-1,1)
          if (rcGetCon(aSpan, 0) != RC_NOT_CONNECTED) {
            const int bX = aX + rcGetDirOffsetX(0);
            const int bY = aY + rcGetDirOffsetY(0);
            const int bIndex =
                (int)compactHeightfield.cells[bX + bY * xSize].index +
                rcGetCon(aSpan, 0);
            newDistance =
                (unsigned char)rcMin((int)distanceToBoundary[bIndex] + 3, 255);
            if (newDistance < distanceToBoundary[spanIndex]) {
              distanceToBoundary[spanIndex] = newDistance;
            }
          }
        }
      }
    }
  }

  const unsigned char minBoundaryDistance = (unsigned char)(erosionRadius * 2);

  for (int spanIndex = 0; spanIndex < compactHeightfield.spanCount;
       ++spanIndex) {
    if (distanceToBoundary[spanIndex] < minBoundaryDistance) {
      compactHeightfield.areas[spanIndex] = RC_NULL_AREA;
    }
  }

  rcFree(distanceToBoundary);

  return true;
}

bool rcMedianFilterWalkableArea(rcContext &context,
                                rcCompactHeightfield &compactHeightfield)
{
  const int xSize = compactHeightfield.numCellsX;
  const int ySize = compactHeightfield.numCellsY;
  const int yStride = xSize;  // For readability

  rcScopedTimer timer(context, RC_TIMER_MEDIAN_AREA);

  unsigned char *areas = (unsigned char *)rcAlloc(
      sizeof(unsigned char) * compactHeightfield.spanCount, RC_ALLOC_TEMP);
  if (!areas) {
    context.log(RC_LOG_ERROR,
                "medianFilterWalkableArea: Out of memory 'areas' (%d).",
                compactHeightfield.spanCount);
    return false;
  }
  memset(areas, 0xff, sizeof(unsigned char) * compactHeightfield.spanCount);

  for (int y = 0; y < ySize; ++y) {
    for (int x = 0; x < xSize; ++x) {
      const rcCompactCell &cell = compactHeightfield.cells[x + y * yStride];
      const int maxSpanIndex = (int)(cell.index + cell.count);
      for (int spanIndex = (int)cell.index; spanIndex < maxSpanIndex;
           ++spanIndex) {
        const rcCompactSpan &span = compactHeightfield.spans[spanIndex];
        if (compactHeightfield.areas[spanIndex] == RC_NULL_AREA) {
          areas[spanIndex] = compactHeightfield.areas[spanIndex];
          continue;
        }

        unsigned char neighborAreas[9];
        for (int neighborIndex = 0; neighborIndex < 9; ++neighborIndex) {
          neighborAreas[neighborIndex] = compactHeightfield.areas[spanIndex];
        }

        for (int dir = 0; dir < 4; ++dir) {
          if (rcGetCon(span, dir) == RC_NOT_CONNECTED) {
            continue;
          }

          const int aX = x + rcGetDirOffsetX(dir);
          const int aY = y + rcGetDirOffsetY(dir);
          const int aIndex =
              (int)compactHeightfield.cells[aX + aY * yStride].index +
              rcGetCon(span, dir);
          if (compactHeightfield.areas[aIndex] != RC_NULL_AREA) {
            neighborAreas[dir * 2 + 0] = compactHeightfield.areas[aIndex];
          }

          const rcCompactSpan &aSpan = compactHeightfield.spans[aIndex];
          const int dir2 = (dir + 1) & 0x3;
          const int neighborConnection2 = rcGetCon(aSpan, dir2);
          if (neighborConnection2 != RC_NOT_CONNECTED) {
            const int bX = aX + rcGetDirOffsetX(dir2);
            const int bY = aY + rcGetDirOffsetY(dir2);
            const int bIndex =
                (int)compactHeightfield.cells[bX + bY * yStride].index +
                neighborConnection2;
            if (compactHeightfield.areas[bIndex] != RC_NULL_AREA) {
              neighborAreas[dir * 2 + 1] = compactHeightfield.areas[bIndex];
            }
          }
        }
        insertSort(neighborAreas, 9);
        areas[spanIndex] = neighborAreas[4];
      }
    }
  }

  memcpy(compactHeightfield.areas, areas,
         sizeof(unsigned char) * compactHeightfield.spanCount);

  rcFree(areas);

  return true;
}

void rcMarkBoxArea(rcContext &context,
                   const rcVec3 boxMinBounds,
                   const rcVec3 boxMaxBounds,
                   unsigned char areaId,
                   rcCompactHeightfield &compactHeightfield)
{
  rcScopedTimer timer(context, RC_TIMER_MARK_BOX_AREA);

  const int xSize = compactHeightfield.numCellsX;
  const int ySize = compactHeightfield.numCellsY;
  const int yStride = xSize;  // For readability

  // Find the footprint of the box area in grid cell coordinates.
  int minX = (int)((boxMinBounds.x - compactHeightfield.bmin.x) /
                   compactHeightfield.cs);
  int minY = (int)((boxMinBounds.y - compactHeightfield.bmin.y) /
                   compactHeightfield.cs);
  int minZ = (int)((boxMinBounds.z - compactHeightfield.bmin.z) /
                   compactHeightfield.ch);
  int maxX = (int)((boxMaxBounds.x - compactHeightfield.bmin.x) /
                   compactHeightfield.cs);
  int maxY = (int)((boxMaxBounds.y - compactHeightfield.bmin.y) /
                   compactHeightfield.cs);
  int maxZ = (int)((boxMaxBounds.z - compactHeightfield.bmin.z) /
                   compactHeightfield.ch);

  // Early-out if the box is outside the bounds of the grid.
  if (maxX < 0) {
    return;
  }
  if (minX >= xSize) {
    return;
  }
  if (maxY < 0) {
    return;
  }
  if (minY >= ySize) {
    return;
  }

  // Clamp relevant bound coordinates to the grid.
  if (minX < 0) {
    minX = 0;
  }
  if (maxX >= xSize) {
    maxX = xSize - 1;
  }
  if (minY < 0) {
    minY = 0;
  }
  if (maxY >= ySize) {
    maxY = ySize - 1;
  }

  // Mark relevant cells.
  for (int y = minY; y <= maxY; ++y) {
    for (int x = minX; x <= maxX; ++x) {
      const rcCompactCell &cell = compactHeightfield.cells[x + y * yStride];
      const int maxSpanIndex = (int)(cell.index + cell.count);
      for (int spanIndex = (int)cell.index; spanIndex < maxSpanIndex;
           ++spanIndex) {
        rcCompactSpan &span = compactHeightfield.spans[spanIndex];

        // Skip if the span is outside the box extents.
        if ((int)span.z < minZ || (int)span.z > maxZ) {
          continue;
        }

        // Skip if the span has been removed.
        if (compactHeightfield.areas[spanIndex] == RC_NULL_AREA) {
          continue;
        }

        // Mark the span.
        compactHeightfield.areas[spanIndex] = areaId;
      }
    }
  }
}

// Brennan: RTCD 5.4.1
static bool triangleIsCCW2D(rcVec2 a, rcVec2 b, rcVec2 c)
{
  float det = ((a.x - c.x) * (b.y - c.y)) - ((a.y - c.y) * (b.x - c.x));

  return det > 0.f;
}

static bool pointInPoly2D(int n, const rcVec2 *v, const rcVec2 p)
{
  int low = 0, high = n;
  do {
    int mid = (low + high) / 2;
    if (triangleIsCCW2D(v[0], v[mid], p)) {
      low = high;
    } else {
      high = mid;
    }
  } while (low + 1 < high);

  if (low == 0 || high == n) {
    return false;
  }

  return triangleIsCCW2D(v[low], v[high], p);
}

void rcMarkConvexPolyArea(rcContext &context,
                          const rcVec2 *verts,
                          const int numVerts,
                          const float minZ,
                          const float maxZ,
                          unsigned char areaId,
                          rcCompactHeightfield &compactHeightfield)
{
  rcScopedTimer timer(context, RC_TIMER_MARK_CONVEXPOLY_AREA);

  const int xSize = compactHeightfield.numCellsX;
  const int ySize = compactHeightfield.numCellsY;
  const int yStride = xSize;  // For readability

  // Compute the bounding box of the polygon
  rcVec2 bmin = verts[0];
  rcVec2 bmax = verts[0];
  for (int i = 1; i < numVerts; ++i) {
    bmin = rcVec2::min(bmin, verts[i]);
    bmax = rcVec2::max(bmax, verts[i]);
  }

  // Compute the grid footprint of the polygon
  int minx =
      (int)((bmin.x - compactHeightfield.bmin.x) / compactHeightfield.cs);
  int miny =
      (int)((bmin.y - compactHeightfield.bmin.y) / compactHeightfield.cs);
  int minz = (int)((minZ - compactHeightfield.bmin.z) / compactHeightfield.ch);
  int maxx =
      (int)((bmax.x - compactHeightfield.bmin.x) / compactHeightfield.cs);
  int maxy =
      (int)((bmax.y - compactHeightfield.bmin.y) / compactHeightfield.cs);
  int maxz = (int)((maxZ - compactHeightfield.bmin.z) / compactHeightfield.ch);

  // Early-out if the polygon lies entirely outside the grid.
  if (maxx < 0) {
    return;
  }
  if (minx >= xSize) {
    return;
  }
  if (maxz < 0) {
    return;
  }
  if (miny >= ySize) {
    return;
  }

  // Clamp the polygon footprint to the grid
  if (minx < 0) {
    minx = 0;
  }
  if (maxx >= xSize) {
    maxx = xSize - 1;
  }
  if (miny < 0) {
    miny = 0;
  }
  if (maxy >= ySize) {
    maxy = ySize - 1;
  }

  // TODO: Optimize.
  for (int y = miny; y <= maxy; ++y) {
    for (int x = minx; x <= maxx; ++x) {
      const rcCompactCell &cell = compactHeightfield.cells[x + y * yStride];
      const int maxSpanIndex = (int)(cell.index + cell.count);
      for (int spanIndex = (int)cell.index; spanIndex < maxSpanIndex;
           ++spanIndex) {
        rcCompactSpan &span = compactHeightfield.spans[spanIndex];

        // Skip if span is removed.
        if (compactHeightfield.areas[spanIndex] == RC_NULL_AREA) {
          continue;
        }

        // Skip if y extents don't overlap.
        if ((int)span.z < minz || (int)span.z > maxz) {
          continue;
        }

        const rcVec2 point = {
            compactHeightfield.bmin.x +
                ((float)x + 0.5f) * compactHeightfield.cs,
            compactHeightfield.bmin.y +
                ((float)y + 0.5f) * compactHeightfield.cs,
        };

        if (pointInPoly2D(numVerts, verts, point)) {
          compactHeightfield.areas[spanIndex] = areaId;
        }
      }
    }
  }
}

void rcMarkCylinderArea(rcContext &context,
                        const rcVec3 position,
                        const float radius,
                        const float height,
                        unsigned char areaId,
                        rcCompactHeightfield &compactHeightfield)
{
  rcScopedTimer timer(context, RC_TIMER_MARK_CYLINDER_AREA);

  const int xSize = compactHeightfield.numCellsX;
  const int ySize = compactHeightfield.numCellsY;
  const int yStride = xSize;  // For readability

  // Compute the bounding box of the cylinder
  rcVec3 cylinderBBMin {
      .x = position.x - radius,
      .y = position.y - radius,
      .z = position.z,
  };

  rcVec3 cylinderBBMax {
      .x = position.x + radius,
      .y = position.y + radius,
      .z = position.z + height,
  };

  // Compute the grid footprint of the cylinder
  int minx = (int)((cylinderBBMin.x - compactHeightfield.bmin.x) /
                   compactHeightfield.cs);
  int miny = (int)((cylinderBBMin.y - compactHeightfield.bmin.y) /
                   compactHeightfield.cs);
  int minz = (int)((cylinderBBMin.z - compactHeightfield.bmin.z) /
                   compactHeightfield.ch);
  int maxx = (int)((cylinderBBMax.x - compactHeightfield.bmin.x) /
                   compactHeightfield.cs);
  int maxy = (int)((cylinderBBMax.y - compactHeightfield.bmin.y) /
                   compactHeightfield.cs);
  int maxz = (int)((cylinderBBMax.z - compactHeightfield.bmin.z) /
                   compactHeightfield.ch);

  // Early-out if the cylinder is completely outside the grid bounds.
  if (maxx < 0) {
    return;
  }
  if (minx >= xSize) {
    return;
  }
  if (maxy < 0) {
    return;
  }
  if (miny >= ySize) {
    return;
  }

  // Clamp the cylinder bounds to the grid.
  if (minx < 0) {
    minx = 0;
  }
  if (maxx >= xSize) {
    maxx = xSize - 1;
  }
  if (miny < 0) {
    miny = 0;
  }
  if (maxy >= ySize) {
    maxy = ySize - 1;
  }

  const float radiusSq = radius * radius;

  for (int y = miny; y <= maxy; ++y) {
    for (int x = minx; x <= maxx; ++x) {
      const rcCompactCell &cell = compactHeightfield.cells[x + y * yStride];
      const int maxSpanIndex = (int)(cell.index + cell.count);

      const float cellX = compactHeightfield.bmin.x +
                          ((float)x + 0.5f) * compactHeightfield.cs;
      const float cellY = compactHeightfield.bmin.y +
                          ((float)y + 0.5f) * compactHeightfield.cs;
      const float deltaX = cellX - position.x;
      const float deltaY = cellY - position.y;

      // Skip this column if it's too far from the center point of the
      // cylinder.
      if (rcSqr(deltaX) + rcSqr(deltaY) >= radiusSq) {
        continue;
      }

      // Mark all overlapping spans
      for (int spanIndex = (int)cell.index; spanIndex < maxSpanIndex;
           ++spanIndex) {
        rcCompactSpan &span = compactHeightfield.spans[spanIndex];

        // Skip if span is removed.
        if (compactHeightfield.areas[spanIndex] == RC_NULL_AREA) {
          continue;
        }

        // Mark if y extents overlap.
        if ((int)span.z >= minz && (int)span.z <= maxz) {
          compactHeightfield.areas[spanIndex] = areaId;
        }
      }
    }
  }
}
