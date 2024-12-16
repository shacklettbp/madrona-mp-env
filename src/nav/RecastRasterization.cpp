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

#include <math.h>
#include "Recast.h"

/// Check whether two bounding boxes overlap
///
/// @param[in]	aMin	Min axis extents of bounding box A
/// @param[in]	aMax	Max axis extents of bounding box A
/// @param[in]	bMin	Min axis extents of bounding box B
/// @param[in]	bMax	Max axis extents of bounding box B
/// @returns true if the two bounding boxes overlap.  False otherwise.
static bool overlapBounds(rcVec3 aMin, rcVec3 aMax, rcVec3 bMin, rcVec3 bMax)
{
  return aMin.x <= bMax.x && aMax.x >= bMin.x && aMin.y <= bMax.y &&
         aMax.y >= bMin.y && aMin.z <= bMax.z && aMax.z >= bMin.z;
}

/// Allocates a new span in the heightfield.
/// Use a memory pool and free list to minimize actual allocations.
///
/// @param[in]	heightfield		The heightfield
/// @returns A pointer to the allocated or re-used span memory.
static rcSpan *allocSpan(rcContext &context, rcHeightfield &heightfield)
{
  // If necessary, allocate new page and update the freelist.
  if (heightfield.freelist == NULL || heightfield.freelist->next == NULL) {
    // Create new page.
    // Allocate memory for the new pool.
    rcSpanPool *pool = (rcSpanPool *)context.tmpAlloc(sizeof(rcSpanPool));
    pool->next = heightfield.pools;
    heightfield.pools = pool;

    // Add new spans to the free list.
    rcSpan *freeList = heightfield.freelist;
    rcSpan *head = &pool->items[0];
    rcSpan *it = &pool->items[RC_SPANS_PER_POOL];
    do {
      --it;
      it->next = freeList;
      freeList = it;
    } while (it != head);
    heightfield.freelist = it;
  }

  // Pop item from the front of the free list.
  rcSpan *newSpan = heightfield.freelist;
  heightfield.freelist = heightfield.freelist->next;
  return newSpan;
}

/// Releases the memory used by the span back to the heightfield, so it can be
/// re-used for new spans.
/// @param[in]	heightfield		The heightfield.
/// @param[in]	span	A pointer to the span to free
static void freeSpan(rcHeightfield &heightfield, rcSpan *span)
{
  if (span == NULL) {
    return;
  }
  // Add the span to the front of the free list.
  span->next = heightfield.freelist;
  heightfield.freelist = span;
}

/// Adds a span to the heightfield.  If the new span overlaps existing spans,
/// it will merge the new span with the existing ones.
///
/// @param[in]	heightfield					Heightfield to
/// add spans to
/// @param[in]	x					The new span's column
/// cell x index
/// @param[in]	z					The new span's column
/// cell z index
/// @param[in]	min					The new span's minimum
/// cell index
/// @param[in]	max					The new span's maximum
/// cell index
/// @param[in]	areaID				The new span's area type ID
/// @param[in]	flagMergeThreshold	How close two spans maximum extents
/// need to be to merge area type IDs
static bool addSpan(rcContext &rc_ctx,
                    rcHeightfield &heightfield,
                    const int x,
                    const int y,
                    const unsigned short min,
                    const unsigned short max,
                    const unsigned char areaID,
                    const int flagMergeThreshold)
{
  // Create the new span.
  rcSpan *newSpan = allocSpan(rc_ctx, heightfield);
  if (newSpan == NULL) {
    return false;
  }
  newSpan->smin = min;
  newSpan->smax = max;
  newSpan->area = areaID;
  newSpan->next = NULL;

  const int columnIndex = x + y * heightfield.numCellsX;
  rcSpan *previousSpan = NULL;
  rcSpan *currentSpan = heightfield.spans[columnIndex];

  // Insert the new span, possibly merging it with existing spans.
  while (currentSpan != NULL) {
    if (currentSpan->smin > newSpan->smax) {
      // Current span is completely after the new span, break.
      break;
    }

    if (currentSpan->smax < newSpan->smin) {
      // Current span is completely before the new span.  Keep going.
      previousSpan = currentSpan;
      currentSpan = currentSpan->next;
    } else {
      // The new span overlaps with an existing span.  Merge them.
      if (currentSpan->smin < newSpan->smin) {
        newSpan->smin = currentSpan->smin;
      }
      if (currentSpan->smax > newSpan->smax) {
        newSpan->smax = currentSpan->smax;
      }

      // Merge flags.
      if (rcAbs((int)newSpan->smax - (int)currentSpan->smax) <=
          flagMergeThreshold) {
        // Higher area ID numbers indicate higher resolution priority.
        newSpan->area = rcMax(newSpan->area, currentSpan->area);
      }

      // Remove the current span since it's now merged with newSpan.
      // Keep going because there might be other overlapping spans that also
      // need to be merged.
      rcSpan *next = currentSpan->next;
      freeSpan(heightfield, currentSpan);
      if (previousSpan) {
        previousSpan->next = next;
      } else {
        heightfield.spans[columnIndex] = next;
      }
      currentSpan = next;
    }
  }

  // Insert new span after prev
  if (previousSpan != NULL) {
    newSpan->next = previousSpan->next;
    previousSpan->next = newSpan;
  } else {
    // This span should go before the others in the list
    newSpan->next = heightfield.spans[columnIndex];
    heightfield.spans[columnIndex] = newSpan;
  }

  return true;
}

bool rcAddSpan(rcContext &context,
               rcHeightfield &heightfield,
               const int x,
               const int y,
               const unsigned short spanMin,
               const unsigned short spanMax,
               const unsigned char areaID,
               const int flagMergeThreshold)
{
  if (!addSpan(context, heightfield, x, y, spanMin, spanMax, areaID,
               flagMergeThreshold)) {
    context.log(RC_LOG_ERROR, "rcAddSpan: Out of memory.");
    return false;
  }

  return true;
}

enum rcAxis { RC_AXIS_X = 0, RC_AXIS_Y = 1, RC_AXIS_Z = 2 };

/// Divides a convex polygon of max 12 vertices into two convex polygons
/// across a separating axis.
///
/// @param[in]	inVerts			The input polygon vertices
/// @param[in]	inVertsCount	The number of input polygon vertices
/// @param[out]	outVerts1		Resulting polygon 1's vertices
/// @param[out]	outVerts1Count	The number of resulting polygon 1 vertices
/// @param[out]	outVerts2		Resulting polygon 2's vertices
/// @param[out]	outVerts2Count	The number of resulting polygon 2 vertices
/// @param[in]	axisOffset		THe offset along the specified axis
/// @param[in]	axis			The separating axis
static void dividePoly(const rcVec3 *inVerts,
                       int inVertsCount,
                       rcVec3 *outVerts1,
                       int *outVerts1Count,
                       rcVec3 *outVerts2,
                       int *outVerts2Count,
                       float axisOffset,
                       rcAxis axis)
{
  assert(inVertsCount <= 12);

  // How far positive or negative away from the separating axis is each vertex.
  float inVertAxisDelta[12];
  for (int inVert = 0; inVert < inVertsCount; ++inVert) {
    inVertAxisDelta[inVert] = axisOffset - inVerts[inVert][axis];
  }

  int poly1Vert = 0;
  int poly2Vert = 0;
  for (int inVertA = 0, inVertB = inVertsCount - 1; inVertA < inVertsCount;
       inVertB = inVertA, ++inVertA) {
    // If the two vertices are on the same side of the separating axis
    bool sameSide =
        (inVertAxisDelta[inVertA] >= 0) == (inVertAxisDelta[inVertB] >= 0);

    if (!sameSide) {
      float s = inVertAxisDelta[inVertB] /
                (inVertAxisDelta[inVertB] - inVertAxisDelta[inVertA]);
      outVerts1[poly1Vert] =
          inVerts[inVertB] + (inVerts[inVertA] - inVerts[inVertB]) * s;

      outVerts2[poly2Vert] = outVerts1[poly1Vert];

      poly1Vert++;
      poly2Vert++;

      // add the inVertA point to the right polygon. Do NOT add points that are
      // on the dividing line since these were already added above
      if (inVertAxisDelta[inVertA] > 0) {
        outVerts1[poly1Vert] = inVerts[inVertA];
        poly1Vert++;
      } else if (inVertAxisDelta[inVertA] < 0) {
        outVerts1[poly2Vert] = inVerts[inVertA];
        poly2Vert++;
      }
    } else {
      // add the inVertA point to the right polygon. Addition is done even for
      // points on the dividing line
      if (inVertAxisDelta[inVertA] >= 0) {
        outVerts1[poly1Vert] = inVerts[inVertA];
        poly1Vert++;
        if (inVertAxisDelta[inVertA] != 0) {
          continue;
        }
      }
      outVerts2[poly2Vert] = inVerts[inVertA];
      poly2Vert++;
    }
  }

  *outVerts1Count = poly1Vert;
  *outVerts2Count = poly2Vert;
}

///	Rasterize a single triangle to the heightfield.
///
///	This code is extremely hot, so much care should be given to maintaining
/// maximum perf here.
///
/// @param[in] 	v0					Triangle vertex 0
/// @param[in] 	v1					Triangle vertex 1
/// @param[in] 	v2					Triangle vertex 2
/// @param[in] 	areaID				The area ID to assign to the
/// rasterized spans
/// @param[in] 	heightfield			Heightfield to rasterize into
/// @param[in] 	heightfieldBBMin	The min extents of the heightfield
/// bounding box
/// @param[in] 	heightfieldBBMax	The max extents of the heightfield
/// bounding box
/// @param[in] 	cellSize			The x and z axis size of a
/// voxel in the heightfield
/// @param[in] 	inverseCellSize		1 / cellSize
/// @param[in] 	inverseCellHeight	1 / cellHeight
/// @param[in] 	flagMergeThreshold	The threshold in which area flags will
/// be merged
/// @returns true if the operation completes successfully.  false if there was
/// an error adding spans to the heightfield.
static bool rasterizeTri(rcVec3 v0,
                         rcVec3 v1,
                         rcVec3 v2,
                         const unsigned char areaID,
                         rcHeightfield &heightfield,
                         rcContext &context,
                         rcVec3 heightfieldBBMin,
                         rcVec3 heightfieldBBMax,
                         const float cellSize,
                         const float inverseCellSize,
                         const float inverseCellHeight,
                         const int flagMergeThreshold)
{
  // Calculate the bounding box of the triangle.
  rcVec3 triBBMin = v0;
  triBBMin = rcVec3::min(triBBMin, v1);
  triBBMin = rcVec3::min(triBBMin, v2);

  rcVec3 triBBMax = v0;
  triBBMax = rcVec3::max(triBBMax, v1);
  triBBMax = rcVec3::max(triBBMax, v2);

  // If the triangle does not touch the bounding box of the heightfield, skip
  // the triangle.
  if (!overlapBounds(triBBMin, triBBMax, heightfieldBBMin, heightfieldBBMax)) {
    return true;
  }

  const int nx = heightfield.numCellsX;
  const int ny = heightfield.numCellsY;
  const float bz = heightfieldBBMax.z - heightfieldBBMin.z;

  // Calculate the footprint of the triangle on the grid's y-axis
  int y0 = (int)((triBBMin.y - heightfieldBBMin.y) * inverseCellSize);
  int y1 = (int)((triBBMax.y - heightfieldBBMin.y) * inverseCellSize);

  // use -1 rather than 0 to cut the polygon properly at the start of the tile
  y0 = rcClamp(y0, -1, ny - 1);
  y1 = rcClamp(y1, 0, ny - 1);

  // Clip the triangle into all grid cells it touches.
  rcVec3 buf[7 * 4];
  rcVec3 *in = buf;
  rcVec3 *inRow = buf + 7;
  rcVec3 *p1 = inRow + 7;
  rcVec3 *p2 = p1 + 7;

  in[0] = v0;
  in[1] = v1;
  in[2] = v2;

  int nvRow;
  int nvIn = 3;

  for (int y = y0; y <= y1; ++y) {
    // Clip polygon to row. Store the remaining polygon as well
    const float cellY = heightfieldBBMin.y + (float)y * cellSize;
    dividePoly(in, nvIn, inRow, &nvRow, p1, &nvIn, cellY + cellSize,
               RC_AXIS_Y);
    rcSwap(in, p1);

    if (nvRow < 3) {
      continue;
    }
    if (y < 0) {
      continue;
    }

    // find X-axis bounds of the row
    float minX = inRow[0].x;
    float maxX = inRow[0].x;
    for (int vert = 1; vert < nvRow; ++vert) {
      if (minX > inRow[vert].x) {
        minX = inRow[vert].x;
      }
      if (maxX < inRow[vert].x) {
        maxX = inRow[vert].x;
      }
    }
    int x0 = (int)((minX - heightfieldBBMin.x) * inverseCellSize);
    int x1 = (int)((maxX - heightfieldBBMin.x) * inverseCellSize);
    if (x1 < 0 || x0 >= nx) {
      continue;
    }
    x0 = rcClamp(x0, -1, nx - 1);
    x1 = rcClamp(x1, 0, nx - 1);

    int nv;
    int nv2 = nvRow;

    for (int x = x0; x <= x1; ++x) {
      // Clip polygon to column. store the remaining polygon as well
      const float cx = heightfieldBBMin.x + (float)x * cellSize;
      dividePoly(inRow, nv2, p1, &nv, p2, &nv2, cx + cellSize, RC_AXIS_X);
      rcSwap(inRow, p2);

      if (nv < 3) {
        continue;
      }
      if (x < 0) {
        continue;
      }

      // Calculate min and max of the span.
      float spanMin = p1[0].z;
      float spanMax = p1[0].z;
      for (int vert = 1; vert < nv; ++vert) {
        spanMin = rcMin(spanMin, p1[vert].z);
        spanMax = rcMax(spanMax, p1[vert].z);
      }
      spanMin -= heightfieldBBMin.z;
      spanMax -= heightfieldBBMin.z;

      // Skip the span if it's completely outside the heightfield bounding box
      if (spanMax < 0.0f) {
        continue;
      }
      if (spanMin > bz) {
        continue;
      }

      // Clamp the span to the heightfield bounding box.
      if (spanMin < 0.0f) {
        spanMin = 0;
      }
      if (spanMax > bz) {
        spanMax = bz;
      }

      // Snap the span to the heightfield height grid.
      unsigned short spanMinCellIndex = (unsigned short)rcClamp(
          (int)floorf(spanMin * inverseCellHeight), 0, RC_SPAN_MAX_HEIGHT);
      unsigned short spanMaxCellIndex = (unsigned short)rcClamp(
          (int)ceilf(spanMax * inverseCellHeight), (int)spanMinCellIndex + 1,
          RC_SPAN_MAX_HEIGHT);

      if (!addSpan(context, heightfield, x, y, spanMinCellIndex,
                   spanMaxCellIndex, areaID, flagMergeThreshold)) {
        return false;
      }
    }
  }

  return true;
}

bool rcRasterizeTriangles(rcContext &context,
                          const rcVec3 *verts,
                          const int /*nv*/,
                          const unsigned int *tris,
                          const unsigned char *triAreaIDs,
                          const int numTris,
                          rcHeightfield &heightfield,
                          const int flagMergeThreshold)
{
  rcScopedTimer timer(context, RC_TIMER_RASTERIZE_TRIANGLES);

  // Rasterize the triangles.
  const float inverseCellSize = 1.0f / heightfield.cs;
  const float inverseCellHeight = 1.0f / heightfield.ch;
  for (int triIndex = 0; triIndex < numTris; ++triIndex) {
    rcVec3 v0 = verts[tris[triIndex * 3 + 0]];
    rcVec3 v1 = verts[tris[triIndex * 3 + 1]];
    rcVec3 v2 = verts[tris[triIndex * 3 + 2]];
    if (!rasterizeTri(v0, v1, v2, triAreaIDs[triIndex], heightfield, context,
                      heightfield.bmin, heightfield.bmax, heightfield.cs,
                      inverseCellSize, inverseCellHeight,
                      flagMergeThreshold)) {
      context.log(RC_LOG_ERROR, "rcRasterizeTriangles: Out of memory.");
      return false;
    }
  }

  return true;
}

bool rcRasterizeTriangles(rcContext &context,
                          const rcVec3 *verts,
                          const int /*nv*/,
                          const unsigned short *tris,
                          const unsigned char *triAreaIDs,
                          const int numTris,
                          rcHeightfield &heightfield,
                          const int flagMergeThreshold)
{
  rcScopedTimer timer(context, RC_TIMER_RASTERIZE_TRIANGLES);

  // Rasterize the triangles.
  const float inverseCellSize = 1.0f / heightfield.cs;
  const float inverseCellHeight = 1.0f / heightfield.ch;
  for (int triIndex = 0; triIndex < numTris; ++triIndex) {
    rcVec3 v0 = verts[tris[triIndex * 3 + 0]];
    rcVec3 v1 = verts[tris[triIndex * 3 + 1]];
    rcVec3 v2 = verts[tris[triIndex * 3 + 2]];
    if (!rasterizeTri(v0, v1, v2, triAreaIDs[triIndex], heightfield, context,
                      heightfield.bmin, heightfield.bmax, heightfield.cs,
                      inverseCellSize, inverseCellHeight,
                      flagMergeThreshold)) {
      context.log(RC_LOG_ERROR, "rcRasterizeTriangles: Out of memory.");
      return false;
    }
  }

  return true;
}

bool rcRasterizeTriangles(rcContext &context,
                          const rcVec3 *verts,
                          const unsigned char *triAreaIDs,
                          const int numTris,
                          rcHeightfield &heightfield,
                          const int flagMergeThreshold)
{
  rcScopedTimer timer(context, RC_TIMER_RASTERIZE_TRIANGLES);

  // Rasterize the triangles.
  const float inverseCellSize = 1.0f / heightfield.cs;
  const float inverseCellHeight = 1.0f / heightfield.ch;
  for (int triIndex = 0; triIndex < numTris; ++triIndex) {
    rcVec3 v0 = verts[triIndex * 3 + 0];
    rcVec3 v1 = verts[triIndex * 3 + 1];
    rcVec3 v2 = verts[triIndex * 3 + 2];
    if (!rasterizeTri(v0, v1, v2, triAreaIDs[triIndex], heightfield, context,
                      heightfield.bmin, heightfield.bmax, heightfield.cs,
                      inverseCellSize, inverseCellHeight,
                      flagMergeThreshold)) {
      context.log(RC_LOG_ERROR, "rcRasterizeTriangles: Out of memory.");
      return false;
    }
  }

  return true;
}
