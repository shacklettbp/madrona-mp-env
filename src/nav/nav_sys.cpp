#include "nav_sys.hpp"
#include "consts.hpp"

#include <madrona/context.hpp>
#include <madrona/memory.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/utils.hpp>
#include <madrona/geo.hpp>

#include "nav_impl.hpp"

namespace madronaMPEnv {

struct NavPriorityQueue {
  f32 *costs;
  i32 *heap;
  i32 *heapIndex;
  i32 heapSize;

  void add(i32 poly, f32 cost);
  i32 removeMin();
  void decreaseCost(i32 poly, f32 cost);
};

struct NavBFSState {
  i32 queue[navMaxPolys];
  bool visited[navMaxPolys];
};

struct DijkstrasState {
  float distances[navMaxPolys];
  Vector3 entryPoints[navMaxPolys];
  i32 heap[navMaxPolys];
  i32 heapIndex[navMaxPolys];
};

struct NavAStarQueueEntry {
  u16 prevPoly;
  u16 edgeVertIdxA;
  u16 edgeVertIdxB;
  
  float baseCost;
};

struct NavFindPathQuery {
  Entity resultEntity;
  Vector3 startPos;
  Vector3 endPos;
  i32 startPoly;
  i32 endPoly;

  i32 heapSize;

  float totalCosts[navMaxPolys];
  i32 heap[navMaxPolys];
  i32 heapIndex[navMaxPolys];

  NavAStarQueueEntry queueEntries[navMaxPolys];

};

struct NavFindPathQueryEntity : madrona::Archetype<NavFindPathQuery> {};

struct NavPolyPathEntity : public madrona::Archetype<NavPolyPath> {};

static inline i32 heapParent(i32 idx)
{
  return (idx - 1) / 2;
}

static inline i32 heapChildOffset(i32 idx)
{
  return 2 * idx + 1;
}

static inline void heapMoveUp(CountT moved_idx,
                              i32 moved_poly,
                              f32 moved_cost,
                              i32 *heap,
                              i32 *heap_index,
                              f32 *costs)
{
  while (moved_idx != 0) {
    CountT parent_idx = heapParent(moved_idx);
    i32 parent_poly = heap[parent_idx];
    if (costs[parent_poly] <= moved_cost) {
      break;
    }

    heap[moved_idx] = parent_poly;
    heap_index[parent_poly] = moved_idx;

    moved_idx = parent_idx;
  }

  heap[moved_idx] = moved_poly;
  heap_index[moved_poly] = moved_idx;
}

void NavPriorityQueue::add(i32 poly, f32 cost)
{
  costs[poly] = cost;

  i32 new_idx = heapSize++;
  heapMoveUp(new_idx, poly, cost, heap, heapIndex, costs);
}

i32 NavPriorityQueue::removeMin()
{
  i32 root_poly = heap[0];

  i32 moved_poly = heap[--heapSize];
  f32 moved_cost = costs[moved_poly];

  i32 moved_idx = 0;
  i32 child_offset;
  while ((child_offset = heapChildOffset(moved_idx)) < heapSize) {
    i32 child_idx = child_offset;
    i32 child_poly = heap[child_idx];
    f32 child_cost = costs[child_poly];
    {
      // Pick the lowest cost child
      i32 right_idx = child_idx + 1;
      if (right_idx < heapSize) {
        i32 right_poly = heap[right_idx];
        f32 right_cost = costs[right_poly];
        if (right_cost < child_cost) {
          child_idx = right_idx;
          child_poly = right_poly;
          child_cost = right_cost;
        }
      }
    }

    // moved_idx is now a valid position for moved_poly in the heap
    if (moved_cost < child_cost) {
      break;
    }

    heap[moved_idx] = child_poly;
    heapIndex[child_poly] = moved_idx;

    moved_idx = child_idx;
  }

  heap[moved_idx] = moved_poly;
  heapIndex[moved_poly] = moved_idx;

  heapIndex[root_poly] = navSentinel;
  return root_poly;
}

void NavPriorityQueue::decreaseCost(i32 poly, f32 cost)
{
  costs[poly] = cost;

  i32 cur_idx = heapIndex[poly];

  heapMoveUp(cur_idx, poly, cost, heap, heapIndex, costs);
}

template <typename Fn>
void navBFSFromPoly(const Navmesh &navmesh,
                    i32 start_poly,
                    NavBFSState &bfs_state,
                    Fn &&fn)
{
  ArrayQueue<i32> bfs_queue(bfs_state.queue, navmesh.numPolys);
  bool *visited = bfs_state.visited;

  utils::zeroN<bool>(visited, navmesh.numPolys);

  bfs_queue.add(start_poly);
  visited[start_poly] = true;

  while (!bfs_queue.isEmpty()) {
    i32 poly_idx = bfs_queue.remove();

    bool accept = fn(poly_idx);
    if (!accept) {
      continue;
    }

    const NavPoly poly = navmesh.polys[poly_idx];

    MADRONA_UNROLL
    for (i32 i = 0; i < navMaxVertsPerPoly; i++) {
      if (i >= (i32)poly.numVerts) {
        continue;
      }

      i32 adjacent_poly_idx = (i32)poly.edgeAdjacency[i];

      if (adjacent_poly_idx != NavPoly::noAdjacent &&
          !visited[adjacent_poly_idx]) {
        bfs_queue.add(adjacent_poly_idx);
        visited[adjacent_poly_idx] = true;
      }
    }
  }
}

template <typename Fn>
inline void navDijkstrasFromPoly(const Navmesh &navmesh,
                                 i32 start_poly,
                                 Vector3 start_pos,
                                 DijkstrasState &dijkstras_state,
                                 Fn &&fn)
{
  const i32 num_polys = navmesh.numPolys;

  float *distances = dijkstras_state.distances;

  NavPriorityQueue prio_queue {
    .costs = distances,
    .heap = dijkstras_state.heap,
    .heapIndex = dijkstras_state.heapIndex,
    .heapSize = 0,
  };
  utils::fillN<i32>(prio_queue.heapIndex, navSentinel, num_polys);
  utils::fillN<float>(distances, FLT_MAX, num_polys);

  Vector3 *entry_points = dijkstras_state.entryPoints;
  entry_points[start_poly] = start_pos;

  prio_queue.add(start_poly, 0.f);
  while (prio_queue.heapSize > 0) {
    i32 min_poly_idx = prio_queue.removeMin();
    Vector3 cur_pos = entry_points[min_poly_idx];
    f32 dist_so_far = distances[min_poly_idx];

    fn(min_poly_idx, cur_pos, dist_so_far);

    NavPoly min_poly = navmesh.polys[min_poly_idx];
    i32 num_poly_verts = (i32)min_poly.numVerts;

    MADRONA_UNROLL
    for (u16 i = 0; i < navMaxVertsPerPoly; i++) {
      if (i >= num_poly_verts) {
        break;
      }

      i32 adjacent = (i32)min_poly.edgeAdjacency[i];

      if (adjacent == NavPoly::noAdjacent) {
        continue;
      }

      Vector3 edge_midpoint;
      {
        u16 edge_next_vert = i + 1;
        if (edge_next_vert == min_poly.numVerts) {
          edge_next_vert = 0;
        }

        u16 a_idx = min_poly.vertIndices[i];
        u16 b_idx = min_poly.vertIndices[edge_next_vert];

        edge_midpoint = 0.5f * (navmesh.verts[a_idx] + navmesh.verts[b_idx]);
      }

      float dist_to_edge = cur_pos.distance(edge_midpoint);
      float new_dist = dist_so_far + dist_to_edge;
      float prev_dist = distances[adjacent];

      if (new_dist >= prev_dist) {
        continue;
      }

      entry_points[adjacent] = edge_midpoint;

      i32 prio_queue_idx = prio_queue.heapIndex[adjacent];
      if (prio_queue_idx == navSentinel) {
        prio_queue.add(adjacent, new_dist);
      } else {
        prio_queue.decreaseCost(adjacent, new_dist);
      }
    }
  }
}

static inline bool triangle2DIsCCW(Vector2 a, Vector2 b, Vector2 c)
{
  // RTCD 3.1.6.1

  Vector2 ca = a - c;
  Vector2 cb = b - c;

  float area2 = ca.x * cb.y - ca.y * cb.x;
  return area2 >= 0.f;
};

static NavPolyPath finalizePath(const Navmesh &navmesh, NavFindPathQuery &query)
{
  i32 start_poly_idx = query.startPoly;
  i32 end_poly_idx = query.endPoly;

  i32 path_len = 0;
  NavPolyPathEdge reverse_edges[NavPolyPath::maxPathLen];
  u16 cur_poly_idx = end_poly_idx;
  while (cur_poly_idx != start_poly_idx) {
    assert(path_len < NavPolyPath::maxPathLen);

    NavAStarQueueEntry queue_entry = query.queueEntries[cur_poly_idx];

    reverse_edges[path_len++] = {
      .leftIDX = queue_entry.edgeVertIdxA,
      .rightIDX = queue_entry.edgeVertIdxB,
    };

    cur_poly_idx = queue_entry.prevPoly;
  }

  NavPolyPath result;
  result.pathLen = path_len;
  result.startPoly = query.startPoly;
  result.endPoly = query.endPoly;

  Vector2 cur_pos = query.startPos.xy();
  for (i32 i = 0; i < path_len; i++) {
    NavPolyPathEdge edge = reverse_edges[path_len - 1 - i];
    {
      Vector3 left = navmesh.verts[edge.leftIDX];
      Vector3 right = navmesh.verts[edge.rightIDX];

      if (!triangle2DIsCCW(right.xy(), left.xy(), cur_pos)) {
        std::swap(edge.leftIDX, edge.rightIDX);
      }

      cur_pos = 0.5f * (left.xy() + right.xy());
    }

    result.edges[i] = edge;
  }

  return result;
}

static NavPolyPath navFindPath(const Navmesh &navmesh,
                               NavFindPathQuery &query)
{
  float *total_costs = query.totalCosts;
  NavAStarQueueEntry *queue_entries = query.queueEntries;

  NavPriorityQueue prio_queue {
    .costs = total_costs,
    .heap = query.heap,
    .heapIndex = query.heapIndex,
    .heapSize = query.heapSize,
  };

  i32 start_poly = query.startPoly;
  i32 end_poly = query.endPoly;

  Vector3 start_pos = query.startPos;
  Vector3 end_pos = query.endPos;

  const i32 num_polys = navmesh.numPolys;

  utils::fillN<i32>(prio_queue.heapIndex, navSentinel, num_polys);
  utils::fillN<float>(total_costs, FLT_MAX, num_polys);

  total_costs[start_poly] = 0;

  queue_entries[start_poly] = {
    .prevPoly = 0xFFFF,
    .edgeVertIdxA = 0,
    .edgeVertIdxB = 0,
    .baseCost = 0.f,
  };

  prio_queue.add(start_poly, 0.f);

  while (prio_queue.heapSize > 0) {
    i32 min_poly_idx = prio_queue.removeMin();

    if (min_poly_idx == end_poly) {
      // Min cost poly is end poly, we've made it
      return finalizePath(navmesh, query);
    }

    NavPoly min_poly = navmesh.polys[min_poly_idx];
    NavAStarQueueEntry min_entry = queue_entries[min_poly_idx];

    Vector3 cur_pos;
    {
      if (min_entry.prevPoly == 0xFFFF) {
        cur_pos = start_pos;
      } else {
        cur_pos = 0.5f * (navmesh.verts[min_entry.edgeVertIdxA] +
                          navmesh.verts[min_entry.edgeVertIdxB]);
      }
    }

    i32 num_poly_verts = (i32)min_poly.numVerts;

    MADRONA_UNROLL
    for (i32 i = 0; i < navMaxVertsPerPoly; i++) {
      if (i >= num_poly_verts) {
        break;
      }

      i32 adjacent = (i32)min_poly.edgeAdjacency[i];

      if (adjacent == NavPoly::noAdjacent) {
        continue;
      }

      NavAStarQueueEntry new_entry;
      {
        new_entry.edgeVertIdxA = min_poly.vertIndices[i];
        u16 next_i = i + 1;
        if (next_i == min_poly.numVerts) {
          next_i = 0;
        }
        new_entry.edgeVertIdxB = min_poly.vertIndices[next_i];
      }

      Vector3 edge_midpoint = 0.5f * (navmesh.verts[new_entry.edgeVertIdxA] +
                                      navmesh.verts[new_entry.edgeVertIdxB]);

      float dist_to_edge = cur_pos.distance(edge_midpoint);
      float new_base_cost = min_entry.baseCost + dist_to_edge;

      float new_total_cost;
      if (adjacent == end_poly) {
        new_base_cost += edge_midpoint.distance(end_pos);
        new_total_cost = new_base_cost;
      } else {
        float heuristic = 0.999f * edge_midpoint.distance(end_pos);
        new_total_cost = new_base_cost + heuristic;
      }

      float prev_total_cost = total_costs[adjacent];

      if (new_total_cost >= prev_total_cost) {
        continue;
      }

      total_costs[adjacent] = new_total_cost;

      new_entry.prevPoly = min_poly_idx;
      new_entry.baseCost = new_base_cost;

      queue_entries[adjacent] = new_entry;

      i32 prio_queue_idx = prio_queue.heapIndex[adjacent];
      if (prio_queue_idx == navSentinel) {
        prio_queue.add(adjacent, new_total_cost);
      } else {
        prio_queue.decreaseCost(adjacent, new_total_cost);
      }
    }
  }

  NavPolyPath not_found;
  not_found.pathLen = 0;
  not_found.startPoly = start_poly;
  not_found.endPoly = end_poly;

  return not_found;
}

inline void navFindPathsSystem(Context &ctx, NavFindPathQuery &query)
{
  const Navmesh &navmesh = ctx.singleton<Navmesh>();
  NavPolyPath result = navFindPath(navmesh, query);
  ctx.getDirect<NavPolyPath>(2, ctx.loc(query.resultEntity)) = result;
}

inline void navFollowPathsSystem(Context &ctx,
                                 Position pos,
                                 NavPathingState &pathing_state,
                                 NavPathingResult &result)
{
  if (pathing_state.pathEntity == Entity::none()) {
    result.numVerts = 0;
    return;
  }

  NavPolyPath &poly_path =
      ctx.getDirect<NavPolyPath>(2, ctx.loc(pathing_state.pathEntity));

  const i32 path_len = poly_path.pathLen;
  if (path_len == 0 && poly_path.startPoly != poly_path.endPoly) {
    result.numVerts = 0;
    return;
  }

  const Navmesh &navmesh = ctx.singleton<Navmesh>();

  // FIXME cache previous start
  if (pathing_state.curPathOffset < path_len) {
    NavPolyPathEdge edge = poly_path.edges[pathing_state.curPathOffset];
    Vector2 left = navmesh.verts[edge.leftIDX].xy();
    Vector2 right = navmesh.verts[edge.rightIDX].xy();

    if (triangle2DIsCCW(left, right, pos.xy())) {
      pathing_state.curPathOffset += 1;
    }
  }
  
  i32 start_idx = pathing_state.curPathOffset;

  i32 num_out_verts = 0;

  NavNearestResult cur_nearest = NavSystem::findNearestPoly(ctx, pos);
  if (cur_nearest.distance2 > 0.f) {
    result.pathVerts[num_out_verts++] = cur_nearest.point;
    pos = cur_nearest.point;
  }

  Vector2 funnel_pt = pos.xy();
  Vector3 funnel_left = pos;
  Vector3 funnel_right = pos;

  i32 left_restart_idx = start_idx;
  i32 right_restart_idx = start_idx;

  for (i32 path_idx = start_idx; path_idx < path_len + 1 &&
       num_out_verts < NavPathingResult::maxNumVerts; path_idx++) {
    Vector3 left;
    Vector3 right;
    {
      if (path_idx < path_len) {
        NavPolyPathEdge edge = poly_path.edges[path_idx];
        left = navmesh.verts[edge.leftIDX];
        right = navmesh.verts[edge.rightIDX];
      } else {
        left = pathing_state.goalPosition;
        right = pathing_state.goalPosition;
      }
    }

    if (triangle2DIsCCW(left.xy(), funnel_left.xy(), funnel_pt)) {
      if (triangle2DIsCCW(funnel_right.xy(), left.xy(), funnel_pt)) {
        funnel_left = left;
        left_restart_idx = path_idx;
      } else {
        result.pathVerts[num_out_verts++] = funnel_right;

        funnel_pt = funnel_right.xy();
        funnel_left = funnel_right;

        path_idx = right_restart_idx;
        left_restart_idx = right_restart_idx;
        continue; // Can't process right side, need to restart
      }
    }

    if (triangle2DIsCCW(funnel_right.xy(), right.xy(), funnel_pt)) {
      if (triangle2DIsCCW(right.xy(), funnel_left.xy(), funnel_pt)) {
        funnel_right = right;
        right_restart_idx = path_idx;
      } else {
        result.pathVerts[num_out_verts++] = funnel_left;

        funnel_pt = funnel_left.xy();
        funnel_right = funnel_left;

        path_idx = left_restart_idx;
        right_restart_idx = left_restart_idx;
      }
    } 
  }

  if (num_out_verts < NavPathingResult::maxNumVerts) {
    result.pathVerts[num_out_verts++] = pathing_state.goalPosition;
  }

  result.numVerts = num_out_verts;

#if 0
  // RTCD 5.4.1
  bool is_inside_poly;
  {
    // Do binary search over polygon vertices to find the fan triangle
    // (v[0], v[low], v[high]) the point p lies within the near sides of
    i32 low = 0, high = poly.numVerts;
    do {
      int mid = (low + high) / 2;
      if (triangleIsCCW(
          poly_verts[0], poly_verts[mid], pos)) {
        low = mid;
      } else {
        high = mid;
      }
    } while (low + 1 < high);

    // If point outside last (or first) edge, then it is not inside the n-gon
    if (low == 0 || high == poly.numVerts) {
      is_inside_poly = false;
    } else {
      // p inside the polygon if it is left of
      // the directed edge from v[low] to v[high]
      is_inside_poly = triangleIsCCW(poly_verts[low], poly_verts[high], pos);
    }
  }
  
  if (!is_inside_poly) {
    Vector3 poly_center = Vector3::zero();
    float inv_verts = 1.f / poly.numVerts;
    for (i32 i = 0; i < poly.numVerts; i++) {
      poly_center += inv_verts * poly_verts[i];
    }

    result.stepDirection = poly_center - pos;
    return;
  }
#endif
}

namespace NavSystem {

i32 findContainingPoly(Context &ctx, Vector3 p)
{
  const Navmesh &navmesh = ctx.singleton<Navmesh>();

  // FIXME BVH over triangles

  const i32 num_tris = navmesh.numTris;
  for (i32 tri_idx = 0; tri_idx < num_tris; tri_idx++) {
    NavTriangleData tri_data = navmesh.subTris[tri_idx];

    i32 tri_poly_idx = (i32)tri_data.poly;
    i32 tri_fan_offset = (i32)tri_data.triFanOffset;

    const NavPoly &poly = navmesh.polys[tri_poly_idx];

    Vector3 a = navmesh.verts[poly.vertIndices[0]];
    Vector3 b = navmesh.verts[poly.vertIndices[tri_fan_offset]];
    Vector3 c = navmesh.verts[poly.vertIndices[tri_fan_offset + 1]];

    a -= p;
    b -= p;
    c -= p;

    // RTCD 5.4.1

    f32 ab = dot(a, b);
    f32 ac = dot(a, c);
    f32 bc = dot(b, c);
    f32 cc = dot(c, c);

    if (bc * ac - cc * ab < 0.f) {
      continue;
    }

    f32 bb = dot(b, b);

    if (ab * bc - ac * bb < 0.f) {
      continue;
    }

    return tri_poly_idx;
  }

  return -1;
}

NavNearestResult findNearestPoly(Context &ctx, Vector3 p)
{
  const Navmesh &navmesh = ctx.singleton<Navmesh>();

  // FIXME BVH over triangles

  const i32 num_tris = navmesh.numTris;

  float min_dist2 = FLT_MAX;
  i32 closest_poly_idx = -1;
  Vector3 closest;
  for (i32 tri_idx = 0; tri_idx < num_tris; tri_idx++) {
    NavTriangleData tri_data = navmesh.subTris[tri_idx];

    i32 tri_poly_idx = (i32)tri_data.poly;
    i32 tri_fan_offset = (i32)tri_data.triFanOffset;

    const NavPoly &poly = navmesh.polys[tri_poly_idx];

    Vector3 a = navmesh.verts[poly.vertIndices[0]];
    Vector3 b = navmesh.verts[poly.vertIndices[tri_fan_offset]];
    Vector3 c = navmesh.verts[poly.vertIndices[tri_fan_offset + 1]];

    a -= p;
    b -= p;
    c -= p;

    Vector3 tri_closest =
        geo::triangleClosestPointToOrigin(a, b, c, b - a, c - a);

    float dist2 = tri_closest.length2();

    if (dist2 < min_dist2) {
      min_dist2 = dist2;
      closest_poly_idx = tri_poly_idx;
      closest = tri_closest + p;

      if (min_dist2 < 1e-6f) {
        // Caller can check if dist2 == 0.f to know if on navmesh
        min_dist2 = 0.f;
        break;
      }
    }
  }

  assert(closest_poly_idx != -1);

  return NavNearestResult {
    .point = closest,
    .distance2 = min_dist2,
    .poly = (u16)closest_poly_idx,
  };
}

NavSampleResult sampleRandomPointOnNavmesh(Context &ctx, RandKey rnd)
{
  const Navmesh &navmesh = ctx.singleton<Navmesh>();

  i32 num_tris = navmesh.numTris;

  RandKey tbl_row_rnd = rand::split_i(rnd, 0);
  RandKey alias_p_rnd = rand::split_i(rnd, 1);
  RandKey bary_rnd = rand::split_i(rnd, 2);

  uint32_t tbl_row_idx = rand::sampleI32(tbl_row_rnd, 0, num_tris);
  float alias_p = rand::sampleUniform(alias_p_rnd);

  NavAliasTableRow tbl_row = navmesh.polySampleAliasTable[tbl_row_idx];

  i32 tri_idx = alias_p < tbl_row.tau ? tbl_row_idx : tbl_row.alias;
  NavTriangleData sample_tri = navmesh.subTris[tri_idx];
  u16 rand_poly_idx = sample_tri.poly;

  const NavPoly &poly = navmesh.polys[rand_poly_idx];

  Vector3 a = navmesh.verts[poly.vertIndices[0]];
  Vector3 b = navmesh.verts[poly.vertIndices[sample_tri.triFanOffset]];
  Vector3 c = navmesh.verts[poly.vertIndices[sample_tri.triFanOffset + 1]];

  Vector2 uv = rand::sample2xUniform(bary_rnd);

  if (uv.x + uv.y > 1.f) {
    uv.x = 1.f - uv.x;
    uv.y = 1.f - uv.y;
  }

  float w = 1.f - uv.x - uv.y;

  Vector3 rnd_point = a * uv.x + b * uv.y + c * w;

  return NavSampleResult {
    .point = rnd_point,
    .poly = rand_poly_idx,
  };
}

NavPathingState queueFindPath(Context &ctx,
                              Vector3 start_pos,
                              i32 start_poly,
                              Vector3 end_pos,
                              i32 end_poly)
{
  Entity path_entity = ctx.makeEntity<NavPolyPathEntity>();

  Loc query_loc = ctx.makeTemporary<NavFindPathQueryEntity>();

  NavFindPathQuery &find_path_query =
      ctx.getDirect<NavFindPathQuery>(2, query_loc);

  find_path_query.resultEntity = path_entity;
  find_path_query.startPos = start_pos;
  find_path_query.endPos = end_pos;
  find_path_query.startPoly = start_poly;
  find_path_query.endPoly = end_poly;

  find_path_query.heapSize = 0;

  return NavPathingState {
    .pathEntity = path_entity,
    .curPathOffset = 0,
    .goalPosition = end_pos,
  };
}

void init(Context &ctx)
{
  Navmesh &navmesh = ctx.singleton<Navmesh>();
  navmesh.numPolys = 0;
  navmesh.numTris = 0;
}

void registerTypes(ECSRegistry &registry)
{
  registry.registerSingleton<Navmesh>();

  registry.registerComponent<NavFindPathQuery>();
  registry.registerArchetype<NavFindPathQueryEntity>();

  registry.registerComponent<NavPolyPath>();
  registry.registerArchetype<NavPolyPathEntity>();

  registry.registerComponent<NavPathingState>();
  registry.registerComponent<NavPathingResult>();
}

TaskGraphNodeID setupFindPathsTasks(TaskGraphBuilder &builder,
                                    Span<const TaskGraphNodeID> deps)
{
  auto find_paths = builder.addToGraph<
      ParallelForNode<Context, navFindPathsSystem, NavFindPathQuery>>(deps);

  return builder.addToGraph<ClearTmpNode<NavFindPathQueryEntity>>(
      {find_paths});
}

TaskGraphNodeID setupFollowPathTasks(TaskGraphBuilder &builder,
                                     Span<const TaskGraphNodeID> deps)
{
  auto follow_paths = builder.addToGraph<ParallelForNode<Context,
    navFollowPathsSystem,
      Position,
      NavPathingState,
      NavPathingResult
    >>(deps);

  return follow_paths;
}

}

}
