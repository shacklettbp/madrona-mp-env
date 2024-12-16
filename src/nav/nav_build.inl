namespace madronaMPEnv {

bool NavmeshVoxelData::isOccupied(i32 cell_x, i32 cell_y, i32 cell_z)
{
  const i32 num_x_occupancy_cells = utils::divideRoundUp(
    (i32)gridNumCellsX, (i32)occupancyBitXDim);

  i32 x_idx = cell_x / occupancyBitXDim;
  i32 y_idx = cell_y;
  i32 z_idx = cell_z;
  i32 idx = (z_idx * gridNumCellsY + y_idx) * num_x_occupancy_cells + x_idx;

  i32 bit_offset = cell_x % occupancyBitXDim;

  u32 cur_bitfield = voxelOccupancy[idx];
  return (cur_bitfield & (1 << bit_offset)) != 0;
}

void NavmeshVoxelData::markOccupied(i32 cell_x, i32 cell_y, i32 cell_z)
{
  const i32 num_x_occupancy_cells = utils::divideRoundUp(
    (i32)gridNumCellsX, (i32)occupancyBitXDim);

  i32 x_idx = cell_x / occupancyBitXDim;
  i32 y_idx = cell_y;
  i32 z_idx = cell_z;
  i32 idx = (z_idx * gridNumCellsY + y_idx) * num_x_occupancy_cells + x_idx;

  i32 bit_offset = cell_x % occupancyBitXDim;

  u32 cur_bitfield = voxelOccupancy[idx];
  cur_bitfield |= 1 << bit_offset;

  voxelOccupancy[idx] = cur_bitfield;
}

}
